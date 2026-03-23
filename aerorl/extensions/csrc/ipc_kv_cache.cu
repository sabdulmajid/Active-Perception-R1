/*
 * aerorl/extensions/csrc/ipc_kv_cache.cu
 *
 * CUDA IPC helpers for zero-copy KV-cache sharing between the vLLM rollout
 * process and the PyTorch training process on the same physical GPU.
 *
 * Build via setup.py (CUDAExtension).  Requires CUDA 12.4+, PyTorch ≥ 2.3.
 *
 * Public C++ / Python API
 * -----------------------
 *   bytes   export_ipc_handle(Tensor t)
 *       Returns a raw CUDA IPC memory handle (cudaIpcMemHandle_t, 64 bytes)
 *       serialised as a Python bytes object.
 *
 *   Tensor  import_ipc_handle(bytes handle,
 *                              list<int64_t> shape,
 *                              int  scalar_type,
 *                              int  device_index)
 *       Reconstructs a Tensor that aliases the memory exported by the peer
 *       process.  The returned Tensor MUST NOT outlive the originating allocation
 *       in the exporting process.
 *
 *   void    close_ipc_handle(Tensor t)
 *       Releases the IPC mapping opened by import_ipc_handle.  Safe to call
 *       multiple times (no-op after first call).
 *
 *   bool    is_ipc_supported()
 *       Returns true when CUDA IPC is available on the current device.
 *
 * Design notes
 * ------------
 * - A single call to cudaIpcGetMemHandle returns a handle that encodes the
 *   base address of the CUDA allocation (not the tensor offset).  We therefore
 *   also store and transfer the byte-offset of the tensor within its allocation.
 * - The importing process calls cudaIpcOpenMemHandle (lazy peer access) and
 *   wraps the raw device pointer in a torch::from_blob call with a custom
 *   deleter that invokes cudaIpcCloseMemHandle.
 * - Both processes must reside on the same node (same physical GPU or NVLink
 *   peer).  Cross-socket IPC requires prior cudaDeviceEnablePeerAccess.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <vector>

// ──────────────────────────────────────────────────────────────────────────
// Wire format: handle_bytes = [cudaIpcMemHandle_t (64B)] + [offset uint64 (8B)]
// Total = 72 bytes per tensor export.
// ──────────────────────────────────────────────────────────────────────────
static constexpr size_t kHandleBytes = sizeof(cudaIpcMemHandle_t);
static constexpr size_t kWireBytes   = kHandleBytes + sizeof(uint64_t);

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            throw std::runtime_error(                                          \
                std::string("CUDA error in " #expr ": ") +                    \
                cudaGetErrorString(_err));                                     \
        }                                                                      \
    } while (0)

// ──────────────────────────────────────────────────────────────────────────
// is_ipc_supported
// ──────────────────────────────────────────────────────────────────────────
bool is_ipc_supported() {
    int device = 0;
    cudaGetDevice(&device);
    int val = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &val, cudaDevAttrIpcEventSupport, device);
    return (err == cudaSuccess && val != 0);
}

// ──────────────────────────────────────────────────────────────────────────
// export_ipc_handle
// ──────────────────────────────────────────────────────────────────────────
py::bytes export_ipc_handle(torch::Tensor tensor) {
    TORCH_CHECK(tensor.is_cuda(), "export_ipc_handle: tensor must be on CUDA");
    TORCH_CHECK(tensor.is_contiguous(),
                "export_ipc_handle: tensor must be contiguous");

    // The IPC handle encodes the *allocation* base, not the tensor pointer.
    // We need to recover the base pointer of the allocation.
    void* data_ptr = tensor.data_ptr();

    // Get the allocation base via CUDA driver API (cuMemGetAddressRange).
    // Falls back to data_ptr when the full driver API is unavailable.
    void*  base_ptr  = nullptr;
    size_t alloc_size = 0;
    CUresult res = cuMemGetAddressRange(
        reinterpret_cast<CUdeviceptr*>(&base_ptr), &alloc_size,
        reinterpret_cast<CUdeviceptr>(data_ptr));

    if (res != CUDA_SUCCESS) {
        // Fallback: assume data_ptr IS the allocation base (common for
        // freshly allocated contiguous tensors).
        base_ptr = data_ptr;
    }

    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, base_ptr));

    uint64_t offset = static_cast<uint64_t>(
        reinterpret_cast<char*>(data_ptr) -
        reinterpret_cast<char*>(base_ptr));

    // Serialise: [handle (64B)] [offset (8B)]
    std::string wire(kWireBytes, '\0');
    std::memcpy(&wire[0], &handle, kHandleBytes);
    std::memcpy(&wire[kHandleBytes], &offset, sizeof(uint64_t));
    return py::bytes(wire);
}

// ──────────────────────────────────────────────────────────────────────────
// import_ipc_handle
// ──────────────────────────────────────────────────────────────────────────
torch::Tensor import_ipc_handle(
    py::bytes        handle_bytes,
    std::vector<int64_t> shape,
    int              scalar_type_int,
    int              device_index)
{
    std::string wire = handle_bytes.cast<std::string>();
    TORCH_CHECK(wire.size() == kWireBytes,
                "import_ipc_handle: unexpected wire size ", wire.size(),
                " (expected ", kWireBytes, ")");

    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, &wire[0], kHandleBytes);
    uint64_t offset = 0;
    std::memcpy(&offset, &wire[kHandleBytes], sizeof(uint64_t));

    void* base_ptr = nullptr;
    CUDA_CHECK(cudaIpcOpenMemHandle(
        &base_ptr, handle, cudaIpcMemLazyEnablePeerAccess));

    void* tensor_ptr = reinterpret_cast<char*>(base_ptr) + offset;

    auto options = torch::TensorOptions()
        .dtype(static_cast<c10::ScalarType>(scalar_type_int))
        .device(torch::kCUDA, device_index);

    // Custom deleter closes the IPC mapping when the Tensor is freed.
    return torch::from_blob(
        tensor_ptr, shape, options,
        [base_ptr](void*) {
            cudaIpcCloseMemHandle(base_ptr);
        });
}

// ──────────────────────────────────────────────────────────────────────────
// close_ipc_handle  (explicit, optional – the deleter fires automatically)
// ──────────────────────────────────────────────────────────────────────────
void close_ipc_handle(torch::Tensor tensor) {
    // Nothing to do here; the deleter registered in import_ipc_handle will
    // call cudaIpcCloseMemHandle when the Tensor storage is released.
    (void)tensor;
}

// ──────────────────────────────────────────────────────────────────────────
// pybind11 module
// ──────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "AeroRL CUDA IPC zero-copy KV-cache extension";

    m.def("export_ipc_handle", &export_ipc_handle,
          "Export a CUDA IPC memory handle for the given tensor.\n"
          "Returns 72 raw bytes (cudaIpcMemHandle_t + uint64 offset).",
          py::arg("tensor"));

    m.def("import_ipc_handle", &import_ipc_handle,
          "Import a CUDA IPC handle and return a Tensor aliasing the "
          "peer-process allocation.",
          py::arg("handle_bytes"),
          py::arg("shape"),
          py::arg("scalar_type"),
          py::arg("device_index") = 0);

    m.def("close_ipc_handle", &close_ipc_handle,
          "No-op helper kept for API compatibility; the memory handle is "
          "closed automatically when the imported Tensor is freed.",
          py::arg("tensor"));

    m.def("is_ipc_supported", &is_ipc_supported,
          "Return True if CUDA IPC is available on the current device.");
}
