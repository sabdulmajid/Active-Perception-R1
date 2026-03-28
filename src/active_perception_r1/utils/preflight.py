from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

from active_perception_r1.utils.python_dev_headers import (
    PythonDevHeaderStatus,
    inspect_python_dev_headers,
)

try:
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover - packaging is expected in normal installs
    Requirement = None


@dataclass(frozen=True)
class GPUStatus:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_gpu_pct: int


@dataclass(frozen=True)
class DependencyStatus:
    module: str
    ok: bool
    version: str | None
    error: str | None


def _probe_vllm_runtime() -> Any:
    from vllm import LLM

    return LLM


KNOWN_IMPORT_PROBES: dict[str, Callable[[], Any]] = {
    "vllm": _probe_vllm_runtime,
}

KNOWN_COMPATIBILITY_REQUIREMENTS: dict[str, set[str]] = {
    "verl": {"vllm"},
    "vllm": {"torch", "torchaudio", "torchvision", "transformers"},
}


def parse_gpu_status_csv(text: str) -> list[GPUStatus]:
    statuses: list[GPUStatus] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Unexpected nvidia-smi output line: {raw_line}")
        statuses.append(
            GPUStatus(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mib=int(parts[2]),
                memory_used_mib=int(parts[3]),
                utilization_gpu_pct=int(parts[4]),
            )
        )
    return statuses


def query_gpu_statuses() -> list[GPUStatus]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is not available; cannot verify GPU state before running.")

    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_status_csv(proc.stdout)


def find_busy_gpus(
    statuses: list[GPUStatus],
    *,
    max_memory_used_mib: int = 2048,
    max_utilization_pct: int = 10,
) -> list[GPUStatus]:
    return [
        status
        for status in statuses
        if status.memory_used_mib > max_memory_used_mib or status.utilization_gpu_pct > max_utilization_pct
    ]


def format_gpu_summary(statuses: list[GPUStatus]) -> str:
    return "; ".join(
        f"gpu{status.index}={status.name} used={status.memory_used_mib}/{status.memory_total_mib}MiB util={status.utilization_gpu_pct}%"
        for status in statuses
    )


def require_idle_gpus(
    *,
    purpose: str,
    required_count: int = 1,
    statuses: list[GPUStatus] | None = None,
    allow_busy: bool = False,
    max_memory_used_mib: int = 2048,
    max_utilization_pct: int = 10,
) -> list[GPUStatus]:
    statuses = statuses or query_gpu_statuses()
    busy = find_busy_gpus(
        statuses,
        max_memory_used_mib=max_memory_used_mib,
        max_utilization_pct=max_utilization_pct,
    )
    idle = [status for status in statuses if status not in busy]

    if not allow_busy and len(idle) < required_count:
        raise RuntimeError(
            f"Refusing to start {purpose}: need {required_count} idle GPU(s), found {len(idle)}. "
            f"Current GPU state: {format_gpu_summary(statuses)}. "
            "Re-run with --allow-busy-gpu only if contention is intentional."
        )
    return statuses


def inspect_dependencies(
    modules: list[str],
    *,
    import_fn=importlib.import_module,
    probe_fns: dict[str, Callable[[], Any]] | None = None,
    distribution_fn=importlib_metadata.distribution,
    version_fn=importlib_metadata.version,
) -> list[DependencyStatus]:
    probe_fns = probe_fns or KNOWN_IMPORT_PROBES
    statuses: list[DependencyStatus] = []
    for module_name in modules:
        try:
            probe = probe_fns.get(module_name)
            if probe is not None:
                module = probe()
            else:
                module = import_fn(module_name)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if module_name == "vllm" and "undefined symbol" in str(exc):
                error += (
                    ". This usually indicates a torch/vllm ABI mismatch. "
                    "For verl 0.7.1, use a vllm release in verl's supported window "
                    "(<=0.12.0) with its matching torch/torchaudio/torchvision pins."
                )
            statuses.append(
                DependencyStatus(
                    module=module_name,
                    ok=False,
                    version=None,
                    error=error,
                )
            )
            continue
        statuses.append(
            DependencyStatus(
                module=module_name,
                ok=True,
                version=getattr(module, "__version__", None),
                error=None,
            )
        )

    imported_modules = {status.module for status in statuses if status.ok}
    statuses.extend(
        inspect_declared_compatibility(
            imported_modules,
            distribution_fn=distribution_fn,
            version_fn=version_fn,
        )
    )
    return statuses


def inspect_declared_compatibility(
    modules: set[str],
    *,
    distribution_fn=importlib_metadata.distribution,
    version_fn=importlib_metadata.version,
) -> list[DependencyStatus]:
    if Requirement is None:
        return []

    statuses: list[DependencyStatus] = []
    for owner_module, required_modules in KNOWN_COMPATIBILITY_REQUIREMENTS.items():
        if owner_module not in modules:
            continue

        try:
            distribution = distribution_fn(owner_module)
        except importlib_metadata.PackageNotFoundError:
            continue

        for raw_requirement in distribution.requires or []:
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            if requirement.name not in required_modules or requirement.name not in modules:
                continue

            try:
                installed_version = version_fn(requirement.name)
            except importlib_metadata.PackageNotFoundError:
                continue

            if requirement.specifier and not requirement.specifier.contains(installed_version, prereleases=True):
                statuses.append(
                    DependencyStatus(
                        module=f"{owner_module}->{requirement.name}",
                        ok=False,
                        version=installed_version,
                        error=(
                            f"Installed {requirement.name} {installed_version} does not satisfy "
                            f"{owner_module} requirement `{raw_requirement}`."
                        ),
                    )
                )
    return statuses


def require_dependencies(
    modules: list[str],
    *,
    purpose: str,
    install_hint: str,
) -> list[DependencyStatus]:
    statuses = inspect_dependencies(modules)
    missing = [status for status in statuses if not status.ok]
    if missing:
        details = ", ".join(f"{status.module} ({status.error})" for status in missing)
        raise RuntimeError(
            f"Missing Python dependencies for {purpose}: {details}. "
            f"Install them with `{install_hint}`."
        )
    return statuses


def require_python_dev_headers(
    *,
    purpose: str,
    env: dict[str, str] | None = None,
    system_include_dir: str | None = None,
) -> PythonDevHeaderStatus:
    status = inspect_python_dev_headers(env=env, system_include_dir=system_include_dir)
    if status is None:
        raise RuntimeError(
            f"Missing Python development headers for {purpose}. Triton/vLLM helper compilation requires `Python.h`, "
            "but it was not found in the interpreter include directory or current compiler include paths. "
            "Set `CPATH`/`C_INCLUDE_PATH` to a valid Python include directory or provide a vendored "
            "`libpythonX.Y-dev` package under `.vendor_runtime/`."
        )
    return status


def _main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for active-perception scripts.")
    parser.add_argument("--purpose", default="run")
    parser.add_argument("--required-gpus", type=int, default=1)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--max-gpu-memory-used-mib", type=int, default=2048)
    parser.add_argument("--max-gpu-utilization-pct", type=int, default=10)
    parser.add_argument(
        "--modules",
        default="",
        help="Comma-separated Python modules to import-check before execution.",
    )
    parser.add_argument(
        "--install-hint",
        default="pip install -e .",
        help="Installation command shown when module checks fail.",
    )
    parser.add_argument(
        "--require-python-dev-headers",
        action="store_true",
        help="Verify that `Python.h` is available to compiler subprocesses before launch.",
    )
    args = parser.parse_args()

    require_idle_gpus(
        purpose=args.purpose,
        required_count=args.required_gpus,
        allow_busy=args.allow_busy_gpu,
        max_memory_used_mib=args.max_gpu_memory_used_mib,
        max_utilization_pct=args.max_gpu_utilization_pct,
    )

    modules = [item.strip() for item in args.modules.split(",") if item.strip()]
    if modules:
        require_dependencies(
            modules,
            purpose=args.purpose,
            install_hint=args.install_hint,
        )

    if args.require_python_dev_headers:
        require_python_dev_headers(purpose=args.purpose)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except RuntimeError as exc:
        print(f"PRECHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
