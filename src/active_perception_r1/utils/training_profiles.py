from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingProfile:
    use_fused_kernels: bool
    actor_use_torch_compile: bool
    actor_fsdp_use_torch_compile: bool
    ref_use_torch_compile: bool
    ref_fsdp_use_torch_compile: bool
    rollout_gpu_memory_utilization: float


def recommend_training_profile(model_path: str) -> TrainingProfile:
    normalized = model_path.strip().lower()
    if "qwen2.5-vl" in normalized or "qwen2-vl" in normalized:
        # Qwen2.x-VL is currently more stable in this repo with engine compile disabled
        # and a slightly lower rollout reservation to leave room for colocated FSDP actors.
        return TrainingProfile(
            use_fused_kernels=False,
            actor_use_torch_compile=False,
            actor_fsdp_use_torch_compile=False,
            ref_use_torch_compile=False,
            ref_fsdp_use_torch_compile=False,
            rollout_gpu_memory_utilization=0.45,
        )

    return TrainingProfile(
        use_fused_kernels=False,
        actor_use_torch_compile=True,
        actor_fsdp_use_torch_compile=True,
        ref_use_torch_compile=True,
        ref_fsdp_use_torch_compile=True,
        rollout_gpu_memory_utilization=0.55,
    )


def _format_exports(profile: TrainingProfile) -> str:
    def _flag(value: bool) -> str:
        return "1" if value else "0"

    return "\n".join(
        [
            f"export ACTIVE_PERCEPTION_DEFAULT_USE_FUSED_KERNELS={_flag(profile.use_fused_kernels)}",
            f"export ACTIVE_PERCEPTION_DEFAULT_ACTOR_USE_TORCH_COMPILE={_flag(profile.actor_use_torch_compile)}",
            (
                "export ACTIVE_PERCEPTION_DEFAULT_ACTOR_FSDP_USE_TORCH_COMPILE="
                f"{_flag(profile.actor_fsdp_use_torch_compile)}"
            ),
            f"export ACTIVE_PERCEPTION_DEFAULT_REF_USE_TORCH_COMPILE={_flag(profile.ref_use_torch_compile)}",
            (
                "export ACTIVE_PERCEPTION_DEFAULT_REF_FSDP_USE_TORCH_COMPILE="
                f"{_flag(profile.ref_fsdp_use_torch_compile)}"
            ),
            (
                "export ACTIVE_PERCEPTION_DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION="
                f"{profile.rollout_gpu_memory_utilization}"
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend stable training-profile defaults.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--format",
        choices=("exports",),
        default="exports",
    )
    args = parser.parse_args()

    profile = recommend_training_profile(args.model_path)
    print(_format_exports(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
