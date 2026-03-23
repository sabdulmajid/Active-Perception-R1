# Active-Perception-R1

Active-Perception-R1 is an open-source research scaffold for training VLMs to actively gather visual evidence before answering. The core idea is simple: instead of treating perception as a one-shot image encoding problem, we train a model to emit visual tool calls like `<zoom_roi ... />` inside its `<think>` trace, then reward it for choosing crops that actually cover task-relevant evidence.

This repository is built around `verl` and targets a dual-GPU workstation with `2 x NVIDIA RTX Pro 6000` cards. As of `March 23, 2026`, the repo contains tested custom reward modules, verl launch scripts tuned for this hardware class, sample task schemas, a synthetic self-driving policy lab for reward-sensitivity experiments, and a research write-up that is intentionally skeptical about what is solved versus what still requires custom work.

## Current Status

What is implemented now:

- A clean Python package under `src/active_perception_r1/`.
- `scripts/train_grpo_active_vision.sh` for multimodal GRPO with the requested memory knobs:
  - `actor_rollout_ref.model.use_remove_padding=True`
  - dynamic batch sizing for actor, ref log-prob, and rollout log-prob
  - `actor_rollout_ref.ref.fsdp_config.param_offload=True`
  - `+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True`
- A custom verl reward function that:
  - parses `<think>` traces for `<zoom_roi>` tags
  - validates ROIs
  - simulates a crop from task metadata
  - appends a structured `<image_crop ... />` token to an augmented context string
  - combines outcome reward with a visual perception/process reward
- `scripts/train_grpo_self_driving.sh` plus [`configs/self_driving_sample_task.json`](/pub7/neel2/active-perception/configs/self_driving_sample_task.json) for a safety-critical driving-flavored variant that rewards both grounded answers and safe high-level actions.
- `src/active_perception_r1/sim/self_driving_lab.py` and [`scripts/run_self_driving_policy_sweep.py`](/pub7/neel2/active-perception/scripts/run_self_driving_policy_sweep.py) for a synthetic policy lab that compares passive, detector-guided, and active-verification crop strategies before we spend real GRPO compute.
- Unit tests covering parser behavior, malformed tags, targeted crops, action rewards, safety penalties, and synthetic report generation.

What is not implemented yet:

- True online mid-generation image reinjection into the live rollout stream. Current stable verl GRPO examples are still one-turn rollouts; this repo simulates crop observations inside the reward path first, which is the safer research starting point.
- An end-to-end local training run in this workspace. The machine has the GPUs, but `torch`, `verl`, and `vllm` are not installed here yet, so the launch path has been validated at the shell/config level rather than by a full GRPO job.
- A real self-driving benchmark run. The driving lab in this repo is a synthetic evaluation harness for reward design and policy-selection behavior, not a claim of leaderboard performance on nuScenes, DriveLM, or DriveBench.

## Local Verification

Verified in this workspace on `March 23, 2026`:

- `nvidia-smi` confirms `2 x NVIDIA RTX Pro 6000 Blackwell` with about `97,887 MiB` VRAM each, idle and available.
- `bash -n scripts/train_grpo_active_vision.sh` passes.
- `bash -n scripts/train_grpo_self_driving.sh` passes.
- `PYTHONPATH=src python3 -m unittest discover -s tests -v` passes `12/12` tests.
- `PYTHONPATH=src python3 scripts/run_self_driving_policy_sweep.py` reproduces the checked-in sweep under [`results/self_driving_policy_sweep.md`](/pub7/neel2/active-perception/results/self_driving_policy_sweep.md) and [`results/self_driving_policy_sweep.json`](/pub7/neel2/active-perception/results/self_driving_policy_sweep.json) with `9,000` scene instances and `54,000` policy rollouts.
- A sample reward probe returns:
  - `score=1.3595`
  - `visual_perception_reward=0.3595`
  - `best_region_coverage=0.9272`

That means the reward logic works as designed on synthetic metadata, but it does not mean we have already demonstrated a trained zoom policy on a real benchmark.

## Research Findings

### What the literature strongly supports

- Explicit visual search helps when evidence is small, dense, high-resolution, or easy to miss. Good anchors here are [V* / SEAL](https://arxiv.org/abs/2312.14135), [CogCoM](https://arxiv.org/abs/2402.04236), [Chain-of-Focus](https://arxiv.org/abs/2505.15436), and [OpenThinkIMG](https://arxiv.org/abs/2505.08617).
- Many multimodal reasoning failures are partly perception failures, not purely reasoning failures. The strongest evidence comes from [ActiView](https://arxiv.org/abs/2410.04659), [MLLMs Know Where to Look](https://arxiv.org/abs/2502.17422), and [Math Blind / MATHGLANCE](https://arxiv.org/abs/2503.20745).
- RL can help when the reward is verifiable and local. The most relevant examples are [Active-O3](https://arxiv.org/abs/2505.21457), [ViCrit](https://arxiv.org/abs/2506.10128), and [GeoEyes](https://arxiv.org/abs/2602.14201).

### What is not well supported yet

- General claims that crop/zoom RL alone solves multimodal reasoning. Most strong results are narrow-domain, warm-started, or use dense geometric supervision.
- Final-answer-only reward as the whole learning signal. The better papers use box rewards, grounding signals, verifier loops, or proxy tasks.
- Claims that tool-calling behavior stays healthy automatically. [GeoEyes](https://arxiv.org/abs/2602.14201) explicitly warns about tool-usage homogenization and collapse.

### Design consequences for this repo

- Start with verifiable perception-heavy tasks, not open-ended free-form VQA.
- Keep the action space minimal: `zoom/crop`, `stop`, and optionally `ocr` later.
- Preserve global context and append local evidence rather than replacing the whole view.
- Compare against strong non-RL baselines like [Zoom-Refine](https://arxiv.org/abs/2506.01663) and [MLLMs Know Where to Look](https://arxiv.org/abs/2502.17422), not just against naive full-image inference.
- Treat true self-verification loops as stage 2, after evidence selection is working.

## Upstream `verl` Findings

As of `March 23, 2026`, upstream `verl` does directly support multimodal GRPO examples for Qwen-family VLMs, including [Qwen2.5-VL](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl-7b.sh), plus other newer VLM examples in the repo. That is the main reason this project is based on `verl` instead of a custom PPO loop.

Two caveats matter:

- Official documentation is a little stale in places. The repo examples are more current than some doc pages.
- The exact vLLM multimodal cache flag is version-sensitive. Upstream `verl` examples still use `disable_mm_preprocessor_cache=True`, while newer vLLM sources document the same behavior through `mm_processor_cache_gb=0`. This repo keeps the official `verl` knob in the launcher because that is what current verl VLM examples still use.

## Repository Layout

```text
.
├── configs/
│   ├── active_vision_sample_task.json
│   └── self_driving_sample_task.json
├── results/
│   ├── self_driving_policy_sweep.json
│   └── self_driving_policy_sweep.md
├── scripts/
│   ├── run_self_driving_policy_sweep.py
│   ├── train_grpo_active_vision.sh
│   └── train_grpo_self_driving.sh
├── src/active_perception_r1/
│   ├── envs/zoom_simulator.py
│   ├── rewards/active_vision_reward.py
│   ├── rewards/self_driving_reward.py
│   ├── sim/self_driving_lab.py
│   └── utils/trace_parser.py
├── tasks/
│   └── todo.md
└── tests/
    ├── test_active_vision_reward.py
    └── test_self_driving_extension.py
```

## Reward Design

The reward function expects examples to carry task metadata in `extra_info`, especially:

- `image_size`
- `relevant_regions`
- `requires_zoom`
- optional `answer_aliases`
- optional `expected_action`, `action_aliases`, and `safety_critical` for task-specific wrappers such as the self-driving setting

The scoring recipe is:

1. Parse the model output for `<zoom_roi ... />` calls inside `<think>`.
2. Reject malformed or nonsensical ROIs.
3. Simulate each crop against the provided relevant regions.
4. Append structured evidence tokens such as `<image_crop ... />` into an augmented context string.
5. Combine:
   - binary outcome reward from the final answer
   - dense process reward from valid, bounded, relevant zoom behavior

The current implementation is intentionally metadata-driven. That keeps the first research loop reproducible and verifiable before we add live image mutation inside rollout.

## Sample Task Schema

See [`configs/active_vision_sample_task.json`](/pub7/neel2/active-perception/configs/active_vision_sample_task.json). The important idea is that each example should include:

- a multimodal `prompt`
- `ground_truth`
- `data_source`
- `images`
- `extra_info.relevant_regions`

For research datasets, those relevant regions can come from:

- chart cell or legend boxes
- OCR spans
- diagram anchor boxes
- detection annotations
- synthetic crop labels generated from a teacher model or heuristic program

For a safety-critical variant, see [`configs/self_driving_sample_task.json`](/pub7/neel2/active-perception/configs/self_driving_sample_task.json), which adds:

- `expected_action`
- `action_aliases`
- `safety_critical`

## Training Launcher

The main launcher is [`scripts/train_grpo_active_vision.sh`](/pub7/neel2/active-perception/scripts/train_grpo_active_vision.sh).

Example:

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
./scripts/train_grpo_active_vision.sh
```

Safer first smoke test on this hardware:

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
N_RESPONSES=2 \
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
./scripts/train_grpo_active_vision.sh
```

Why the defaults look conservative:

- `ROLLOUT_TP_SIZE=1` is a workstation-oriented choice so rollout does not automatically lock both GPUs into one tensor-parallel shard.
- reference-model offload is enabled by default
- dynamic batch sizing is enabled for the expensive token-length-sensitive paths
- `use_remove_padding=True` is enabled for long `<think>` traces

The self-driving variant uses [`scripts/train_grpo_self_driving.sh`](/pub7/neel2/active-perception/scripts/train_grpo_self_driving.sh) and swaps in the task-specific reward wrapper from [`src/active_perception_r1/rewards/self_driving_reward.py`](/pub7/neel2/active-perception/src/active_perception_r1/rewards/self_driving_reward.py).

## Synthetic Self-Driving Lab

The synthetic driving lab is a cheaper way to answer a narrow but important question before running GRPO: if we change the reward from answer-only to grounded perception-aware reward, do the selected inspection policies actually shift toward evidence-seeking behavior?

Run it with:

```bash
PYTHONPATH=src python3 scripts/run_self_driving_policy_sweep.py
```

The checked-in run in [`results/self_driving_policy_sweep.md`](/pub7/neel2/active-perception/results/self_driving_policy_sweep.md) uses `120` scenes per task-condition-seed across seeds `[7, 11, 23]`. Key takeaways:

- `active_verify` is the strongest non-oracle policy by grounded accuracy at `0.620`.
- answer-only reward still selects `passive_cot` `0.640` of the time.
- grounded reward drives `passive_cot` selection to `0.000` and prefers `active_verify` over `passive_cot` pairwise `0.765` of the time.
- allowing a third crop still helps in this synthetic lab, but the jump from budget `2` to `3` is smaller than the jump from `1` to `2`.

This is still only a synthetic scaffold. Its value is in de-risking reward design and exposing failure modes early, not in proving real-world driving competence.

## Recommended First Experiments

1. Chart or OCR-heavy tasks with verifiable local evidence.
2. Compare four baselines:
   - full image only
   - training-free crop/refine
   - supervised tool traces
   - GRPO active perception
   - synthetic self-driving policy sweep as a reward-behavior sanity check
3. Track failure modes explicitly:
   - repeated center crops
   - over-zooming
   - no-stop behavior
   - correct answer without relevant evidence
   - relevant crop without answer improvement

## Honest Next Step

The next serious milestone is not “train the biggest model immediately.” It is:

1. install `torch`, `verl`, and `vllm`
2. run a 3B smoke experiment on a narrow verifiable benchmark
3. benchmark against non-RL crop baselines
4. only then scale to 7B or Kimi variants

That keeps the project scientifically honest and avoids mistaking infrastructure complexity for research progress.
