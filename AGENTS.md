The Novel Research Direction: "Active Perception and Visual Self-Verification"
The Problem: Recent 2026 research indicates a critical flaw in current "thinking" VLMs: they suffer from passive perception. While text-only models successfully learn self-correction (the "Aha! moment") via RL, inference-time scaling in VLMs often fails because the models do not effectively integrate visual information into their self-verification process. Furthermore, standard Reinforcement Learning with Verifiable Rewards (RLVR) only rewards the final textual answer, which encourages "reward hacking" where the model hallucinates visual evidence to force a correct logical deduction.

The Solution: Build a VLM that utilizes Active Perception. Instead of just outputting text in its <think> block, the model is trained via GRPO to emit tool-calling tokens (e.g., <zoom_roi(x,y)> or <crop>) during its reasoning phase. If the model is uncertain about a visual detail, it actively "looks closer". You will optimize this using a hybrid reward function: a standard verifiable outcome reward for the final answer, plus a "Visual Perception Reward" that explicitly scores the consistency between the model's intermediate visual tool calls and the actual visual evidence.

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)  
- If something goes sideways, STOP and re-plan immediately - don't keep pushing  
- Use plan mode for verification steps, not just building  
- Write detailed specs upfront to reduce ambiguity  

---

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean  
- Offload research, exploration, and parallel analysis to subagents  
- For complex problems, throw more compute at it via subagents  
- One task per subagent for focused execution  

---

### 3. Self-Improvement Loop
- After ANY correction from the user: see if  
 `tasks/lessons.md` exists, if not, then create and update with the pattern  
- Write rules for yourself that prevent the same mistake  
- Ruthlessly iterate on these lessons until mistake rate drops  
- Review lessons at session start for relevant project  

---

### 4. Verification Before Done
- Never mark a task complete without proving it works  
- Diff behavior between main and your changes when relevant  
- Ask yourself: "Would a staff/principal engineer approve this?"  
- Run tests, check logs, demonstrate correctness  

---

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"  
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"  
- Skip this for simple, obvious fixes - don't over-engineer  
- Challenge your own work before presenting it  

---

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding  
- Point at logs, errors, failing tests - then resolve them  
- Zero context switching required from the user  
- Go fix failing CI, unit, integration tests without being told how  

---

## Task Management
1. **Plan First**: Write plan to `tasks/todo.md` with checkable items  
2. **Verify Plan**: Check in before starting implementation  
3. **Track Progress**: Mark items complete as you go  
4. **Explain Changes**: High-level summary at each step  
5. **Document Results**: Add review section to `tasks/todo.md`  
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections  

---

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code  
- **No Laziness**: Find root causes. No temporary fixes. Senior/staff developer standards


Framework Constraints
Orchestration: All RL training loops must be written using verl (Volcengine RL framework). Do not write raw PyTorch PPO/GRPO loops from scratch unless explicitly asked to modify the core algorithm.

Inference Backend: Use vLLM or SGLang for the rollout generation phase.

Models: Target 

Hardware & Memory Optimization Rules
We are operating on exactly two (2) NVIDIA RTX Pro 6000 GPUs, yielding 192GB of total VRAM. You must aggressively optimize for memory:

Reference Model Offloading: Always offload the reference model to CPU RAM during training (fsdp_config.param_offload=True).

Sequence Packing: Ensure use_remove_padding=True is enabled to maximize throughput for long Chain-of-Thought traces.

KL Divergence: We are using GRPO. Do not instantiate a separate Value Model. Apply the KL divergence penalty directly into the reward calculation to save VRAM.

Vision Caching: Ensure the vllm rollout engine is configured with disable_mm_preprocessor_cache=True to prevent the vision encoder's intermediate states from causing OOM errors during high-concurrency rollouts.

Reward Function Design
When writing reward functions, you must handle two streams of logic:

Outcome Reward: A hard binary check (e.g., checking if the parsed XML answer matches the ground truth).

Process/Perception Reward: A dense reward that evaluates the <think> trace. If the trace contains a <zoom_roi> tag, the reward function must validate that the coordinates are within bounds and penalize random, non-sensical cropping behavior.