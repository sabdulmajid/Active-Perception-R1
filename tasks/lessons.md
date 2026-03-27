# Lessons

## 2026-03-27

- Before any benchmark or training launch, run `nvidia-smi` immediately beforehand and abort if the required GPUs are not idle. Do not assume earlier GPU checks are still valid.
- For compiled training stacks (`verl`, `vllm`, `torch`), verify binary and version compatibility explicitly. A top-level Python import is not enough when lazy imports can hide ABI failures until rollout startup.
- Keep `tasks/todo.md` synchronized with pushed commits so the task log matches repository history and does not leave already-finished work marked as pending.
