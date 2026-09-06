# OpenCode Workspace Rules

## Developer Agent Constraints
- Do not execute external python monitoring tools or background loops.
- Prevent reading recursive standard input/output terminal logs directly during model calls.
- Fallback strictly to prompt-response behaviors.
