# File Locks

> **Purpose**: Prevent merge conflicts by tracking which files are currently being modified.
> 
> Check this file before editing any source code. Add your lock before starting work.

---

## Currently Locked Files

| File | Locked By | Since | Task |
|------|-----------|-------|------|
| `train_specialized.py` | Antigravity | 2026-01-31 07:14 | WO-004, WO-005: Training pipeline |
| `models/specialized/` | Antigravity | 2026-01-31 07:14 | WO-004, WO-005: Model outputs |

---

## Lock Rules

### Before Editing a File
1. ✅ Check if the file is listed above
2. ✅ If locked by another AI, **do not edit** - pick a different task
3. ✅ Add your lock entry before making changes
4. ✅ Include the task/work order you're working on

### When Done with a File
1. ✅ Remove your lock entry
2. ✅ Update TASK_LOG.md with what you changed
3. ✅ Add context to HANDOFF_NOTES.md if relevant

### Lock Conflicts
If you urgently need a locked file:
1. 🚨 Notify the user
2. 🚨 User decides whether to force unlock
3. 🚨 Never override locks without explicit user permission

---

## Lock Entry Format

```markdown
| `path/to/file.py` | AI Name | YYYY-MM-DD HH:MM | Brief task description |
```

### Examples
```markdown
| `src/skills/periodogram.py` | Claude | 2026-01-31 10:00 | WO-001: BLS benchmarking |
| `train_real_data.py` | Antigravity | 2026-01-31 14:30 | WO-004: Expanding dataset |
```
