---
description: Multi-IDE coordination protocol for Antigravity and Claude Code
---

# Coordination Workflow

This workflow ensures Antigravity (Gemini) and Claude Code can work on LARUN without conflicts.

## Session Start Protocol

// turbo-all

1. Read `.coordination/TASK_LOG.md` to check for active sessions
   - If another AI is active, notify user and wait for guidance
   
2. Read `.coordination/FILE_LOCKS.md` to see locked files
   - Do not modify any locked files
   
3. Read `.coordination/HANDOFF_NOTES.md` for context from last session
   - Understand what was done and what needs attention

4. Add your session to TASK_LOG.md:
   ```markdown
   | [Your AI Name] | [timestamp] | 🟢 Active | [initial task] |
   ```

5. Read `.coordination/WORK_ORDERS.md` to pick up tasks
   - Move your task from "Open Tasks" to "In Progress"
   - Add affected files to FILE_LOCKS.md

## During Session

6. Before editing any file, add it to FILE_LOCKS.md:
   ```markdown
   | `path/to/file` | [Your AI Name] | [timestamp] | [task description] |
   ```

7. Update TASK_LOG.md "Working On" column as you switch tasks

8. If you create new work orders, add them to WORK_ORDERS.md

## Session End Protocol

9. Update HANDOFF_NOTES.md with:
   - What you completed
   - What's still in progress
   - Any blockers or issues
   - Suggested next steps
   - Important context for next session

10. Update WORK_ORDERS.md:
    - Move completed tasks to "Completed" section
    - Add any new tasks discovered

11. Remove your entries from FILE_LOCKS.md

12. Update TASK_LOG.md:
    - Move your entry from "Active Session" to "Session History"
    - Include session duration and summary
