# Work Orders

> **Purpose**: Shared task queue for Antigravity and Claude Code to coordinate work.
> 
> Tasks can be assigned to a specific AI or left open for either to pick up.

---

## Open Tasks

| ID | Priority | Task | Preferred AI | Files Affected |
|----|:--------:|------|--------------|----------------|
| WO-001 | 🔴 High | Benchmark BLS against published Kepler results | Any | `src/skills/periodogram.py`, tests |
| WO-002 | 🔴 High | Implement centroid shift analysis (requires TPF data) | Any | `src/skills/vetting.py` |
| WO-003 | 🔴 High | Create FPP (False Positive Probability) calculator | Any | `src/skills/vetting.py` |
| WO-006 | 🟡 Medium | Add Gaia DR3 integration | Any | New `src/skills/gaia.py` |

---

## In Progress

| ID | AI | Started | Task |
|----|----|---------:|------|
| WO-004 | Antigravity | 2026-01-31 07:14 | Fetch 500+ confirmed exoplanets from NASA Archive |
| WO-005 | Antigravity | 2026-01-31 07:14 | Add K-fold cross-validation |

---

## Completed

| ID | AI | Date | Task | Notes |
|----|----|-----------:|------|-------|
| WO-SYS | Antigravity | 2026-01-31 | Setting up coordination system | Created `.coordination/`, workflow, docs |
| WO-008 | Antigravity | 2026-01-31 | Improve Data Augmentation | Added seeds, fixed transit expansion logic, added tests |

---

## How to Use

### Adding a New Task
```markdown
| WO-XXX | 🔴/🟡/🟢 | Task description | Any/Antigravity/Claude | affected files |
```

### Picking Up a Task
1. Move task from "Open Tasks" to "In Progress"
2. Add your AI name, start date
3. Add affected files to FILE_LOCKS.md

### Completing a Task
1. Move task from "In Progress" to "Completed"
2. Add completion date and notes
3. Remove file locks from FILE_LOCKS.md
4. Update HANDOFF_NOTES.md with context

### Priority Legend
- 🔴 **High**: Critical path, blocking other work
- 🟡 **Medium**: Important but not blocking
- 🟢 **Lower**: Nice to have, can wait
