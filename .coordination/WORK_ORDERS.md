# Work Orders

> **Purpose**: Shared task queue for Antigravity and Claude Code to coordinate work.
>
> Tasks can be assigned to a specific AI or left open for either to pick up.

---

## Production Release Sprint

**Goal**: Make LARUN model production-ready for users
**Timeline**: 2 weeks
**Success Criteria**: 90%+ accuracy, trained model files available, 100% tests passing

---

## Open Tasks

| ID | Priority | Task | Assigned | Files Affected |
|----|:--------:|------|----------|----------------|
| WO-010 | 🔴 Critical | Train production model (90%+ accuracy) | Antigravity | `train_real_data.py`, `models/` |
| WO-011 | 🔴 Critical | Save & export trained model files (.h5, .tflite, int8) | Antigravity | `models/real/` |
| WO-012 | 🔴 Critical | Fix failing augmentation test (transit depth) | Claude | `tests/test_augmentation.py` |
| WO-013 | 🔴 High | Create model training documentation | Claude | `docs/TRAINING_GUIDE.md` |
| WO-014 | 🔴 High | Benchmark model on test set, create confusion matrix | Antigravity | `output/`, `docs/` |
| WO-015 | 🟡 Medium | Create Quick Start guide (5 min to first detection) | Claude | `docs/QUICKSTART.md` |
| WO-016 | 🟡 Medium | Create inference examples notebook | Antigravity | `notebooks/inference_demo.ipynb` |
| WO-017 | 🟡 Medium | Add Docker containerization | Claude | `Dockerfile`, `docker-compose.yml` |
| WO-018 | 🟡 Medium | Create deployment guide (RPi, ESP32) | Antigravity | `docs/DEPLOYMENT.md` |
| WO-019 | 🟢 Lower | Edge device benchmarks (actual hardware) | Any | `docs/BENCHMARKS.md` |
| WO-020 | 🟢 Lower | Create model download/install script | Any | `scripts/download_model.py` |

---

## In Progress

| ID | AI | Started | Task | Status |
|----|----|---------:|------|--------|
| WO-004 | Antigravity | 2026-01-31 | Fetch 500+ confirmed exoplanets from NASA Archive | Data cached |
| WO-005 | Antigravity | 2026-01-31 | Add K-fold cross-validation | Implemented |

---

## Completed

| ID | AI | Date | Task | Notes |
|----|----|-----------:|------|-------|
| WO-SYS | Antigravity | 2026-01-31 | Setting up coordination system | Created `.coordination/`, workflow, docs |
| WO-008 | Antigravity | 2026-01-31 | Improve Data Augmentation | Added seeds, fixed transit expansion logic, added tests |
| WO-006 | Claude | 2026-01-31 | Add Gaia DR3 integration | `src/skills/gaia.py` complete |
| WO-007 | Claude | 2026-01-31 | Planet radius/habitability skill | `src/skills/planet.py` complete |
| WO-009 | Claude | 2026-01-31 | CI/CD Pipeline & BSL License | GitHub Actions, BSL-1.1 license |

---

## Suggested Assignment

### Antigravity (Model Training Focus)
| ID | Task |
|----|------|
| WO-010 | Train production model (90%+ accuracy) |
| WO-011 | Save & export trained model files |
| WO-014 | Benchmark model, create confusion matrix |
| WO-016 | Create inference examples notebook |
| WO-018 | Create deployment guide |

### Claude (Testing & Documentation Focus)
| ID | Task |
|----|------|
| WO-012 | Fix failing augmentation test |
| WO-013 | Create model training documentation |
| WO-015 | Create Quick Start guide |
| WO-017 | Add Docker containerization |

---

## Blocked Tasks

| ID | Task | Blocked By | Notes |
|----|------|------------|-------|
| WO-001 | Benchmark BLS against Kepler | WO-010 | Need trained model first |
| WO-002 | Centroid shift analysis | - | Requires TPF data access |
| WO-003 | FPP Calculator | WO-010 | Need validated model |

---

## Priority Legend
- 🔴 **Critical**: Must complete for production release
- 🟡 **Medium**: Important for good user experience
- 🟢 **Lower**: Nice to have, can follow release

---

## How to Use

### Picking Up a Task
1. Move task from "Open Tasks" to "In Progress"
2. Add your AI name, start date
3. Add affected files to FILE_LOCKS.md

### Completing a Task
1. Move task from "In Progress" to "Completed"
2. Add completion date and notes
3. Remove file locks from FILE_LOCKS.md
4. Update HANDOFF_NOTES.md with context
