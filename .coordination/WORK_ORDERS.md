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
| WO-014 | 🔴 High | Benchmark model on test set, create confusion matrix | Antigravity | `output/`, `docs/` |
| WO-016 | 🟡 Medium | Create inference examples notebook | Antigravity | `notebooks/inference_demo.ipynb` |
| WO-018 | 🟡 Medium | Create deployment guide (RPi, ESP32) | Antigravity | `docs/DEPLOYMENT.md` |
| WO-019 | 🟢 Lower | Edge device benchmarks (actual hardware) | Any | `docs/BENCHMARKS.md` |
| WO-020 | 🟢 Lower | Create model download/install script | Any | `scripts/download_model.py` |

---

## Vinveli-Vinoli-Vidhai Rebrand Sprint

**Goal**: Rename project from LARUN (name taken) to Vinveli-Vinoli-Vidhai
**Reference**: `docs/VINVELI_VINOLI_MIGRATION.md`

### Three-Tier Brand Structure
- **Vinveli** (விண்வெளி - Space) = Main ML system, Central orchestration, Training
- **Vinoli** (வெளிச்சம் - Light) = CLI interface, Communication layer, Data flow
- **Vidhai** (விதை - Seed) = Seeds planted on user systems to harvest knowledge

*"Plant a seed. Harvest the stars. Discover new worlds."*

| ID | Priority | Task | Assigned | Files Affected |
|----|:--------:|------|----------|----------------|
| WO-021 | 🔴 Critical | Phase 1: Foundation files | Claude | `pyproject.toml`, `vinoli.py`, `Dockerfile` |
| WO-022 | 🔴 Critical | Phase 2: Source code updates | Claude | `src/**/*.py` |
| WO-023 | 🔴 High | Phase 3: Documentation updates | Antigravity | `README.md`, `docs/*.md` |
| WO-024 | 🟡 Medium | Phase 4: Skills & config | Claude | `skills/*.yaml`, `config/` |
| WO-025 | 🟡 Medium | Phase 5: Notebooks rename | Antigravity | `notebooks/*.ipynb` |
| WO-026 | 🟡 Medium | Phase 6: Web/UI updates | Claude | `dashboard.html`, `src/reporter/` |
| WO-027 | 🟢 Lower | Phase 7: Tests & validation | Both | `tests/*.py` |
| WO-028 | 🟢 Lower | Phase 8: Repository & CI/CD | User + Both | `.github/workflows/` |
| WO-029 | 🟢 Lower | Create new logo assets | Any | `assets/vinoli-logo.*` |

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
| WO-012 | Claude | 2026-01-31 | Fix failing augmentation test | Fixed `flux < threshold` → `flux <= threshold` |
| WO-013 | Claude | 2026-01-31 | Create model training documentation | `docs/TRAINING_GUIDE.md` created |
| WO-015 | Claude | 2026-01-31 | Create Quick Start guide | `docs/QUICKSTART.md` created |
| WO-017 | Claude | 2026-01-31 | Add Docker containerization | `Dockerfile`, `docker-compose.yml`, `.dockerignore` |

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
