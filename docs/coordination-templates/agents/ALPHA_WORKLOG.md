# Agent ALPHA Worklog - Detection Engine

**Branch:** `claude/mvp-alpha-detection`
**Owner:** Claude Agent 1
**Status:** 🔵 Ready to Start

---

## My Responsibilities

- Transit detection model optimization
- BLS periodogram implementation
- Phase folding accuracy
- Vetting suite (odd-even, V-shape, secondary eclipse)
- Detection service layer for BETA

## My Files (Exclusive Write Access)

```
src/detection/
├── __init__.py
├── service.py          # DetectionService class
├── detector.py         # Transit detection logic
├── bls_engine.py       # BLS periodogram
├── phase_folder.py     # Phase folding
└── models.py           # Data classes

src/skills/
├── periodogram.py      # Existing (refactor)
├── vetting.py          # Existing (refactor)
└── transit_fitting.py  # New

tests/test_detection/
├── __init__.py
├── test_service.py
├── test_bls.py
├── test_vetting.py
└── test_phase_fold.py
```

---

## Daily Log

### Day 0 - Setup (Date: ______)

**Status:** Not started

**Tasks:**
- [ ] Create branch `claude/mvp-alpha-detection`
- [ ] Create `src/detection/` directory structure
- [ ] Review existing `src/skills/periodogram.py`
- [ ] Review existing `src/skills/vetting.py`
- [ ] Read MVP_INTERFACES.md

**Notes:**
- (Add notes here)

**Blockers:**
- None

---

### Day 1 (Date: ______)

**Status:** (🔵 Ready | 🟢 Active | 🟡 Waiting | 🔴 Blocked)

**Yesterday:**
- (What was completed)

**Today:**
- [ ] Task 1
- [ ] Task 2

**Blockers:**
- (List any blockers)

**Handoffs:**
- (List any handoffs to other agents)

---

## Interface I Provide

```python
# DetectionService - for BETA to consume

from src.detection import DetectionService

service = DetectionService()
result = await service.analyze("TIC 12345678")

# result.detection: bool
# result.confidence: float (0-1)
# result.period_days: float
# result.vetting: VettingResult
# result.phase_folded: PhaseFoldedData
```

**Interface Status:** 🟡 Draft → 🟢 Approved → 🔵 Implemented → ✅ Verified

---

## Dependencies I Need

| From | What | Status |
|------|------|--------|
| - | No external dependencies | ✅ |

---

## My Progress

| Week | Day | Task | Status |
|------|-----|------|--------|
| 1 | 1 | Create detection module structure | ⬜ |
| 1 | 1 | Define dataclasses | ⬜ |
| 1 | 2 | Refactor BLS | ⬜ |
| 1 | 2 | Create DetectionService interface | ⬜ |
| 1 | 3 | Phase folding accuracy | ⬜ |
| 1 | 3 | Refactor vetting | ⬜ |
| 1 | 4 | Logging | ⬜ |
| 1 | 4 | Detection CLI | ⬜ |
| 1 | 5 | Unit tests | ⬜ |
| 1 | 6 | Integration tests | ⬜ |
| 1 | 7 | Documentation | ⬜ |

**Legend:** ⬜ Not Started | 🔄 In Progress | ✅ Complete | ⛔ Blocked

---

## Notes

(Add any notes, decisions, or observations here)
