# LARUN - Antigravity (Gemini) Integration Guide

```
╔══════════════════════════════════════════════════════════════════════════╗
║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗                          ║
║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║                          ║
║     ██║     ███████║██████╔╝██║   ██║██╔██╗ ██║                          ║
║     ██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║                          ║
║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║                          ║
║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                          ║
║                                                                          ║
║     TinyML for Space Science - Antigravity Integration                   ║
║     Larun. × Astrodata                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## Overview

This document guides **Antigravity (Gemini)** on how to work on LARUN while coordinating with **Claude Code**. Both AIs share the same codebase and must follow coordination protocols.

---

## 🔴 CRITICAL: Multi-AI Coordination

> **IMPORTANT**: Before doing ANY work, follow the coordination protocol!

### Session Start Checklist

1. ✅ **Check `.coordination/TASK_LOG.md`**
   - Is another AI currently active?
   - If yes, notify user before proceeding

2. ✅ **Check `.coordination/FILE_LOCKS.md`**
   - Which files are currently locked?
   - Do NOT edit locked files

3. ✅ **Read `.coordination/HANDOFF_NOTES.md`**
   - What did the last session accomplish?
   - What context do you need?

4. ✅ **Log your session in `TASK_LOG.md`**
   ```markdown
   | Antigravity | [timestamp] | 🟢 Active | [what you're working on] |
   ```

5. ✅ **Check `.coordination/WORK_ORDERS.md`**
   - Pick up assigned or open tasks
   - Move task to "In Progress"

### Before Editing Files

```markdown
# Add to FILE_LOCKS.md:
| `path/to/file.py` | Antigravity | [timestamp] | [task description] |
```

### Session End Checklist

1. ✅ Update `HANDOFF_NOTES.md` with context for next session
2. ✅ Update `WORK_ORDERS.md` (complete tasks, add new ones)
3. ✅ Remove your entries from `FILE_LOCKS.md`
4. ✅ Move your `TASK_LOG.md` entry to history

---

## Project Structure

```
larun/
├── .coordination/           # 🔴 CHECK FIRST - Multi-AI coordination
│   ├── TASK_LOG.md          # Session tracking
│   ├── WORK_ORDERS.md       # Task queue
│   ├── FILE_LOCKS.md        # File ownership
│   └── HANDOFF_NOTES.md     # Context passing
├── docs/
│   ├── CLAUDE.md            # Claude Code instructions
│   ├── ANTIGRAVITY.md       # This file
│   └── research/            # Research documentation
├── skills/                  # Skill definitions (YAML)
├── src/                     # Source code
│   └── skills/              # Skill implementations
├── models/                  # Trained models
├── data/                    # Data cache
└── tests/                   # Unit tests
```

---

## Development Guidelines

### When Working on LARUN:

1. **Always check research docs first** - Before implementing any astronomy feature, read the relevant research .md file in `docs/research/`

2. **Follow skill patterns** - New capabilities should follow the skill definition format in `skills/`

3. **Use NASA APIs correctly** - Refer to `docs/research/NASA_DATA_SOURCES.md`

4. **Maintain TinyML focus** - Keep models small (<100KB) for edge deployment

5. **Test with real data** - Always validate against actual NASA data

### Priority Areas

| Priority | Area | Research Doc |
|----------|------|--------------| 
| 🔴 High | BLS Benchmarking | EXOPLANET_DETECTION.md |
| 🔴 High | False Positive Probability | EXOPLANET_DETECTION.md |
| 🔴 High | Gaia Integration | GAIA_INTEGRATION.md (to create) |
| 🟡 Medium | Training Data Expansion | NASA_DATA_SOURCES.md |
| 🟡 Medium | Model Architecture | TINYML_OPTIMIZATION.md |

---

## Code Style

```python
# LARUN Code Style
# - Use type hints
# - Include docstrings with examples
# - Follow astronomy naming conventions
# - Keep functions focused and small
# - Cache expensive computations

def analyze_light_curve(
    flux: np.ndarray,
    time: np.ndarray,
    target_name: str = "Unknown"
) -> AnalysisResult:
    """
    Analyze a light curve for transit signals.
    
    Args:
        flux: Normalized flux values
        time: Time array (BJD)
        target_name: Name of the target star
        
    Returns:
        AnalysisResult with detected signals
        
    Example:
        >>> result = analyze_light_curve(flux, time, "TIC 12345678")
        >>> print(f"Found {len(result.candidates)} candidates")
    """
    pass
```

---

## Model Constraints (TinyML)

| Constraint | Value | Reason |
|------------|-------|--------|
| Max Model Size | 100 KB | Microcontroller deployment |
| Max Parameters | 100,000 | Memory limits |
| Quantization | INT8 | Speed + size |
| Input Size | 1024 pts | Fixed processing |
| Inference Time | <10 ms | Real-time analysis |

---

## Workflow Reference

For detailed step-by-step coordination protocol, see:
`.agent/workflows/coordination.md`

---

*This document enables Antigravity (Gemini) to work effectively on LARUN while coordinating with Claude Code.*
