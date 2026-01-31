# Larun Self-Development Plan
## "Project Bootstrap" - Internal Tooling Roadmap

```
╔══════════════════════════════════════════════════════════════════════════╗
║  LARUN "BOOTSTRAP" INITIATIVE                                            ║
║  Objective: Enable Larun to develop, test, and improve itself.           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## 1. Vision: The Autopoietic System
For Larun to "self-develop" without external constraints, it must transition from a passive tool to an active **Agentic Platform**. This means Larun must possess the capabilities of a software engineer: reading code, writing code, testing logic, and managing versions—all contained within its own runtime.

## 2. The "Internal Developer" Skillset (DEV-Tier)
We will introduce a new Tier of skills specialized for software engineering.

| Skill ID | Skill Name | Description | Status |
|----------|------------|-------------|--------|
| `DEV-001` | **Introspection** | Ability to read, search, and map its own codebase. | 📋 Planned |
| `DEV-002` | **Test Runner** | Execute unit tests (`pytest`) and parse results internally. | 📋 Planned |
| `DEV-003` | **Git Operator** | Manage version control (branch, commit, diff) from within Larun. | 📋 Planned |
| `DEV-004` | **Skill Scaffolder** | Advanced generation of new Skill/Addon structures. | 📋 Planned |
| `DEV-005` | **Linter/Fixer** | Run static analysis (ruff/flake8) and apply auto-fixes. | 📋 Planned |
| `DEV-006` | **Doc Updater** | Automatically keep documentation in sync with code changes. | 📋 Planned |

## 3. Required Architecture

### 3.1 The "Mirror" (Introspection Engine)
Larun needs a way to understand its current state.
- **Requirement**: A `CodebaseIndex` class that maintains a map of all functions, classes, and their docstrings.
- **Implementation**: `src/skills/development/introspection.py`
- **Feature**: `larun dev map` to show project structure.

### 3.2 The "Workbench" (Sandboxed Execution)
To write code safely, Larun needs a sandbox.
- **Requirement**: A temporary workspace (`.larun_workspace/`) where generated code can be written and tested *before* being moved to `src/`.
- **Implementation**: `src/skills/development/sandbox.py`
- **Workflow**:
    1. Agent generates code → `.larun_workspace/new_skill.py`
    2. `DEV-002` runs tests on `.larun_workspace/`
    3. If pass → Move to `src/skills/`

### 3.3 The "Reviewer" (Automated QA)
- **Requirement**: An internal feedback loop.
- **Implementation**: Integration with `pytest` where the output is parsed into a structured JSON object that the Agent can read to understand *why* a test failed.

## 4. Implementation Roadmap

### Phase 1: The Foundation (Week 1)
- [ ] Create `src/skills/development/` directory.
- [ ] Implement `DEV-001 Introspection`: Simple file reading and `grep` capability.
- [ ] Implement `DEV-004 Skill Scaffolder`: Extend `codegen.py` to create full Python modules + YAML definitions.

### Phase 2: Safe Evolution (Week 2)
- [ ] Implement `DEV-002 Test Runner`: Wrapper around `pytest`.
- [ ] Create a "Self-Test" command: `larun self-test` that verifies the core system is healthy.

### Phase 3: Autonomy (Week 3)
- [ ] Implement `DEV-003 Git Operator`: `larun fit --save --commit "Added new model"`
- [ ] Create "Request" command: `larun develop "Add a new skill for asteroid detection"`
    - This triggers the Agent to:
        1. Read `DEV-001` (Understand where to put it)
        2. Run `DEV-004` (Create scaffolding)
        3. Edit files (Fill in logic)
        4. Run `DEV-002` (Verify)
        5. Run `DEV-003` (Commit)

## 5. Technical Requirements for "Outside Independence"

To fully decouple from external tools, **Larun** needs:
1.  **Built-in Library Management**: A way to install pip packages without leaving the Larun shell.
    - *Solutiaon*: `larun install <package>` wrapper.
2.  **Built-in Editor Interface**: Capability to perform search-and-replace edits on files (sed-like).
3.  **Process Management**: Ability to restart itself to reload new code.

## 6. Example "Self-Development" Workflow
*User Input*: "Larun, create a new skill to calculate the habitable zone of a star."

**Larun Internal Steps**:
1.  **Plan**: Uses `DEV-001` to check if `PLANET-005` exists. It doesn't.
2.  **Scaffold**: Uses `DEV-004` to create `src/skills/habitable.py` and `tests/test_habitable.py`.
3.  **Implement**: Writes the logic (Flux calculation) into the file.
4.  **Verify**: Runs `DEV-002` on `tests/test_habitable.py`.
5.  **Refine**: If test fails, reads error, patches `src/skills/habitable.py`, runs `DEV-002` again.
6.  **Deploy**: Reloads `skill_loader`.
7.  **Result**: Skill `PLANET-005` is now available for use.

## 7. Execution Timeline (Time-Based)

This schedule assumes a start date of **2026-01-31** (Day 0).

### Week 1: Foundation Building (Jan 31 - Feb 6)

| Date | Day | Objective | Key Tasks |
|------|-----|-----------|-----------|
| **Jan 31** | Sat | **Planning & Setup** | • Create `SELF_DEVELOPMENT_PLAN.md`<br>• Create `src/skills/development/`<br>• Define `skills/development.yaml` structure |
| **Feb 1** | Sun | **Core Introspection** | • Implement `DEV-001` (Introspection)<br>• Build `CodebaseIndex` class<br>• Add `larun dev map` command |
| **Feb 2** | Mon | **Scaffolding Tool** | • Implement `DEV-004` (Skill Scaffolder)<br>• Port logic from `codegen.py` to `DEV-004`<br>• Create `larun dev new-skill` command |
| **Feb 3** | Tue | **Scaffolding Refine** | • Add jinja2 templates for robust code generation<br>• Ensure generated code passes linting |
| **Feb 4** | Wed | **Testing Infra** | • Setup `tests/dev/` directory<br>• Write unit tests for `DEV-001` and `DEV-004`<br>• Ensure CI passes |
| **Feb 5** | Thu | **Manual Review** | • User review of generated code quality<br>• Adjust templates based on feedback |
| **Feb 6** | Fri | **Sandbox Init** | • Prototype `.larun_workspace` logic (fs isolation)<br>• Write specs for `DEV-002` (Test Runner) |

### Week 2: Test & Validation Loop (Feb 7 - Feb 13)

| Date | Day | Objective | Key Tasks |
|------|-----|-----------|-----------|
| **Feb 7** | Sat | **Test Runner v1** | • Implement `DEV-002` (Test Runner)<br>• Integrate `pytest` JSON output parsing |
| **Feb 8** | Sun | **Feedback Loop** | • Connect `DEV-004` output to `DEV-002`<br>• Auto-run tests after generation |
| **Feb 9** | Mon | **Security/Sandboxing** | • Harden `src/skills/development/sandbox.py`<br>• Ensure generated code can't delete sys32 |
| **Feb 10** | Tue | **Self-Test Cmd** | • Create `larun self-test` command<br>• Verify system integrity logic |
| **Feb 11** | Wed | **Linter Integration** | • Implement `DEV-005` basics (run ruff/flake8 via CLI)<br>• Add "Auto-Format" step to scaffold |
| **Feb 12** | Thu | **Bug Fixing** | • Resolve process management issues<br>• Fix reload-after-update bugs |
| **Feb 13** | Fri | **Milestone Demo** | • Demonstrate "Create Skill -> Test -> Deploy" loop manually |

### Week 3: Full Autonomy (Feb 14 - Feb 20)

| Date | Day | Objective | Key Tasks |
|------|-----|-----------|-----------|
| **Feb 14** | Sat | **Git Operations** | • Implement `DEV-003` (Git Operator)<br>• `larun dev commit` logic |
| **Feb 15** | Sun | **Agent Integration** | • Connect LLM/Agent logic to `larun develop`<br>• Implement the "Plan-Code-Verify" loop |
| **Feb 16** | Mon | **Doc Auto-Update** | • Implement `DEV-006` (Doc Updater)<br>• Auto-update `SKILLS_ROADMAP.md` when new skills add |
| **Feb 17** | Tue | **Package Mgmt** | • Implement internal `pip` wrapper (`larun install`)<br>• Dependency resolution check |
| **Feb 18** | Wed | **End-to-End Test** | • "Larun, create `PLANET-005`"<br>• Watch it build, test, and commit purely autonomously |
| **Feb 19** | Thu | **Refinement** | • Optimize prompt engineering for Agent<br>• Improve error recovery strategies |
| **Feb 20** | Fri | **Release v2.1** | • Tag release "Larun Self-Evolving"<br>• Deploy to main |

