# Contributing to LARUN

Thank you for your interest in contributing to LARUN! This document provides guidelines for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Multi-IDE Coordination](#multi-ide-coordination)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to advance exoplanet science.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/larun.git`
3. Add upstream remote: `git remote add upstream https://github.com/larun-engineering/larun.git`

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Run tests to verify setup
pytest tests/ -v
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feat/skill-name` - New skill or feature
- `fix/issue-description` - Bug fix
- `docs/topic` - Documentation
- `refactor/component` - Refactoring
- `test/what-testing` - Test additions

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

Examples:
```
feat(stellar): add luminosity class estimation
fix(gaia): handle missing parallax values
docs(api): add endpoint documentation
test(transit): add edge case tests for transit fitting
```

## Multi-IDE Coordination

LARUN supports multi-IDE development with AI assistants (Claude Code, Antigravity/Gemini, etc.).

### Before Starting Work

1. Check `.coordination/FILE_LOCKS.md` for locked files
2. Review `.coordination/TASK_LOG.md` for ongoing work
3. Check `.coordination/WORK_ORDERS.md` for assigned tasks

### While Working

1. Lock files you're editing:
   ```markdown
   ## Active Locks
   | File | Locked By | Since | Reason |
   |------|-----------|-------|--------|
   | src/skills/stellar.py | Claude Code | 2024-01-31 10:00 | Implementing STAR-004 |
   ```

2. Log your tasks in `TASK_LOG.md`

### After Completing Work

1. Release file locks
2. Update `HANDOFF_NOTES.md` with:
   - What was completed
   - Known issues
   - Suggested next steps
3. Update `skills/skills.yaml` if skills were modified

## Pull Request Process

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run checks locally**
   ```bash
   # Linting
   ruff check src/ tests/
   black --check src/ tests/

   # Type checking
   mypy src/

   # Tests
   pytest tests/ -v
   ```

3. **Create PR**
   - Use the PR template
   - Link related issues
   - Request review from CODEOWNERS

4. **Address review feedback**
   - Make changes in new commits (don't force-push during review)
   - Mark conversations as resolved

5. **Merge**
   - Squash and merge for clean history
   - Delete branch after merge

## Code Style

### Python Style

- **Line length**: 100 characters
- **Formatting**: Black + isort
- **Linting**: Ruff
- **Type hints**: Required for public APIs

### Documentation

- Docstrings for all public functions/classes
- Google-style docstrings preferred:
  ```python
  def calculate_radius(depth: float, stellar_radius: float) -> float:
      """Calculate planet radius from transit depth.

      Args:
          depth: Transit depth (fractional, 0-1)
          stellar_radius: Stellar radius in solar radii

      Returns:
          Planet radius in Earth radii

      Raises:
          ValueError: If depth is negative or > 1
      """
  ```

### Skill Development

When creating new skills:

1. Follow the skill template pattern in `src/skills/`
2. Include `SKILL_INFO` dict for registration
3. Add dataclasses for results
4. Include CLI convenience functions
5. Add tests in `tests/`
6. Update `skills/skills.yaml`
7. Update `src/skills/__init__.py` exports

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_stellar.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Skip slow tests
pytest tests/ -m "not slow"
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_<module>.py`
- Use descriptive test names: `test_classify_star_returns_g_type_for_solar_temp`
- Mock external API calls
- Use fixtures for common setup

### Test Categories

Mark tests appropriately:
```python
@pytest.mark.slow
def test_full_pipeline_integration():
    ...

@pytest.mark.integration
def test_gaia_api_query():
    ...
```

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Join discussions in the Discussions tab

---

Thank you for contributing to exoplanet discovery!
