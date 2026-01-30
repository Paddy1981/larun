# Skill Development Guide - LARUN

## Overview

This document describes how to create new skills for LARUN, enabling Claude Code to extend LARUN's astronomical analysis capabilities.

---

## 1. Skill Architecture

### Skill Definition Structure

Each LARUN skill follows a consistent structure:

```yaml
# skills/my_skill.yaml

- id: "SKILL-001"
  name: "Human Readable Name"
  category: "category_name"
  tier: 1-6
  status: "active|planned|research"
  description: "What this skill does"
  command: "larun category action"
  
  dependencies:
    python: ["numpy", "astropy"]
    models: ["model_name.tflite"]
    data: ["catalog_name"]
  
  inputs:
    - name: "input_name"
      type: "string|float|int|filepath|enum"
      required: true|false
      default: "default_value"
      description: "What this input does"
  
  outputs:
    - type: "output_type"
      format: ["json", "csv", "fits"]
  
  examples:
    - "larun category action --input value"
```

### Skill Tiers

| Tier | Description | Complexity | Examples |
|------|-------------|------------|----------|
| 1 | Core Foundation | Simple | Data fetch, basic stats |
| 2 | Standard Analysis | Medium | BLS, classification |
| 3 | Advanced Analysis | High | Transit fitting, FPP |
| 4 | Multi-source | High | Cross-matching, fusion |
| 5 | Discovery | Very High | Multi-planet, exomoons |
| 6 | Research | Cutting-edge | Novel algorithms |

---

## 2. Creating a New Skill

### Step 1: Define the Skill

Create YAML definition in `skills/`:

```yaml
# skills/periodogram_skills.yaml

- id: "ANAL-010"
  name: "Lomb-Scargle Periodogram"
  category: "analysis"
  tier: 2
  status: "active"
  description: "Compute Lomb-Scargle periodogram for unevenly sampled data"
  command: "larun analyze lomb-scargle"
  
  dependencies:
    python: ["numpy", "scipy", "astropy"]
  
  inputs:
    - name: "time"
      type: "array"
      required: true
      description: "Time array"
    - name: "flux"
      type: "array"
      required: true
      description: "Flux array"
    - name: "min_period"
      type: "float"
      default: 0.1
      description: "Minimum period to search (days)"
    - name: "max_period"
      type: "float"
      default: 100.0
      description: "Maximum period to search (days)"
    - name: "samples_per_peak"
      type: "int"
      default: 5
      description: "Frequency resolution"
  
  outputs:
    - type: "periodogram"
      format: ["json", "csv"]
      fields: ["frequency", "power", "period", "fap"]
  
  metrics:
    - "best_period"
    - "best_power"
    - "false_alarm_probability"
  
  examples:
    - "larun analyze lomb-scargle --input lightcurve.csv --min-period 0.5 --max-period 50"
```

### Step 2: Implement the Skill

Create Python implementation in `src/skills/`:

```python
# src/skills/periodogram.py

"""
LARUN Skill: Lomb-Scargle Periodogram
=====================================
Compute periodogram for period detection in unevenly sampled data.

Skill ID: ANAL-010
Command: larun analyze lomb-scargle
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PeriodogramResult:
    """Result of periodogram analysis."""
    frequency: np.ndarray
    power: np.ndarray
    period: np.ndarray
    best_period: float
    best_power: float
    fap: float  # False alarm probability
    
    def to_dict(self):
        return {
            'best_period': self.best_period,
            'best_power': self.best_power,
            'fap': self.fap,
            'periods': self.period.tolist(),
            'powers': self.power.tolist()
        }


class LombScarglePeriodogram:
    """
    Compute Lomb-Scargle periodogram.
    
    Based on: Lomb (1976), Scargle (1982)
    
    Example:
        >>> lsp = LombScarglePeriodogram()
        >>> result = lsp.compute(time, flux)
        >>> print(f"Best period: {result.best_period:.2f} days")
    """
    
    def __init__(
        self,
        min_period: float = 0.1,
        max_period: float = 100.0,
        samples_per_peak: int = 5
    ):
        """
        Initialize periodogram parameters.
        
        Args:
            min_period: Minimum period to search (days)
            max_period: Maximum period to search (days)
            samples_per_peak: Frequency resolution (higher = finer)
        """
        self.min_period = min_period
        self.max_period = max_period
        self.samples_per_peak = samples_per_peak
    
    def compute(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None
    ) -> PeriodogramResult:
        """
        Compute Lomb-Scargle periodogram.
        
        Args:
            time: Time array (days)
            flux: Flux array (normalized)
            flux_err: Optional flux uncertainties
        
        Returns:
            PeriodogramResult with periods, powers, and best period
        """
        logger.info("Computing Lomb-Scargle periodogram...")
        
        # Input validation
        if len(time) != len(flux):
            raise ValueError("Time and flux arrays must have same length")
        
        if len(time) < 10:
            raise ValueError("Need at least 10 data points")
        
        # Remove NaN values
        mask = ~(np.isnan(time) | np.isnan(flux))
        time = time[mask]
        flux = flux[mask]
        
        # Frequency grid
        baseline = time.max() - time.min()
        df = 1.0 / (baseline * self.samples_per_peak)
        
        f_min = 1.0 / self.max_period
        f_max = 1.0 / self.min_period
        
        frequency = np.arange(f_min, f_max, df)
        
        # Compute periodogram using astropy
        try:
            from astropy.timeseries import LombScargle
            
            ls = LombScargle(time, flux, flux_err)
            power = ls.power(frequency)
            
            # False alarm probability
            fap = ls.false_alarm_probability(power.max())
            
        except ImportError:
            # Fallback to scipy
            from scipy.signal import lombscargle
            
            angular_freq = 2 * np.pi * frequency
            flux_centered = flux - np.mean(flux)
            power = lombscargle(time, flux_centered, angular_freq)
            power = power / (0.5 * len(flux) * np.var(flux))
            
            fap = self._estimate_fap(power.max(), len(frequency))
        
        # Convert to period
        period = 1.0 / frequency
        
        # Find best period
        best_idx = np.argmax(power)
        best_period = period[best_idx]
        best_power = power[best_idx]
        
        logger.info(f"Best period: {best_period:.4f} days (power={best_power:.4f})")
        
        return PeriodogramResult(
            frequency=frequency,
            power=power,
            period=period,
            best_period=best_period,
            best_power=best_power,
            fap=fap
        )
    
    def _estimate_fap(self, max_power: float, n_freq: int) -> float:
        """
        Estimate false alarm probability.
        
        Approximate using Baluev (2008) formula.
        """
        # Simplified FAP estimate
        fap = 1 - (1 - np.exp(-max_power))**n_freq
        return min(fap, 1.0)


# CLI interface
def cli_lomb_scargle(args):
    """CLI entry point for Lomb-Scargle skill."""
    from ..io import load_lightcurve, save_result
    
    # Load data
    lc = load_lightcurve(args.input)
    
    # Compute periodogram
    lsp = LombScarglePeriodogram(
        min_period=args.min_period,
        max_period=args.max_period,
        samples_per_peak=args.samples_per_peak
    )
    
    result = lsp.compute(lc.time, lc.flux, lc.flux_err)
    
    # Save result
    save_result(result.to_dict(), args.output, format=args.format)
    
    print(f"Best period: {result.best_period:.4f} days")
    print(f"FAP: {result.fap:.2e}")
    
    return result


# Registration
SKILL_INFO = {
    'id': 'ANAL-010',
    'name': 'Lomb-Scargle Periodogram',
    'command': 'analyze lomb-scargle',
    'function': cli_lomb_scargle
}
```

### Step 3: Add Tests

Create test file in `tests/`:

```python
# tests/test_periodogram.py

import pytest
import numpy as np
from src.skills.periodogram import LombScarglePeriodogram


class TestLombScargle:
    """Tests for Lomb-Scargle periodogram skill."""
    
    def test_sinusoidal_signal(self):
        """Test detection of simple sinusoidal signal."""
        # Generate test data
        np.random.seed(42)
        period = 5.0  # days
        time = np.sort(np.random.uniform(0, 100, 500))
        flux = 1.0 + 0.01 * np.sin(2 * np.pi * time / period)
        flux += np.random.normal(0, 0.001, len(time))
        
        # Compute periodogram
        lsp = LombScarglePeriodogram(min_period=1, max_period=20)
        result = lsp.compute(time, flux)
        
        # Check best period is close to true period
        assert abs(result.best_period - period) < 0.1
    
    def test_no_signal(self):
        """Test behavior on noise-only data."""
        np.random.seed(42)
        time = np.linspace(0, 100, 500)
        flux = 1.0 + np.random.normal(0, 0.01, len(time))
        
        lsp = LombScarglePeriodogram()
        result = lsp.compute(time, flux)
        
        # FAP should be high (no significant period)
        assert result.fap > 0.01
    
    def test_input_validation(self):
        """Test input validation."""
        lsp = LombScarglePeriodogram()
        
        # Mismatched lengths
        with pytest.raises(ValueError):
            lsp.compute(np.array([1, 2, 3]), np.array([1, 2]))
        
        # Too few points
        with pytest.raises(ValueError):
            lsp.compute(np.array([1, 2, 3]), np.array([1, 2, 3]))
```

### Step 4: Register the Skill

Add to skill registry in `src/skills/__init__.py`:

```python
# src/skills/__init__.py

from .periodogram import LombScarglePeriodogram, SKILL_INFO as LS_SKILL

# Skill registry
SKILLS = {
    'ANAL-010': LS_SKILL,
    # ... other skills
}

def get_skill(skill_id):
    """Get skill by ID."""
    return SKILLS.get(skill_id)

def list_skills(category=None):
    """List available skills."""
    skills = SKILLS.values()
    if category:
        skills = [s for s in skills if s.get('category') == category]
    return list(skills)
```

---

## 3. Skill Categories

### Data Skills (`data`)

Handle data input/output and preprocessing.

```python
# Examples
- DATA-001: NASA Data Ingestion
- DATA-002: FITS Parser
- DATA-003: Light Curve Preprocessing
- DATA-004: Spectral Extraction
```

### Analysis Skills (`analysis`)

Perform scientific analysis.

```python
# Examples
- ANAL-001: BLS Periodogram
- ANAL-002: Phase Folding
- ANAL-003: Transit Fitting
- ANAL-010: Lomb-Scargle
```

### Detection Skills (`detection`)

Detect astronomical objects/events.

```python
# Examples
- DETECT-001: Transit Detection
- DETECT-002: Anomaly Detection
- DETECT-003: Variable Detection
```

### Classification Skills (`classification`)

Classify objects.

```python
# Examples
- CLASS-001: Star Classification
- CLASS-002: Galaxy Morphology
- CLASS-003: Transient Classification
```

### Model Skills (`model`)

ML model operations.

```python
# Examples
- MODEL-001: Train CNN
- MODEL-002: TFLite Export
- MODEL-003: Model Evaluation
```

### Report Skills (`report`)

Generate outputs.

```python
# Examples
- REPORT-001: NASA Report
- REPORT-002: Dashboard
- REPORT-003: Figure Generation
```

---

## 4. Skill Dependencies

### Dependency Types

```yaml
dependencies:
  # Python packages
  python:
    - numpy>=1.21
    - astropy>=5.0
    - tensorflow>=2.10
  
  # Required models
  models:
    - spectral_cnn.tflite
  
  # Required data/catalogs
  data:
    - exoplanet_archive
    - tic_catalog
  
  # Other skills
  skills:
    - DATA-001  # Requires data ingestion
```

### Dependency Resolution

```python
def resolve_dependencies(skill_id):
    """Resolve all dependencies for a skill."""
    skill = get_skill(skill_id)
    deps = skill.get('dependencies', {})
    
    # Check Python packages
    missing_python = []
    for pkg in deps.get('python', []):
        try:
            __import__(pkg.split('>=')[0])
        except ImportError:
            missing_python.append(pkg)
    
    if missing_python:
        print(f"Missing packages: {missing_python}")
        print(f"Install with: pip install {' '.join(missing_python)}")
        return False
    
    # Check models
    for model in deps.get('models', []):
        if not Path(f"models/{model}").exists():
            print(f"Missing model: {model}")
            return False
    
    # Check dependent skills
    for dep_skill in deps.get('skills', []):
        if not resolve_dependencies(dep_skill):
            return False
    
    return True
```

---

## 5. CLI Integration

### Adding Commands

```python
# src/cli/main.py

import argparse
from ..skills import SKILLS

def build_parser():
    """Build CLI parser from skill definitions."""
    parser = argparse.ArgumentParser(
        prog='larun',
        description='LARUN TinyML - Astronomical Analysis'
    )
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Group skills by category
    categories = {}
    for skill_id, skill in SKILLS.items():
        cmd_parts = skill['command'].split()
        if len(cmd_parts) >= 2:
            cat = cmd_parts[1]  # e.g., 'analyze'
            if cat not in categories:
                categories[cat] = subparsers.add_parser(cat)
                categories[cat]._subparsers = categories[cat].add_subparsers()
            
            # Add skill command
            action = cmd_parts[2] if len(cmd_parts) > 2 else 'run'
            skill_parser = categories[cat]._subparsers.add_parser(action)
            
            # Add arguments from skill definition
            add_skill_arguments(skill_parser, skill)
            skill_parser.set_defaults(func=skill['function'])
    
    return parser


def add_skill_arguments(parser, skill):
    """Add arguments from skill YAML definition."""
    for inp in skill.get('inputs', []):
        name = f"--{inp['name'].replace('_', '-')}"
        kwargs = {
            'help': inp.get('description', ''),
            'required': inp.get('required', False)
        }
        
        if 'default' in inp:
            kwargs['default'] = inp['default']
        
        # Type conversion
        type_map = {'float': float, 'int': int, 'string': str}
        if inp['type'] in type_map:
            kwargs['type'] = type_map[inp['type']]
        
        parser.add_argument(name, **kwargs)
```

---

## 6. Documentation Template

### Skill Documentation

```markdown
# Skill Name (SKILL-ID)

## Overview

Brief description of what the skill does.

## Command

```bash
larun category action [options]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| --input | filepath | required | Input file |
| --param | float | 1.0 | Description |

## Examples

### Basic Usage
```bash
larun category action --input data.fits
```

### Advanced Usage
```bash
larun category action --input data.fits --param 2.0 --format json
```

## Output

Description of output format and fields.

```json
{
  "result": "value",
  "metric": 0.95
}
```

## Algorithm

Brief description of the algorithm/method used.

## References

1. Author (Year). "Paper Title." Journal.

## See Also

- [Related Skill 1](SKILL-002.md)
- [Related Skill 2](SKILL-003.md)
```

---

## 7. Best Practices

### Code Style

```python
# 1. Type hints
def compute(self, time: np.ndarray, flux: np.ndarray) -> Result:
    pass

# 2. Docstrings with examples
def function(arg):
    """
    Brief description.
    
    Args:
        arg: Description
    
    Returns:
        Description
    
    Example:
        >>> result = function(value)
    """
    pass

# 3. Logging
import logging
logger = logging.getLogger(__name__)
logger.info("Processing started")
logger.warning("Low SNR detected")

# 4. Error handling
if value < 0:
    raise ValueError(f"Value must be positive, got {value}")
```

### Testing

```python
# 1. Unit tests for each public function
# 2. Integration tests with real data samples
# 3. Edge cases (empty input, NaN values, etc.)
# 4. Performance benchmarks for TinyML constraints
```

### Documentation

```python
# 1. YAML skill definition is authoritative
# 2. Docstrings explain implementation details
# 3. Examples show common usage patterns
# 4. References cite scientific methods
```

---

## 8. Checklist for New Skills

- [ ] YAML definition in `skills/`
- [ ] Python implementation in `src/skills/`
- [ ] Unit tests in `tests/`
- [ ] CLI integration
- [ ] Documentation
- [ ] Dependencies listed
- [ ] Examples provided
- [ ] TinyML constraints considered
- [ ] Error handling implemented
- [ ] Logging added

---

*Last Updated: 2024*
*LARUN - Larun. × Astrodata*
