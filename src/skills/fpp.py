"""
LARUN Skill: False Positive Probability (FPP) Calculator
=========================================================
Bayesian calculation of False Positive Probability for transit candidates.

Skill ID: FPP-001
Commands: larun fpp, /fpp

Based on: Morton et al. (2016) - VESPA method
Reference: docs/research/EXOPLANET_DETECTION.md

Created by: Padmanaban Veeraragavalu (Larun Engineering)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StellarParams:
    """Stellar parameters for FPP calculation."""
    teff: float = 5778.0  # Effective temperature (K), default Sun
    logg: float = 4.44    # Surface gravity (log g), default Sun
    radius: float = 1.0   # Stellar radius (R_sun)
    mass: float = 1.0     # Stellar mass (M_sun)
    metallicity: float = 0.0  # [Fe/H]
    distance_pc: float = 100.0  # Distance in parsecs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'teff_k': round(self.teff, 0),
            'logg': round(self.logg, 3),
            'radius_rsun': round(self.radius, 3),
            'mass_msun': round(self.mass, 3),
            'metallicity': round(self.metallicity, 3),
            'distance_pc': round(self.distance_pc, 1)
        }


@dataclass
class ScenarioProbability:
    """Probability for a single false positive scenario."""
    scenario: str
    prior: float
    likelihood: float
    posterior: float = 0.0
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario': self.scenario,
            'prior': round(self.prior, 6),
            'likelihood': round(self.likelihood, 6),
            'posterior': round(self.posterior, 6),
            'description': self.description
        }


@dataclass
class FPPResult:
    """Result of False Positive Probability calculation."""
    fpp: float  # False Positive Probability (0-1)
    disposition: str  # "VALIDATED", "CANDIDATE", "LIKELY_FP"
    confidence: float
    planet_probability: float
    scenarios: List[ScenarioProbability]
    target_name: str = ""
    period: float = 0.0
    depth_ppm: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'target': self.target_name,
            'period_days': round(self.period, 6),
            'depth_ppm': round(self.depth_ppm, 1),
            'fpp': round(self.fpp, 6),
            'planet_probability': round(self.planet_probability, 6),
            'disposition': self.disposition,
            'confidence': round(self.confidence, 3),
            'scenarios': [s.to_dict() for s in self.scenarios]
        }
    
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"═══════════════════════════════════════════════════════════════",
            f"  FALSE POSITIVE PROBABILITY ANALYSIS",
            f"═══════════════════════════════════════════════════════════════",
            f"  Target: {self.target_name}",
            f"  Period: {self.period:.4f} days",
            f"  Depth: {self.depth_ppm:.0f} ppm",
            f"───────────────────────────────────────────────────────────────",
            f"  FPP: {self.fpp:.2e} ({self.fpp*100:.4f}%)",
            f"  Planet Probability: {self.planet_probability:.2%}",
            f"  Disposition: {self.disposition}",
            f"───────────────────────────────────────────────────────────────",
            f"  Scenario Probabilities:",
        ]
        
        for scenario in sorted(self.scenarios, key=lambda x: -x.posterior):
            lines.append(f"    • {scenario.scenario}: {scenario.posterior:.2e}")
        
        lines.append(f"═══════════════════════════════════════════════════════════════")
        return "\n".join(lines)


# ============================================================================
# FPP Calculator
# ============================================================================

class FPPCalculator:
    """
    Bayesian False Positive Probability calculator.
    
    Calculates the probability that a transit signal is caused by
    a false positive scenario rather than a genuine exoplanet.
    
    Based on the VESPA method (Morton 2012, 2016).
    
    False Positive Scenarios:
    1. Planet (genuine transiting planet)
    2. EB (eclipsing binary - target star is actually a binary)
    3. BEB (background eclipsing binary - unrelated binary in field)
    4. HEB (hierarchical eclipsing binary - bound triple system)
    5. PL (planet around secondary - planet transits companion star)
    
    Example:
        >>> calc = FPPCalculator()
        >>> result = calc.calculate(vetting_result, stellar_params)
        >>> print(f"FPP: {result.fpp:.2e}")
    """
    
    # Default prior probabilities based on occurrence rates
    # These are rough estimates based on Kepler statistics
    DEFAULT_PRIORS = {
        'planet': 0.01,      # ~1% of stars have detectable transiting planets
        'eb': 0.003,         # ~0.3% are eclipsing binaries with given depth
        'beb': 0.001,        # ~0.1% background EBs at this depth
        'heb': 0.0003,       # ~0.03% hierarchical EBs
        'pl': 0.0001,        # ~0.01% planets around secondary
    }
    
    # Stellar density constraints (solar units)
    STELLAR_DENSITY_RANGE = {
        'dwarf': (0.1, 10.0),    # Main sequence stars
        'giant': (0.001, 0.1),   # Giant stars
    }
    
    def __init__(
        self,
        priors: Optional[Dict[str, float]] = None,
        fpp_threshold: float = 0.01,  # 1% for validation
        candidate_threshold: float = 0.1  # 10% for candidate
    ):
        """
        Initialize FPP Calculator.
        
        Args:
            priors: Custom prior probabilities for each scenario
            fpp_threshold: FPP below this = VALIDATED (default 1%)
            candidate_threshold: FPP below this = CANDIDATE (default 10%)
        """
        self.priors = priors or self.DEFAULT_PRIORS.copy()
        self.fpp_threshold = fpp_threshold
        self.candidate_threshold = candidate_threshold
    
    def calculate(
        self,
        vetting_result: Optional[Any] = None,
        stellar_params: Optional[StellarParams] = None,
        period: float = 1.0,
        depth_ppm: float = 1000.0,
        duration_hours: float = 3.0,
        transit_snr: float = 10.0,
        odd_even_sigma: float = 0.0,
        secondary_depth_ratio: float = 0.0,
        v_shape_ratio: float = 0.0,
        target_name: str = ""
    ) -> FPPResult:
        """
        Calculate False Positive Probability.
        
        Args:
            vetting_result: Result from TransitVetter (optional)
            stellar_params: Stellar parameters (optional, uses defaults)
            period: Orbital period in days
            depth_ppm: Transit depth in parts per million
            duration_hours: Transit duration in hours
            transit_snr: Signal-to-noise ratio of transit
            odd_even_sigma: Odd-even depth difference in sigma
            secondary_depth_ratio: Secondary eclipse depth / primary depth
            v_shape_ratio: V-shape ratio (0 = flat, 1 = fully V-shaped)
            target_name: Target identifier
            
        Returns:
            FPPResult with FPP value and scenario probabilities
        """
        logger.info(f"Calculating FPP for P={period:.4f}d, depth={depth_ppm:.0f}ppm")
        
        # Use defaults if not provided
        if stellar_params is None:
            stellar_params = StellarParams()
        
        # Extract vetting results if provided
        if vetting_result is not None:
            odd_even_sigma = self._extract_odd_even(vetting_result)
            secondary_depth_ratio = self._extract_secondary(vetting_result)
            v_shape_ratio = self._extract_v_shape(vetting_result)
        
        # Calculate depth ratio (Rp/Rs)^2 ≈ depth
        depth_fraction = depth_ppm / 1e6
        rp_rs = np.sqrt(depth_fraction)
        
        # Calculate likelihoods for each scenario
        scenarios = []
        
        # 1. Planet likelihood
        planet_likelihood = self._planet_likelihood(
            depth_fraction, period, duration_hours, stellar_params,
            odd_even_sigma, secondary_depth_ratio, v_shape_ratio
        )
        scenarios.append(ScenarioProbability(
            scenario="Planet",
            prior=self.priors['planet'],
            likelihood=planet_likelihood,
            posterior=0.0,  # Will be computed later
            description="Genuine transiting exoplanet"
        ))
        
        # 2. Eclipsing Binary likelihood
        eb_likelihood = self._eb_likelihood(
            depth_fraction, period, odd_even_sigma, 
            secondary_depth_ratio, v_shape_ratio
        )
        scenarios.append(ScenarioProbability(
            scenario="EB",
            prior=self.priors['eb'],
            likelihood=eb_likelihood,
            description="Eclipsing binary (target is binary star)"
        ))
        
        # 3. Background Eclipsing Binary likelihood
        beb_likelihood = self._beb_likelihood(
            depth_fraction, period, stellar_params
        )
        scenarios.append(ScenarioProbability(
            scenario="BEB",
            prior=self.priors['beb'],
            likelihood=beb_likelihood,
            description="Background eclipsing binary (blended source)"
        ))
        
        # 4. Hierarchical Eclipsing Binary likelihood
        heb_likelihood = self._heb_likelihood(
            depth_fraction, period, stellar_params
        )
        scenarios.append(ScenarioProbability(
            scenario="HEB",
            prior=self.priors['heb'],
            likelihood=heb_likelihood,
            description="Hierarchical eclipsing binary (bound triple)"
        ))
        
        # 5. Planet around secondary likelihood
        pl_likelihood = self._pl_likelihood(
            depth_fraction, period
        )
        scenarios.append(ScenarioProbability(
            scenario="PL",
            prior=self.priors['pl'],
            likelihood=pl_likelihood,
            description="Planet transiting secondary star"
        ))
        
        # Calculate posteriors using Bayes theorem
        # P(scenario|data) ∝ P(data|scenario) × P(scenario)
        unnormalized = []
        for s in scenarios:
            s.posterior = s.prior * s.likelihood
            unnormalized.append(s.posterior)
        
        # Normalize
        total = sum(unnormalized)
        if total > 0:
            for s in scenarios:
                s.posterior = s.posterior / total
        
        # Calculate FPP = 1 - P(planet|data)
        planet_prob = scenarios[0].posterior
        fpp = 1.0 - planet_prob
        
        # Determine disposition
        if fpp < self.fpp_threshold:
            disposition = "VALIDATED"
        elif fpp < self.candidate_threshold:
            disposition = "CANDIDATE"
        else:
            disposition = "LIKELY_FP"
        
        # Calculate confidence based on SNR and number of transits
        confidence = min(0.95, 0.5 + 0.05 * transit_snr)
        
        return FPPResult(
            fpp=fpp,
            disposition=disposition,
            confidence=confidence,
            planet_probability=planet_prob,
            scenarios=scenarios,
            target_name=target_name,
            period=period,
            depth_ppm=depth_ppm
        )
    
    def _planet_likelihood(
        self,
        depth: float,
        period: float,
        duration: float,
        stellar: StellarParams,
        odd_even_sigma: float,
        secondary_ratio: float,
        v_shape: float
    ) -> float:
        """
        Calculate likelihood that signal is from a genuine planet.
        
        Factors considered:
        - Transit depth consistent with planet sizes
        - Duration consistent with stellar density
        - No significant odd-even difference
        - No secondary eclipse
        - U-shaped (not V-shaped) transit
        """
        likelihood = 1.0
        
        # Depth constraint: planets typically have depth < 3%
        # (Jupiter around Sun is ~1%)
        if depth > 0.03:
            likelihood *= 0.01  # Very unlikely for planet
        elif depth > 0.01:
            likelihood *= 0.5
        else:
            likelihood *= 1.0
        
        # Odd-even consistency: planets should have equal depths
        if odd_even_sigma > 3.0:
            likelihood *= 0.1
        elif odd_even_sigma > 2.0:
            likelihood *= 0.5
        else:
            likelihood *= 1.0
        
        # Secondary eclipse: planets have very small secondaries
        if secondary_ratio > 0.1:
            likelihood *= 0.05
        elif secondary_ratio > 0.01:
            likelihood *= 0.7
        else:
            likelihood *= 1.0
        
        # Shape: planets have U-shaped transits
        if v_shape > 0.5:
            likelihood *= 0.1
        elif v_shape > 0.3:
            likelihood *= 0.5
        else:
            likelihood *= 1.0
        
        # Period constraint: use occurrence rate from Kepler
        # Short periods are more common for transiting planets
        if period < 10:
            likelihood *= 1.0
        elif period < 100:
            likelihood *= 0.5
        else:
            likelihood *= 0.1
        
        return max(likelihood, 1e-10)
    
    def _eb_likelihood(
        self,
        depth: float,
        period: float,
        odd_even_sigma: float,
        secondary_ratio: float,
        v_shape: float
    ) -> float:
        """
        Calculate likelihood that signal is from an eclipsing binary.
        
        EBs typically show:
        - Deep eclipses (up to 50%)
        - Significant secondaries
        - V-shaped transits (grazing)
        - Odd-even differences if period is wrong by factor of 2
        """
        likelihood = 1.0
        
        # Depth constraint: EBs can be very deep
        if depth > 0.01:
            likelihood *= 2.0  # More likely EB if deep
        else:
            likelihood *= 0.3  # Shallow transits less common for EBs
        
        # Secondary eclipse: EBs often have significant secondaries
        if secondary_ratio > 0.1:
            likelihood *= 5.0  # Strong indicator of EB
        elif secondary_ratio > 0.01:
            likelihood *= 2.0
        else:
            likelihood *= 0.5
        
        # V-shape: grazing EBs are V-shaped
        if v_shape > 0.3:
            likelihood *= 3.0
        else:
            likelihood *= 0.7
        
        # Odd-even test: if periods are confused
        if odd_even_sigma > 3.0:
            likelihood *= 3.0  # Could be EB at half period
        else:
            likelihood *= 0.8
        
        return max(likelihood, 1e-10)
    
    def _beb_likelihood(
        self,
        depth: float,
        period: float,
        stellar: StellarParams
    ) -> float:
        """
        Calculate likelihood of background eclipsing binary.
        
        BEBs are diluted by the target star flux:
        - Appear as shallow transits
        - More likely in crowded fields
        - More likely for nearby/bright targets (larger aperture)
        """
        likelihood = 1.0
        
        # Shallow transits more consistent with dilution
        if depth < 0.005:
            likelihood *= 1.5
        elif depth < 0.01:
            likelihood *= 1.0
        else:
            likelihood *= 0.3  # Deep transits hard to explain with BEB
        
        # Distance constraint: closer stars have larger apertures
        if stellar.distance_pc < 50:
            likelihood *= 2.0  # More background sources in aperture
        elif stellar.distance_pc < 200:
            likelihood *= 1.0
        else:
            likelihood *= 0.5
        
        return max(likelihood, 1e-10)
    
    def _heb_likelihood(
        self,
        depth: float,
        period: float,
        stellar: StellarParams
    ) -> float:
        """
        Calculate likelihood of hierarchical eclipsing binary.
        
        HEBs are bound triple systems where two components eclipse.
        - Third light dilutes the eclipse
        - More common for wide binaries
        """
        likelihood = 0.5  # Base rate is low
        
        # Shallow depths consistent with dilution
        if depth < 0.005:
            likelihood *= 1.5
        else:
            likelihood *= 0.5
        
        # Main-sequence primaries more likely to have companions
        if stellar.logg > 4.0:  # Dwarf
            likelihood *= 1.0
        else:  # Giant - evolved past potential companion
            likelihood *= 0.3
        
        return max(likelihood, 1e-10)
    
    def _pl_likelihood(
        self,
        depth: float,
        period: float
    ) -> float:
        """
        Calculate likelihood of planet around secondary star.
        
        In binary systems, the planet might orbit the fainter companion.
        This creates a diluted transit signal.
        """
        likelihood = 0.3  # Base rate is quite low
        
        # Shallow transits more consistent with dilution
        if depth < 0.003:
            likelihood *= 2.0
        elif depth < 0.01:
            likelihood *= 1.0
        else:
            likelihood *= 0.2
        
        return max(likelihood, 1e-10)
    
    def _extract_odd_even(self, vetting_result) -> float:
        """Extract odd-even sigma from vetting result."""
        try:
            for test in vetting_result.tests:
                if "odd" in test.test_name.lower():
                    return test.details.get('difference_sigma', 0.0)
        except (AttributeError, TypeError):
            pass
        return 0.0
    
    def _extract_secondary(self, vetting_result) -> float:
        """Extract secondary eclipse ratio from vetting result."""
        try:
            for test in vetting_result.tests:
                if "secondary" in test.test_name.lower():
                    return test.details.get('depth_ratio', 0.0)
        except (AttributeError, TypeError):
            pass
        return 0.0
    
    def _extract_v_shape(self, vetting_result) -> float:
        """Extract V-shape ratio from vetting result."""
        try:
            for test in vetting_result.tests:
                if "shape" in test.test_name.lower() or "v-" in test.test_name.lower():
                    return test.details.get('flatness_ratio', 0.0)
        except (AttributeError, TypeError):
            pass
        return 0.0


# ============================================================================
# Convenience Functions
# ============================================================================

def calculate_fpp(
    period: float,
    depth_ppm: float,
    vetting_result: Optional[Any] = None,
    stellar_params: Optional[StellarParams] = None,
    **kwargs
) -> FPPResult:
    """
    Convenience function to calculate False Positive Probability.
    
    Args:
        period: Orbital period (days)
        depth_ppm: Transit depth (ppm)
        vetting_result: Optional VettingResult from TransitVetter
        stellar_params: Optional stellar parameters
        **kwargs: Additional arguments for FPPCalculator.calculate()
        
    Returns:
        FPPResult with FPP and scenario probabilities
    """
    calc = FPPCalculator()
    return calc.calculate(
        period=period,
        depth_ppm=depth_ppm,
        vetting_result=vetting_result,
        stellar_params=stellar_params,
        **kwargs
    )


def combine_with_vetting(vetting_result, stellar_params: Optional[StellarParams] = None) -> FPPResult:
    """
    Calculate FPP using vetting results.
    
    Args:
        vetting_result: VettingResult from TransitVetter
        stellar_params: Optional stellar parameters
        
    Returns:
        FPPResult
    """
    calc = FPPCalculator()
    
    # Extract parameters from vetting result
    period = getattr(vetting_result, 'period', 1.0)
    depth = 1000.0  # Default depth
    
    # Try to get depth from test details
    try:
        for test in vetting_result.tests:
            if 'primary_depth_ppm' in test.details:
                depth = test.details['primary_depth_ppm']
                break
    except (AttributeError, TypeError):
        pass
    
    return calc.calculate(
        vetting_result=vetting_result,
        stellar_params=stellar_params,
        period=period,
        depth_ppm=depth,
        target_name=getattr(vetting_result, 'target_name', '')
    )


# ============================================================================
# Skill Registration
# ============================================================================

SKILL_INFO = {
    'FPP-001': {
        'id': 'FPP-001',
        'name': 'False Positive Probability',
        'command': 'fpp',
        'class': FPPCalculator,
        'description': 'Bayesian FPP calculation for transit candidates'
    }
}


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing FPP Calculator...")
    print("=" * 60)
    
    # Test with typical planet-like parameters
    calc = FPPCalculator()
    
    # Case 1: Likely planet
    result = calc.calculate(
        period=3.5,
        depth_ppm=1200,
        duration_hours=2.5,
        transit_snr=15.0,
        odd_even_sigma=0.5,
        secondary_depth_ratio=0.001,
        v_shape_ratio=0.1,
        target_name="Test Planet"
    )
    print("\nCase 1: Planet-like signal")
    print(result.summary())
    
    # Case 2: Likely eclipsing binary
    result2 = calc.calculate(
        period=2.0,
        depth_ppm=15000,
        duration_hours=3.0,
        transit_snr=50.0,
        odd_even_sigma=4.0,
        secondary_depth_ratio=0.3,
        v_shape_ratio=0.6,
        target_name="Test EB"
    )
    print("\nCase 2: Binary-like signal")
    print(result2.summary())
    
    # Case 3: Ambiguous case
    result3 = calc.calculate(
        period=5.0,
        depth_ppm=800,
        duration_hours=4.0,
        transit_snr=8.0,
        odd_even_sigma=2.5,
        secondary_depth_ratio=0.05,
        v_shape_ratio=0.25,
        target_name="Ambiguous"
    )
    print("\nCase 3: Ambiguous signal")
    print(result3.summary())
