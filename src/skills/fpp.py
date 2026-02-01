"""
False Positive Probability Calculator for LARUN
================================================

Calculates the false positive probability (FPP) for transit signals,
implementing a simplified version of vespa/triceratops methodology.

The FPP represents the probability that a transit signal is caused by
a false positive scenario (eclipsing binary, background eclipsing binary,
etc.) rather than a genuine planetary transit.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class StellarParams:
    """Stellar parameters from Gaia or other sources."""
    tic_id: Optional[str] = None
    teff: float = 5778.0  # Effective temperature (K)
    logg: float = 4.44    # Surface gravity (log g)
    radius: float = 1.0   # Stellar radius (R_sun)
    mass: float = 1.0     # Stellar mass (M_sun)
    metallicity: float = 0.0  # [Fe/H]
    distance_pc: float = 100.0  # Distance in parsecs
    is_valid: bool = True
    quality_flags: List[str] = field(default_factory=list)

    def spectral_type(self) -> str:
        """Estimate spectral type from effective temperature."""
        if self.teff >= 30000:
            return "O"
        elif self.teff >= 10000:
            return "B"
        elif self.teff >= 7500:
            return "A"
        elif self.teff >= 6000:
            return "F"
        elif self.teff >= 5200:
            return "G"
        elif self.teff >= 3700:
            return "K"
        elif self.teff >= 2400:
            return "M"
        else:
            return "L"

    def luminosity_class(self) -> str:
        """Estimate luminosity class from surface gravity."""
        if self.logg >= 4.0:
            return "V"  # Dwarf
        elif self.logg >= 3.0:
            return "IV"  # Subgiant
        elif self.logg >= 1.5:
            return "III"  # Giant
        elif self.logg >= 0.5:
            return "II"  # Bright giant
        else:
            return "I"  # Supergiant

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'tic_id': self.tic_id,
            'teff_k': self.teff,
            'logg': self.logg,
            'radius_rsun': self.radius,
            'mass_msun': self.mass,
            'metallicity': self.metallicity,
            'distance_pc': self.distance_pc,
            'spectral_type': self.spectral_type(),
            'luminosity_class': self.luminosity_class(),
            'is_valid': self.is_valid,
            'quality_flags': self.quality_flags,
        }


@dataclass
class FPScenario:
    """A single false positive scenario with its probability."""
    name: str
    description: str
    prior: float
    likelihood: float
    posterior: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'prior': self.prior,
            'likelihood': self.likelihood,
            'posterior': self.posterior,
        }


@dataclass
class FPPResult:
    """Result of FPP calculation."""
    fpp: float
    disposition: str  # VALIDATED, CANDIDATE, LIKELY_FP
    scenarios: List[FPScenario]
    target_name: Optional[str] = None
    period_days: float = 0.0
    depth_ppm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'fpp': self.fpp,
            'disposition': self.disposition,
            'target_name': self.target_name,
            'period_days': self.period_days,
            'depth_ppm': self.depth_ppm,
            'scenarios': [s.to_dict() for s in self.scenarios],
        }


class FPPCalculator:
    """
    False Positive Probability Calculator.

    Implements a simplified FPP calculation based on:
    - Transit depth (deeper transits more likely to be EBs)
    - Orbital period (short periods can indicate contact binaries)
    - Stellar parameters (giant stars more likely to show EB-like signals)
    - Secondary eclipse depth

    For full FPP calculations, see vespa or triceratops packages.
    """

    # Default priors for each scenario
    DEFAULT_PRIORS = {
        'planet': 0.1,           # Genuine transiting planet
        'eb': 0.3,               # Eclipsing binary
        'beb': 0.2,              # Background eclipsing binary
        'heb': 0.15,             # Hierarchical eclipsing binary
        'planet_twin': 0.05,     # Planet around binary companion
        'blend': 0.2,            # Blended signal (other)
    }

    def __init__(self, priors: Optional[Dict[str, float]] = None):
        """
        Initialize FPP calculator.

        Args:
            priors: Custom prior probabilities for each scenario
        """
        self.priors = priors or self.DEFAULT_PRIORS.copy()

    def calculate(
        self,
        period: float,
        depth_ppm: float,
        stellar_params: Optional[StellarParams] = None,
        secondary_depth_ppm: Optional[float] = None,
        odd_even_diff: Optional[float] = None,
        centroid_offset: Optional[float] = None,
        target_name: Optional[str] = None,
    ) -> FPPResult:
        """
        Calculate False Positive Probability.

        Args:
            period: Orbital period in days
            depth_ppm: Transit depth in parts per million
            stellar_params: Stellar parameters (uses defaults if None)
            secondary_depth_ppm: Secondary eclipse depth if detected
            odd_even_diff: Difference between odd and even transit depths
            centroid_offset: Centroid shift during transit (arcsec)
            target_name: Target identifier

        Returns:
            FPPResult with FPP value and scenario breakdown
        """
        if stellar_params is None:
            stellar_params = StellarParams()

        # Convert depth to fractional
        depth_frac = depth_ppm / 1e6

        # Calculate likelihoods for each scenario
        scenarios = []

        # Planet likelihood - higher for shallow transits around dwarf stars
        planet_likelihood = self._planet_likelihood(
            depth_frac, period, stellar_params
        )
        scenarios.append(FPScenario(
            name='planet',
            description='Genuine transiting exoplanet',
            prior=self.priors['planet'],
            likelihood=planet_likelihood,
        ))

        # Eclipsing binary likelihood - higher for deep transits
        eb_likelihood = self._eb_likelihood(
            depth_frac, period, secondary_depth_ppm, odd_even_diff
        )
        scenarios.append(FPScenario(
            name='eb',
            description='Eclipsing binary',
            prior=self.priors['eb'],
            likelihood=eb_likelihood,
        ))

        # Background eclipsing binary
        beb_likelihood = self._beb_likelihood(
            depth_frac, centroid_offset, stellar_params
        )
        scenarios.append(FPScenario(
            name='beb',
            description='Background eclipsing binary',
            prior=self.priors['beb'],
            likelihood=beb_likelihood,
        ))

        # Hierarchical eclipsing binary
        heb_likelihood = self._heb_likelihood(depth_frac, stellar_params)
        scenarios.append(FPScenario(
            name='heb',
            description='Hierarchical eclipsing binary',
            prior=self.priors['heb'],
            likelihood=heb_likelihood,
        ))

        # Blend scenario
        blend_likelihood = self._blend_likelihood(depth_frac)
        scenarios.append(FPScenario(
            name='blend',
            description='Blended signal',
            prior=self.priors['blend'],
            likelihood=blend_likelihood,
        ))

        # Calculate posteriors using Bayes' theorem
        total_evidence = sum(s.prior * s.likelihood for s in scenarios)

        if total_evidence > 0:
            for s in scenarios:
                s.posterior = (s.prior * s.likelihood) / total_evidence
        else:
            # Equal posteriors if no evidence
            for s in scenarios:
                s.posterior = 1.0 / len(scenarios)

        # FPP is 1 - P(planet)
        planet_posterior = next(
            s.posterior for s in scenarios if s.name == 'planet'
        )
        fpp = 1.0 - planet_posterior

        # Determine disposition
        if fpp < 0.01:
            disposition = "VALIDATED"
        elif fpp < 0.5:
            disposition = "CANDIDATE"
        else:
            disposition = "LIKELY_FP"

        return FPPResult(
            fpp=fpp,
            disposition=disposition,
            scenarios=scenarios,
            target_name=target_name,
            period_days=period,
            depth_ppm=depth_ppm,
        )

    def _planet_likelihood(
        self,
        depth: float,
        period: float,
        stellar: StellarParams,
    ) -> float:
        """Calculate likelihood for planet scenario."""
        likelihood = 1.0

        # Shallow transits strongly favor planet hypothesis
        # Planet transits typically < 1% (10000 ppm)
        # Hot Jupiters: 1-2%, Earth-sized: 0.01%, Super-Earths: 0.1%
        if depth < 0.0005:  # < 0.05% - small planet
            likelihood *= 3.0
        elif depth < 0.002:  # < 0.2% - typical planet range
            likelihood *= 2.5
        elif depth < 0.01:  # < 1% - larger planet
            likelihood *= 1.5
        elif depth < 0.05:  # < 5% - possible grazing or hot Jupiter
            likelihood *= 0.3
        else:  # Very deep - unlikely planet
            likelihood *= 0.01

        # Dwarf stars more likely to show planet transits
        if stellar.logg >= 4.0:
            likelihood *= 2.0
        elif stellar.logg < 3.0:  # Giant star
            likelihood *= 0.3

        # Period consideration - most known planets have P > 1 day
        if period > 1.0:  # Reasonable planet period
            likelihood *= 1.5
        elif period > 0.5:  # Short but possible (hot Jupiters)
            likelihood *= 1.0
        else:  # Very short period - less likely planet
            likelihood *= 0.5

        return max(0.001, likelihood)  # No upper cap - shallow transits strongly favor planets

    def _eb_likelihood(
        self,
        depth: float,
        period: float,
        secondary_depth: Optional[float],
        odd_even_diff: Optional[float],
    ) -> float:
        """Calculate likelihood for eclipsing binary scenario."""
        likelihood = 0.5

        # Deep eclipses strongly favor EB, shallow disfavor
        if depth > 0.05:  # > 5%
            likelihood *= 5.0
        elif depth > 0.01:  # > 1%
            likelihood *= 2.0
        elif depth > 0.005:  # > 0.5%
            likelihood *= 0.8
        elif depth > 0.002:  # > 0.2%
            likelihood *= 0.4
        else:  # Very shallow - unlikely EB
            likelihood *= 0.2

        # Secondary eclipse is strong evidence for EB
        if secondary_depth is not None:
            if secondary_depth > 100:  # > 100 ppm
                likelihood *= 3.0

        # Odd-even depth difference indicates EB
        if odd_even_diff is not None:
            if odd_even_diff > 0.05:  # > 5% difference
                likelihood *= 3.0

        # Very short periods favor contact binaries
        if period < 0.5:
            likelihood *= 2.0

        return max(0.001, min(1.0, likelihood))

    def _beb_likelihood(
        self,
        depth: float,
        centroid_offset: Optional[float],
        stellar: StellarParams,
    ) -> float:
        """Calculate likelihood for background EB scenario."""
        likelihood = 0.3

        # Centroid offset is strong evidence for BEB
        if centroid_offset is not None:
            if centroid_offset > 1.0:  # > 1 arcsec
                likelihood *= 5.0
            elif centroid_offset > 0.5:
                likelihood *= 2.0

        # Crowded fields (low galactic latitude) increase BEB probability
        # For now, use distance as rough proxy
        if stellar.distance_pc > 500:
            likelihood *= 1.5

        return max(0.001, min(1.0, likelihood))

    def _heb_likelihood(
        self,
        depth: float,
        stellar: StellarParams,
    ) -> float:
        """Calculate likelihood for hierarchical EB scenario."""
        likelihood = 0.2

        # Moderate depths can indicate diluted EB
        if 0.001 < depth < 0.02:
            likelihood *= 1.5

        # Higher mass stars more likely in hierarchical systems
        if stellar.mass > 1.5:
            likelihood *= 1.5

        return max(0.001, min(1.0, likelihood))

    def _blend_likelihood(self, depth: float) -> float:
        """Calculate likelihood for generic blend scenario."""
        likelihood = 0.2

        # Shallow transits could be diluted deep eclipses
        if depth < 0.005:
            likelihood *= 1.5

        return max(0.001, min(1.0, likelihood))
