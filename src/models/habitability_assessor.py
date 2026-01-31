"""
LARUN Habitability Assessor
===========================
Assesses planetary habitability potential based on
orbital and stellar parameters.

Zones:
- Hot Zone: Too hot for liquid water
- Optimistic Habitable Zone: Extended HZ
- Conservative Habitable Zone: Traditional HZ
- Cold Zone: Too cold for liquid water

Created by: Padmanaban Veeraragavalu (Larun Engineering)
Reference: Kopparapu et al. (2013, 2014)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HabitabilityResult:
    """Result of habitability assessment."""
    habitability_score: float  # 0-1 overall score
    zone: str  # HOT, OPTIMISTIC_HZ, CONSERVATIVE_HZ, COLD
    is_potentially_habitable: bool
    
    # Insolation parameters
    insolation_earth: float  # S/S_earth
    equilibrium_temp_k: float
    
    # Zone boundaries
    inner_hz_au: float
    outer_hz_au: float
    planet_distance_au: float
    
    # Contributing factors
    factors: Dict[str, float]
    
    # Confidence and warnings
    confidence: float
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'habitability_score': round(self.habitability_score, 4),
            'zone': self.zone,
            'is_potentially_habitable': self.is_potentially_habitable,
            'insolation_earth': round(self.insolation_earth, 4),
            'equilibrium_temp_k': round(self.equilibrium_temp_k, 1),
            'inner_hz_au': round(self.inner_hz_au, 4),
            'outer_hz_au': round(self.outer_hz_au, 4),
            'planet_distance_au': round(self.planet_distance_au, 4),
            'factors': {k: round(v, 4) for k, v in self.factors.items()},
            'confidence': round(self.confidence, 4),
            'warnings': self.warnings
        }
    
    def summary(self) -> str:
        zone_emoji = {
            'HOT': '🔥',
            'OPTIMISTIC_HZ': '🌍',
            'CONSERVATIVE_HZ': '💚',
            'COLD': '❄️'
        }
        
        lines = [
            "═══════════════════════════════════════════════════════════",
            f"  HABITABILITY ASSESSMENT {zone_emoji.get(self.zone, '🌐')}",
            "═══════════════════════════════════════════════════════════",
            f"  Zone: {self.zone.replace('_', ' ')}",
            f"  Habitability Score: {self.habitability_score:.0%}",
            f"  Potentially Habitable: {'Yes' if self.is_potentially_habitable else 'No'}",
            "───────────────────────────────────────────────────────────",
            f"  Planet Distance: {self.planet_distance_au:.4f} AU",
            f"  Inner HZ: {self.inner_hz_au:.4f} AU",
            f"  Outer HZ: {self.outer_hz_au:.4f} AU",
            "───────────────────────────────────────────────────────────",
            f"  Insolation: {self.insolation_earth:.2f} S⊕",
            f"  Equilibrium Temp: {self.equilibrium_temp_k:.0f} K ({self.equilibrium_temp_k-273:.0f}°C)",
            "───────────────────────────────────────────────────────────",
            "  Factors:"
        ]
        
        for factor, value in self.factors.items():
            bar = "█" * int(value * 10)
            lines.append(f"    {factor:20} {bar:10} {value:.0%}")
        
        if self.warnings:
            lines.append("───────────────────────────────────────────────────────────")
            lines.append("  ⚠ Warnings:")
            for warning in self.warnings:
                lines.append(f"    • {warning}")
        
        lines.append("═══════════════════════════════════════════════════════════")
        return "\n".join(lines)


# ============================================================================
# Habitability Assessor
# ============================================================================

class HabitabilityAssessor:
    """
    Planetary habitability assessment based on physical parameters.
    
    Uses the Kopparapu et al. (2013, 2014) habitable zone boundaries
    and considers multiple factors for habitability scoring.
    
    Example:
        >>> assessor = HabitabilityAssessor()
        >>> result = assessor.assess(
        ...     period_days=365,
        ...     planet_radius_earth=1.0,
        ...     stellar_teff=5778,
        ...     stellar_radius_solar=1.0,
        ...     stellar_luminosity_solar=1.0
        ... )
        >>> print(result.summary())
    """
    
    # HZ coefficients from Kopparapu et al. 2014
    # [S_eff_sun, a, b, c, d] for Teff range 2600-7200K
    HZ_COEFFICIENTS = {
        # Recent Venus (optimistic inner)
        'recent_venus': [1.7763, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15],
        # Runaway greenhouse (conservative inner)
        'runaway_greenhouse': [1.0385, 1.2456e-4, 1.4612e-8, -7.6345e-12, -1.7511e-15],
        # Moist greenhouse
        'moist_greenhouse': [1.0146, 8.1884e-5, 1.9394e-9, -4.3618e-12, -6.8260e-16],
        # Maximum greenhouse (conservative outer)
        'max_greenhouse': [0.3507, 5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16],
        # Early Mars (optimistic outer)
        'early_mars': [0.3207, 5.4471e-5, 1.5275e-9, -2.1709e-12, -3.8282e-16],
    }
    
    def __init__(self):
        """Initialize the habitability assessor."""
        self.model = None
    
    def assess(
        self,
        period_days: float,
        planet_radius_earth: float,
        stellar_teff: float,
        stellar_radius_solar: float = 1.0,
        stellar_luminosity_solar: float = 1.0,
        stellar_mass_solar: float = 1.0,
        bond_albedo: float = 0.3
    ) -> HabitabilityResult:
        """
        Assess planetary habitability.
        
        Args:
            period_days: Orbital period in days
            planet_radius_earth: Planet radius in Earth radii
            stellar_teff: Stellar effective temperature (K)
            stellar_radius_solar: Stellar radius in solar radii
            stellar_luminosity_solar: Stellar luminosity in solar luminosities
            stellar_mass_solar: Stellar mass in solar masses
            bond_albedo: Planetary Bond albedo (default 0.3 for Earth-like)
            
        Returns:
            HabitabilityResult with full assessment
        """
        warnings = []
        
        # Calculate semi-major axis from Kepler's third law
        # a^3 / P^2 = G*M / (4*pi^2)
        # For M in solar masses, P in years, a in AU: a = (M * P^2)^(1/3)
        period_years = period_days / 365.25
        semi_major_axis_au = (stellar_mass_solar * period_years**2) ** (1/3)
        
        # Calculate HZ boundaries
        hz_boundaries = self._calculate_hz_boundaries(
            stellar_teff,
            stellar_luminosity_solar
        )
        
        inner_hz = hz_boundaries['runaway_greenhouse']  # Conservative inner
        outer_hz = hz_boundaries['max_greenhouse']       # Conservative outer
        optimistic_inner = hz_boundaries['recent_venus']
        optimistic_outer = hz_boundaries['early_mars']
        
        # Determine zone
        if semi_major_axis_au < optimistic_inner:
            zone = "HOT"
        elif semi_major_axis_au < inner_hz:
            zone = "OPTIMISTIC_HZ"
        elif semi_major_axis_au <= outer_hz:
            zone = "CONSERVATIVE_HZ"
        elif semi_major_axis_au <= optimistic_outer:
            zone = "OPTIMISTIC_HZ"
        else:
            zone = "COLD"
        
        # Calculate insolation
        insolation = stellar_luminosity_solar / (semi_major_axis_au ** 2)
        
        # Calculate equilibrium temperature
        sigma = 5.67e-8  # Stefan-Boltzmann constant
        L_sun = 3.828e26  # Solar luminosity in Watts
        AU_m = 1.496e11   # AU in meters
        
        L_star = stellar_luminosity_solar * L_sun
        d = semi_major_axis_au * AU_m
        
        T_eq = ((L_star * (1 - bond_albedo)) / (16 * np.pi * sigma * d**2)) ** 0.25
        
        # Calculate habitability factors
        factors = {}
        
        # Size factor: Earth-like is best (0.5-2 R_earth)
        if 0.5 <= planet_radius_earth <= 1.5:
            factors['size'] = 1.0
        elif 0.3 <= planet_radius_earth <= 2.0:
            factors['size'] = 0.7
        elif planet_radius_earth <= 0.3:
            factors['size'] = 0.2
            warnings.append("Planet may be too small to retain atmosphere")
        else:
            factors['size'] = 0.3
            warnings.append("Planet may be a mini-Neptune, not rocky")
        
        # Insolation factor
        if 0.8 <= insolation <= 1.2:
            factors['insolation'] = 1.0
        elif 0.4 <= insolation <= 2.0:
            factors['insolation'] = 0.7 - 0.3 * abs(insolation - 1.0)
        else:
            factors['insolation'] = max(0, 0.3 - 0.1 * abs(insolation - 1.0))
        
        # Zone factor
        if zone == "CONSERVATIVE_HZ":
            factors['zone'] = 1.0
        elif zone == "OPTIMISTIC_HZ":
            factors['zone'] = 0.6
        else:
            factors['zone'] = 0.1
        
        # Temperature factor
        if 200 <= T_eq <= 350:
            factors['temperature'] = 1.0 - abs(T_eq - 288) / 200
        else:
            factors['temperature'] = 0.1
        
        # Stellar type factor (K dwarfs are best, then G, M, F)
        if 4000 <= stellar_teff <= 5500:
            factors['stellar_type'] = 1.0  # K dwarf
        elif 5500 <= stellar_teff <= 6000:
            factors['stellar_type'] = 0.9  # G dwarf
        elif 3000 <= stellar_teff <= 4000:
            factors['stellar_type'] = 0.6  # M dwarf
            warnings.append("M dwarf host: potential tidal locking and flare activity")
        elif 6000 <= stellar_teff <= 7500:
            factors['stellar_type'] = 0.5  # F dwarf
            warnings.append("F dwarf host: shorter stellar lifetime")
        else:
            factors['stellar_type'] = 0.2
            warnings.append("Stellar type less favorable for habitability")
        
        # Calculate overall score
        habitability_score = (
            factors['size'] * 0.2 +
            factors['insolation'] * 0.25 +
            factors['zone'] * 0.25 +
            factors['temperature'] * 0.2 +
            factors['stellar_type'] * 0.1
        )
        
        # Determine if potentially habitable
        is_potentially_habitable = (
            zone in ["CONSERVATIVE_HZ", "OPTIMISTIC_HZ"] and
            0.5 <= planet_radius_earth <= 2.0 and
            habitability_score >= 0.4
        )
        
        # Confidence based on data quality (simplified)
        confidence = 0.8 if stellar_luminosity_solar > 0 else 0.5
        
        return HabitabilityResult(
            habitability_score=habitability_score,
            zone=zone,
            is_potentially_habitable=is_potentially_habitable,
            insolation_earth=insolation,
            equilibrium_temp_k=T_eq,
            inner_hz_au=inner_hz,
            outer_hz_au=outer_hz,
            planet_distance_au=semi_major_axis_au,
            factors=factors,
            confidence=confidence,
            warnings=warnings
        )
    
    def _calculate_hz_boundaries(
        self,
        teff: float,
        luminosity: float
    ) -> Dict[str, float]:
        """
        Calculate HZ boundaries using Kopparapu et al. coefficients.
        
        Args:
            teff: Stellar effective temperature (K)
            luminosity: Stellar luminosity in solar units
            
        Returns:
            Dict of HZ boundary distances in AU
        """
        # Teff offset from solar
        T_star = teff - 5780
        
        boundaries = {}
        for name, coeffs in self.HZ_COEFFICIENTS.items():
            S_eff_sun, a, b, c, d = coeffs
            
            # Calculate effective flux at boundary
            S_eff = S_eff_sun + a*T_star + b*T_star**2 + c*T_star**3 + d*T_star**4
            
            # Convert to distance in AU
            distance_au = np.sqrt(luminosity / S_eff)
            boundaries[name] = distance_au
        
        return boundaries
    
    def batch_assess(
        self,
        planets: List[Dict[str, float]]
    ) -> List[HabitabilityResult]:
        """
        Assess habitability for multiple planets.
        
        Args:
            planets: List of dicts with planet parameters
            
        Returns:
            List of HabitabilityResult
        """
        results = []
        for planet in planets:
            result = self.assess(
                period_days=planet.get('period_days', 365),
                planet_radius_earth=planet.get('planet_radius_earth', 1.0),
                stellar_teff=planet.get('stellar_teff', 5778),
                stellar_radius_solar=planet.get('stellar_radius_solar', 1.0),
                stellar_luminosity_solar=planet.get('stellar_luminosity_solar', 1.0),
                stellar_mass_solar=planet.get('stellar_mass_solar', 1.0)
            )
            results.append(result)
        
        return results


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Habitability Assessor...")
    print("=" * 60)
    
    assessor = HabitabilityAssessor()
    
    # Test 1: Earth
    print("\nTest 1: Earth (should be Conservative HZ)")
    result_earth = assessor.assess(
        period_days=365.25,
        planet_radius_earth=1.0,
        stellar_teff=5778,
        stellar_luminosity_solar=1.0,
        stellar_mass_solar=1.0
    )
    print(result_earth.summary())
    
    # Test 2: Hot planet
    print("\nTest 2: Hot planet (Mercury-like)")
    result_hot = assessor.assess(
        period_days=88,
        planet_radius_earth=0.4,
        stellar_teff=5778,
        stellar_luminosity_solar=1.0,
        stellar_mass_solar=1.0
    )
    print(f"Zone: {result_hot.zone}, Score: {result_hot.habitability_score:.0%}")
    
    # Test 3: Cold planet
    print("\nTest 3: Cold planet (Mars-like)")
    result_cold = assessor.assess(
        period_days=687,
        planet_radius_earth=0.53,
        stellar_teff=5778,
        stellar_luminosity_solar=1.0,
        stellar_mass_solar=1.0
    )
    print(f"Zone: {result_cold.zone}, Score: {result_cold.habitability_score:.0%}")
    
    # Test 4: Proxima b
    print("\nTest 4: Proxima b (M-dwarf HZ)")
    result_proxima = assessor.assess(
        period_days=11.2,
        planet_radius_earth=1.1,
        stellar_teff=3042,
        stellar_luminosity_solar=0.0017,
        stellar_mass_solar=0.12
    )
    print(result_proxima.summary())
    
    print("\nTest complete!")
