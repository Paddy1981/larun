"""
LARUN Skill: Exoplanet Characterization
=======================================
Calculate planet properties from transit observations and stellar parameters.

Skill IDs: PLANET-001 (Radius), PLANET-002 (Period), PLANET-005 (Habitability)
Commands: larun planet radius, larun planet habitable

Created by: Padmanaban Veeraragavalu (Larun Engineering)
With AI assistance from: Claude (Anthropic)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Physical constants
R_SUN = 6.9634e8       # Solar radius (m)
R_EARTH = 6.371e6      # Earth radius (m)
R_JUPITER = 6.9911e7   # Jupiter radius (m)
AU = 1.496e11          # Astronomical unit (m)
L_SUN = 3.828e26       # Solar luminosity (W)
T_SUN = 5778           # Solar effective temperature (K)

# Conversion factors
R_SUN_TO_R_EARTH = R_SUN / R_EARTH  # ~109.2
R_SUN_TO_R_JUPITER = R_SUN / R_JUPITER  # ~9.95

# Planet classification thresholds (Earth radii)
PLANET_CLASSES = {
    'sub-Earth': (0, 0.8),
    'Earth-sized': (0.8, 1.25),
    'super-Earth': (1.25, 2.0),
    'mini-Neptune': (2.0, 4.0),
    'Neptune-sized': (4.0, 6.0),
    'sub-Saturn': (6.0, 8.0),
    'Saturn-sized': (8.0, 10.0),
    'Jupiter-sized': (10.0, 15.0),
    'super-Jupiter': (15.0, float('inf'))
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PlanetParameters:
    """Derived planet parameters from transit observations."""
    radius_earth: float              # Planet radius in Earth radii
    radius_jupiter: Optional[float]  # Planet radius in Jupiter radii
    radius_err: Optional[float]      # Uncertainty in Earth radii
    transit_depth: float             # Transit depth (fractional)
    transit_depth_ppm: float         # Transit depth in ppm
    stellar_radius: float            # Stellar radius in R_sun
    planet_class: str                # Classification (super-Earth, etc.)

    # Optional derived parameters
    period: Optional[float] = None          # Orbital period (days)
    semi_major_axis: Optional[float] = None # Semi-major axis (AU)
    equilibrium_temp: Optional[float] = None # Equilibrium temperature (K)
    insolation: Optional[float] = None      # Insolation (Earth units)
    in_habitable_zone: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'radius_earth': round(self.radius_earth, 3),
            'radius_jupiter': round(self.radius_jupiter, 4) if self.radius_jupiter else None,
            'radius_err_earth': round(self.radius_err, 3) if self.radius_err else None,
            'transit_depth': round(self.transit_depth, 8),
            'transit_depth_ppm': round(self.transit_depth_ppm, 1),
            'stellar_radius_rsun': round(self.stellar_radius, 3),
            'planet_class': self.planet_class,
            'period_days': round(self.period, 6) if self.period else None,
            'semi_major_axis_au': round(self.semi_major_axis, 4) if self.semi_major_axis else None,
            'equilibrium_temp_K': round(self.equilibrium_temp, 0) if self.equilibrium_temp else None,
            'insolation_earth': round(self.insolation, 2) if self.insolation else None,
            'in_habitable_zone': self.in_habitable_zone
        }

    def __str__(self) -> str:
        hz = " (in HZ)" if self.in_habitable_zone else ""
        return f"Planet: {self.radius_earth:.2f} R_Earth ({self.planet_class}){hz}"


@dataclass
class HabitableZone:
    """Habitable zone boundaries for a star."""
    inner_au: float          # Inner edge (AU)
    outer_au: float          # Outer edge (AU)
    inner_conservative: float  # Conservative inner (AU)
    outer_conservative: float  # Conservative outer (AU)
    stellar_teff: float
    stellar_luminosity: float

    def contains(self, distance_au: float) -> bool:
        """Check if distance is within habitable zone."""
        return self.inner_au <= distance_au <= self.outer_au

    def contains_conservative(self, distance_au: float) -> bool:
        """Check if distance is within conservative habitable zone."""
        return self.inner_conservative <= distance_au <= self.outer_conservative

    def to_dict(self) -> Dict[str, Any]:
        return {
            'inner_au': round(self.inner_au, 4),
            'outer_au': round(self.outer_au, 4),
            'inner_conservative_au': round(self.inner_conservative, 4),
            'outer_conservative_au': round(self.outer_conservative, 4),
            'stellar_teff_K': round(self.stellar_teff, 0),
            'stellar_luminosity_lsun': round(self.stellar_luminosity, 4)
        }


# ============================================================================
# Planet Radius Calculator
# ============================================================================

class PlanetRadiusCalculator:
    """
    Calculate planet radius from transit depth and stellar radius.

    The transit depth is related to planet and star radii by:
        depth = (R_p / R_star)^2

    Therefore:
        R_p = R_star * sqrt(depth)

    Example:
        >>> calc = PlanetRadiusCalculator()
        >>> result = calc.from_transit_depth(depth=0.0001, stellar_radius=1.0)
        >>> print(f"Planet radius: {result.radius_earth:.2f} R_Earth")
    """

    def __init__(self):
        logger.info("PlanetRadiusCalculator initialized")

    def from_transit_depth(
        self,
        depth: float,
        stellar_radius: float,
        depth_err: Optional[float] = None,
        stellar_radius_err: Optional[float] = None,
        period: Optional[float] = None,
        stellar_mass: Optional[float] = None,
        stellar_teff: Optional[float] = None,
        stellar_luminosity: Optional[float] = None
    ) -> PlanetParameters:
        """
        Calculate planet radius from transit depth.

        Args:
            depth: Transit depth (fractional, e.g., 0.0001 for 100 ppm)
            stellar_radius: Stellar radius in solar radii
            depth_err: Transit depth uncertainty
            stellar_radius_err: Stellar radius uncertainty (R_sun)
            period: Orbital period (days) - for semi-major axis calculation
            stellar_mass: Stellar mass (M_sun) - for semi-major axis
            stellar_teff: Stellar effective temperature (K) - for Teq
            stellar_luminosity: Stellar luminosity (L_sun) - for insolation

        Returns:
            PlanetParameters with derived properties
        """
        if depth <= 0:
            raise ValueError("Transit depth must be positive")
        if stellar_radius <= 0:
            raise ValueError("Stellar radius must be positive")

        # Calculate planet radius
        # R_p / R_star = sqrt(depth)
        rp_rs = np.sqrt(depth)  # Planet-to-star radius ratio

        # Planet radius in solar radii
        radius_rsun = stellar_radius * rp_rs

        # Convert to Earth and Jupiter radii
        radius_earth = radius_rsun * R_SUN_TO_R_EARTH
        radius_jupiter = radius_rsun * R_SUN_TO_R_JUPITER

        # Uncertainty propagation
        radius_err = None
        if depth_err is not None or stellar_radius_err is not None:
            # Error propagation: sigma_Rp = Rp * sqrt((sigma_depth/2*depth)^2 + (sigma_Rs/Rs)^2)
            rel_err_depth = (depth_err / (2 * depth))**2 if depth_err else 0
            rel_err_rs = (stellar_radius_err / stellar_radius)**2 if stellar_radius_err else 0
            radius_err = radius_earth * np.sqrt(rel_err_depth + rel_err_rs)

        # Classify planet
        planet_class = self._classify_planet(radius_earth)

        # Create result
        result = PlanetParameters(
            radius_earth=radius_earth,
            radius_jupiter=radius_jupiter,
            radius_err=radius_err,
            transit_depth=depth,
            transit_depth_ppm=depth * 1e6,
            stellar_radius=stellar_radius,
            planet_class=planet_class
        )

        # Calculate orbital parameters if period provided
        if period is not None and stellar_mass is not None:
            result.period = period
            result.semi_major_axis = self._calculate_semi_major_axis(period, stellar_mass)

            # Calculate equilibrium temperature and insolation
            if stellar_teff is not None:
                result.equilibrium_temp = self._calculate_equilibrium_temp(
                    result.semi_major_axis, stellar_radius, stellar_teff
                )

            if stellar_luminosity is not None:
                result.insolation = self._calculate_insolation(
                    result.semi_major_axis, stellar_luminosity
                )

                # Check habitable zone
                hz = self.calculate_habitable_zone(stellar_teff or T_SUN, stellar_luminosity)
                result.in_habitable_zone = hz.contains(result.semi_major_axis)

        logger.info(f"Planet radius: {radius_earth:.2f} R_Earth ({planet_class})")
        return result

    def from_depth_ppm(
        self,
        depth_ppm: float,
        stellar_radius: float,
        **kwargs
    ) -> PlanetParameters:
        """
        Calculate planet radius from transit depth in ppm.

        Args:
            depth_ppm: Transit depth in parts per million
            stellar_radius: Stellar radius in solar radii
            **kwargs: Additional arguments passed to from_transit_depth

        Returns:
            PlanetParameters
        """
        depth = depth_ppm / 1e6
        depth_err = kwargs.pop('depth_err_ppm', None)
        if depth_err is not None:
            kwargs['depth_err'] = depth_err / 1e6
        return self.from_transit_depth(depth, stellar_radius, **kwargs)

    def _classify_planet(self, radius_earth: float) -> str:
        """Classify planet by radius."""
        for class_name, (r_min, r_max) in PLANET_CLASSES.items():
            if r_min <= radius_earth < r_max:
                return class_name
        return 'unknown'

    def _calculate_semi_major_axis(self, period_days: float, stellar_mass: float) -> float:
        """
        Calculate semi-major axis from Kepler's 3rd law.

        a^3 / P^2 = M_star (in solar units with P in years, a in AU)

        Args:
            period_days: Orbital period in days
            stellar_mass: Stellar mass in solar masses

        Returns:
            Semi-major axis in AU
        """
        period_years = period_days / 365.25
        a_cubed = stellar_mass * period_years**2
        return a_cubed**(1/3)

    def _calculate_equilibrium_temp(
        self,
        semi_major_axis: float,
        stellar_radius: float,
        stellar_teff: float,
        albedo: float = 0.3
    ) -> float:
        """
        Calculate planet equilibrium temperature.

        T_eq = T_star * sqrt(R_star / (2 * a)) * (1 - A)^0.25

        Args:
            semi_major_axis: Semi-major axis in AU
            stellar_radius: Stellar radius in R_sun
            stellar_teff: Stellar effective temperature in K
            albedo: Bond albedo (default 0.3)

        Returns:
            Equilibrium temperature in K
        """
        # Convert to consistent units
        a_rsun = semi_major_axis * AU / R_SUN  # Convert AU to R_sun

        t_eq = stellar_teff * np.sqrt(stellar_radius / (2 * a_rsun)) * (1 - albedo)**0.25
        return t_eq

    def _calculate_insolation(self, semi_major_axis: float, stellar_luminosity: float) -> float:
        """
        Calculate insolation relative to Earth.

        S = L_star / a^2 (where L is in L_sun and a in AU)

        Args:
            semi_major_axis: Semi-major axis in AU
            stellar_luminosity: Stellar luminosity in L_sun

        Returns:
            Insolation in Earth units (S_Earth = 1)
        """
        return stellar_luminosity / semi_major_axis**2

    def calculate_habitable_zone(
        self,
        stellar_teff: float,
        stellar_luminosity: float
    ) -> HabitableZone:
        """
        Calculate habitable zone boundaries.

        Based on Kopparapu et al. (2013) empirical relations.

        Args:
            stellar_teff: Stellar effective temperature (K)
            stellar_luminosity: Stellar luminosity (L_sun)

        Returns:
            HabitableZone with inner and outer boundaries
        """
        # Coefficients from Kopparapu et al. (2013)
        # Moist greenhouse (inner edge)
        s_eff_inner = 1.0140 + 8.1774e-5 * (stellar_teff - 5780) + 1.7063e-9 * (stellar_teff - 5780)**2
        # Maximum greenhouse (outer edge)
        s_eff_outer = 0.3438 + 5.8942e-5 * (stellar_teff - 5780) + 1.6558e-9 * (stellar_teff - 5780)**2

        # Conservative limits
        # Runaway greenhouse (conservative inner)
        s_eff_inner_con = 1.0512 + 1.3242e-4 * (stellar_teff - 5780) + 1.5418e-8 * (stellar_teff - 5780)**2
        # Early Mars (conservative outer)
        s_eff_outer_con = 0.3179 + 5.4513e-5 * (stellar_teff - 5780) + 1.5313e-9 * (stellar_teff - 5780)**2

        # Calculate distances
        inner_au = np.sqrt(stellar_luminosity / s_eff_inner)
        outer_au = np.sqrt(stellar_luminosity / s_eff_outer)
        inner_conservative = np.sqrt(stellar_luminosity / s_eff_inner_con)
        outer_conservative = np.sqrt(stellar_luminosity / s_eff_outer_con)

        return HabitableZone(
            inner_au=inner_au,
            outer_au=outer_au,
            inner_conservative=inner_conservative,
            outer_conservative=outer_conservative,
            stellar_teff=stellar_teff,
            stellar_luminosity=stellar_luminosity
        )


# ============================================================================
# Skill Registration
# ============================================================================

SKILL_INFO = {
    'PLANET-001': {
        'id': 'PLANET-001',
        'name': 'Radius Estimation',
        'command': 'planet radius',
        'class': PlanetRadiusCalculator,
        'description': 'Calculate planet radius from transit depth'
    },
    'PLANET-005': {
        'id': 'PLANET-005',
        'name': 'Habitability Assessment',
        'command': 'planet habitable',
        'class': PlanetRadiusCalculator,
        'description': 'Check if planet is in habitable zone'
    }
}


# ============================================================================
# CLI Functions
# ============================================================================

def calculate_planet_radius(
    depth: float,
    stellar_radius: float,
    **kwargs
) -> PlanetParameters:
    """Calculate planet radius (convenience function)."""
    calc = PlanetRadiusCalculator()
    return calc.from_transit_depth(depth, stellar_radius, **kwargs)


def calculate_habitable_zone(
    stellar_teff: float,
    stellar_luminosity: float
) -> HabitableZone:
    """Calculate habitable zone boundaries (convenience function)."""
    calc = PlanetRadiusCalculator()
    return calc.calculate_habitable_zone(stellar_teff, stellar_luminosity)


def is_in_habitable_zone(
    semi_major_axis: float,
    stellar_teff: float,
    stellar_luminosity: float
) -> bool:
    """Check if orbit is in habitable zone (convenience function)."""
    hz = calculate_habitable_zone(stellar_teff, stellar_luminosity)
    return hz.contains(semi_major_axis)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Planet Radius Calculator...")
    print("=" * 50)

    calc = PlanetRadiusCalculator()

    # Test 1: Earth-like transit around Sun-like star
    print("\n1. Earth transiting the Sun:")
    # Earth's transit depth would be (R_Earth/R_Sun)^2 = (1/109.2)^2 = 84 ppm
    result = calc.from_depth_ppm(
        depth_ppm=84,
        stellar_radius=1.0,
        period=365.25,
        stellar_mass=1.0,
        stellar_teff=5778,
        stellar_luminosity=1.0
    )
    print(f"   {result}")
    print(f"   Period: {result.period:.1f} days")
    print(f"   Semi-major axis: {result.semi_major_axis:.3f} AU")
    print(f"   Equilibrium temp: {result.equilibrium_temp:.0f} K")
    print(f"   In habitable zone: {result.in_habitable_zone}")

    # Test 2: Hot Jupiter
    print("\n2. Hot Jupiter (1% depth, 3-day period):")
    result = calc.from_depth_ppm(
        depth_ppm=10000,  # 1% depth
        stellar_radius=1.0,
        period=3.0,
        stellar_mass=1.0,
        stellar_teff=5778,
        stellar_luminosity=1.0
    )
    print(f"   {result}")
    print(f"   Radius: {result.radius_jupiter:.2f} R_Jupiter")
    print(f"   Equilibrium temp: {result.equilibrium_temp:.0f} K")

    # Test 3: Super-Earth around M dwarf
    print("\n3. Super-Earth around M dwarf (TRAPPIST-1e like):")
    result = calc.from_depth_ppm(
        depth_ppm=7000,  # 0.7% depth
        stellar_radius=0.12,  # M8 dwarf
        period=6.1,
        stellar_mass=0.09,
        stellar_teff=2566,
        stellar_luminosity=0.000553
    )
    print(f"   {result}")
    print(f"   Equilibrium temp: {result.equilibrium_temp:.0f} K")
    print(f"   In habitable zone: {result.in_habitable_zone}")

    # Test 4: Habitable zone boundaries
    print("\n4. Habitable Zone for Sun:")
    hz = calc.calculate_habitable_zone(5778, 1.0)
    print(f"   Inner edge: {hz.inner_au:.3f} AU")
    print(f"   Outer edge: {hz.outer_au:.3f} AU")
    print(f"   Earth at 1 AU: {hz.contains(1.0)}")

    print("\n5. Habitable Zone for TRAPPIST-1:")
    hz = calc.calculate_habitable_zone(2566, 0.000553)
    print(f"   Inner edge: {hz.inner_au:.4f} AU")
    print(f"   Outer edge: {hz.outer_au:.4f} AU")
