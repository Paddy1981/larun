"""
LARUN Skill: Stellar Classification
====================================
Classify stars by spectral type (OBAFGKM) based on temperature and photometry.

Skill IDs: STAR-001 (Classification), STAR-002 (Teff), STAR-003 (Radius)
Commands: larun stellar classify, larun stellar teff, larun stellar radius

Created by: Padmanaban Veeraragavalu (Larun Engineering)
With AI assistance from: Claude (Anthropic)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Stellar Classification Data
# ============================================================================

# Spectral type temperature ranges (K)
# Reference: Pecaut & Mamajek (2013), updated values
SPECTRAL_TYPES = {
    'O': {'teff_min': 30000, 'teff_max': 50000, 'color': 'blue', 'mass_solar': (16, 150), 'radius_solar': (6.6, 20)},
    'B': {'teff_min': 10000, 'teff_max': 30000, 'color': 'blue-white', 'mass_solar': (2.1, 16), 'radius_solar': (1.8, 6.6)},
    'A': {'teff_min': 7500, 'teff_max': 10000, 'color': 'white', 'mass_solar': (1.4, 2.1), 'radius_solar': (1.4, 1.8)},
    'F': {'teff_min': 6000, 'teff_max': 7500, 'color': 'yellow-white', 'mass_solar': (1.04, 1.4), 'radius_solar': (1.15, 1.4)},
    'G': {'teff_min': 5200, 'teff_max': 6000, 'color': 'yellow', 'mass_solar': (0.8, 1.04), 'radius_solar': (0.96, 1.15)},
    'K': {'teff_min': 3700, 'teff_max': 5200, 'color': 'orange', 'mass_solar': (0.45, 0.8), 'radius_solar': (0.7, 0.96)},
    'M': {'teff_min': 2400, 'teff_max': 3700, 'color': 'red', 'mass_solar': (0.08, 0.45), 'radius_solar': (0.1, 0.7)},
}

# Subtype temperature boundaries (approximate)
SUBTYPE_BOUNDARIES = {
    'O': [50000, 45000, 42000, 39000, 36000, 33000, 31500, 30500, 30000, 30000],
    'B': [30000, 25000, 20000, 17000, 15000, 14000, 13500, 12500, 11500, 10500],
    'A': [10000, 9500, 9000, 8500, 8250, 8000, 7750, 7500, 7350, 7200],
    'F': [7500, 7200, 6900, 6650, 6400, 6200, 6050, 5950, 5800, 5700],
    'G': [6000, 5900, 5800, 5700, 5600, 5500, 5400, 5350, 5300, 5250],
    'K': [5200, 5000, 4800, 4600, 4400, 4200, 4000, 3900, 3800, 3750],
    'M': [3700, 3500, 3350, 3200, 3050, 2900, 2750, 2600, 2500, 2400],
}

# Luminosity class criteria
LUMINOSITY_CLASSES = {
    'Ia': 'Bright supergiant',
    'Ib': 'Supergiant',
    'II': 'Bright giant',
    'III': 'Giant',
    'IV': 'Subgiant',
    'V': 'Main sequence (dwarf)',
    'VI': 'Subdwarf',
    'VII': 'White dwarf',
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StellarParameters:
    """Stellar parameters result."""
    teff: float                    # Effective temperature (K)
    teff_err: Optional[float]      # Temperature uncertainty
    logg: Optional[float]          # Surface gravity (log10 cm/s^2)
    logg_err: Optional[float]      # Gravity uncertainty
    metallicity: Optional[float]   # [Fe/H]
    metallicity_err: Optional[float]
    radius: Optional[float]        # Stellar radius (R_sun)
    radius_err: Optional[float]
    mass: Optional[float]          # Stellar mass (M_sun)
    mass_err: Optional[float]
    luminosity: Optional[float]    # Stellar luminosity (L_sun)
    spectral_type: str             # Full spectral type (e.g., "G2V")
    spectral_class: str            # Main class (e.g., "G")
    subtype: int                   # Numeric subtype (e.g., 2)
    luminosity_class: str          # Luminosity class (e.g., "V")
    source: str                    # Data source

    def to_dict(self) -> Dict[str, Any]:
        return {
            'teff_K': round(self.teff, 0),
            'teff_err_K': round(self.teff_err, 0) if self.teff_err else None,
            'logg': round(self.logg, 2) if self.logg else None,
            'logg_err': round(self.logg_err, 2) if self.logg_err else None,
            'metallicity_feh': round(self.metallicity, 2) if self.metallicity else None,
            'metallicity_err': round(self.metallicity_err, 2) if self.metallicity_err else None,
            'radius_rsun': round(self.radius, 3) if self.radius else None,
            'radius_err_rsun': round(self.radius_err, 3) if self.radius_err else None,
            'mass_msun': round(self.mass, 3) if self.mass else None,
            'mass_err_msun': round(self.mass_err, 3) if self.mass_err else None,
            'luminosity_lsun': round(self.luminosity, 4) if self.luminosity else None,
            'spectral_type': self.spectral_type,
            'spectral_class': self.spectral_class,
            'subtype': self.subtype,
            'luminosity_class': self.luminosity_class,
            'luminosity_class_name': LUMINOSITY_CLASSES.get(self.luminosity_class, 'Unknown'),
            'source': self.source
        }

    def __str__(self) -> str:
        return (f"Stellar: {self.spectral_type}, Teff={self.teff:.0f}K, "
                f"R={self.radius:.2f}R_sun" if self.radius else f"Stellar: {self.spectral_type}, Teff={self.teff:.0f}K")


# ============================================================================
# Stellar Classifier
# ============================================================================

class StellarClassifier:
    """
    Classify stars by spectral type based on temperature and other parameters.

    Based on Morgan-Keenan (MK) spectral classification system.

    Example:
        >>> classifier = StellarClassifier()
        >>> result = classifier.classify_from_teff(5778)
        >>> print(result.spectral_type)  # "G2V"
    """

    def __init__(self):
        logger.info("StellarClassifier initialized")

    def classify_from_teff(
        self,
        teff: float,
        logg: Optional[float] = None,
        metallicity: Optional[float] = None,
        radius: Optional[float] = None,
        source: str = "manual"
    ) -> StellarParameters:
        """
        Classify a star based on effective temperature.

        Args:
            teff: Effective temperature in Kelvin
            logg: Surface gravity (log g) - used for luminosity class
            metallicity: [Fe/H] metallicity
            radius: Stellar radius in solar radii
            source: Data source identifier

        Returns:
            StellarParameters with classification
        """
        if teff <= 0:
            raise ValueError("Temperature must be positive")

        # Determine spectral class
        spectral_class = self._get_spectral_class(teff)

        # Determine subtype
        subtype = self._get_subtype(teff, spectral_class)

        # Determine luminosity class from logg
        luminosity_class = self._get_luminosity_class(logg, teff)

        # Full spectral type string
        spectral_type = f"{spectral_class}{subtype}{luminosity_class}"

        # Estimate mass if not provided
        mass = self._estimate_mass(spectral_class, subtype, luminosity_class)

        # Estimate radius if not provided
        if radius is None:
            radius = self._estimate_radius(spectral_class, subtype, luminosity_class)

        # Estimate luminosity from Teff and radius
        luminosity = self._calculate_luminosity(teff, radius) if radius else None

        logger.info(f"Classified: {spectral_type}, Teff={teff:.0f}K")

        return StellarParameters(
            teff=teff,
            teff_err=None,
            logg=logg,
            logg_err=None,
            metallicity=metallicity,
            metallicity_err=None,
            radius=radius,
            radius_err=None,
            mass=mass,
            mass_err=None,
            luminosity=luminosity,
            spectral_type=spectral_type,
            spectral_class=spectral_class,
            subtype=subtype,
            luminosity_class=luminosity_class,
            source=source
        )

    def _get_spectral_class(self, teff: float) -> str:
        """Determine main spectral class from temperature."""
        for spec_class, props in SPECTRAL_TYPES.items():
            if props['teff_min'] <= teff < props['teff_max']:
                return spec_class

        # Edge cases
        if teff >= 50000:
            return 'O'
        elif teff < 2400:
            return 'M'

        return 'G'  # Default to solar-like

    def _get_subtype(self, teff: float, spectral_class: str) -> int:
        """Determine numeric subtype (0-9) within spectral class."""
        boundaries = SUBTYPE_BOUNDARIES.get(spectral_class, [])

        if not boundaries:
            return 5  # Default to middle

        for i, temp_boundary in enumerate(boundaries):
            if teff >= temp_boundary:
                return i

        return 9  # Coolest subtype

    def _get_luminosity_class(self, logg: Optional[float], teff: float) -> str:
        """Determine luminosity class from surface gravity."""
        if logg is None:
            # Default to main sequence
            return 'V'

        # Approximate boundaries (vary with Teff)
        if logg >= 4.0:
            return 'V'    # Main sequence
        elif logg >= 3.5:
            return 'IV'   # Subgiant
        elif logg >= 2.5:
            return 'III'  # Giant
        elif logg >= 1.5:
            return 'II'   # Bright giant
        elif logg >= 0.5:
            return 'Ib'   # Supergiant
        else:
            return 'Ia'   # Bright supergiant

    def _estimate_mass(self, spectral_class: str, subtype: int, lum_class: str) -> Optional[float]:
        """Estimate stellar mass from spectral type."""
        props = SPECTRAL_TYPES.get(spectral_class)
        if not props:
            return None

        mass_min, mass_max = props['mass_solar']

        # Interpolate based on subtype (0 = hot/massive, 9 = cool/less massive)
        fraction = subtype / 9.0
        mass = mass_max - fraction * (mass_max - mass_min)

        # Adjust for luminosity class
        lum_factors = {
            'Ia': 3.0, 'Ib': 2.5, 'II': 2.0, 'III': 1.5,
            'IV': 1.1, 'V': 1.0, 'VI': 0.9, 'VII': 0.6
        }
        mass *= lum_factors.get(lum_class, 1.0)

        return mass

    def _estimate_radius(self, spectral_class: str, subtype: int, lum_class: str) -> Optional[float]:
        """Estimate stellar radius from spectral type."""
        props = SPECTRAL_TYPES.get(spectral_class)
        if not props:
            return None

        radius_min, radius_max = props['radius_solar']

        # Interpolate based on subtype
        fraction = subtype / 9.0
        radius = radius_max - fraction * (radius_max - radius_min)

        # Adjust for luminosity class (giants/supergiants are much larger)
        lum_factors = {
            'Ia': 100.0, 'Ib': 50.0, 'II': 20.0, 'III': 10.0,
            'IV': 2.0, 'V': 1.0, 'VI': 0.8, 'VII': 0.01
        }
        radius *= lum_factors.get(lum_class, 1.0)

        return radius

    def _calculate_luminosity(self, teff: float, radius: float) -> float:
        """
        Calculate luminosity from Stefan-Boltzmann law.
        L/L_sun = (R/R_sun)^2 * (T/T_sun)^4
        """
        T_SUN = 5778  # K
        return radius**2 * (teff / T_SUN)**4


# ============================================================================
# Temperature Estimation
# ============================================================================

class TemperatureEstimator:
    """
    Estimate stellar effective temperature from photometry.

    Methods include color-temperature relations from various surveys.
    """

    # Color-temperature relations (polynomial coefficients)
    # Reference: Casagrande et al. (2010), Huang et al. (2015)
    COLOR_RELATIONS = {
        'B-V': {
            'coeffs': [8080, -6160, 2425, -380],  # Polynomial coefficients
            'valid_range': (-0.3, 1.5),
            'description': 'Johnson B-V color'
        },
        'BP-RP': {
            'coeffs': [8500, -3500, 900, -100],  # Gaia BP-RP
            'valid_range': (0.0, 4.0),
            'description': 'Gaia BP-RP color'
        },
        'G-Ks': {
            'coeffs': [8200, -2000, 200, -10],  # Gaia G - 2MASS Ks
            'valid_range': (0.0, 6.0),
            'description': 'Gaia G minus 2MASS Ks'
        }
    }

    def __init__(self):
        logger.info("TemperatureEstimator initialized")

    def from_color(
        self,
        color_value: float,
        color_type: str = 'BP-RP',
        metallicity: float = 0.0
    ) -> Tuple[float, float]:
        """
        Estimate temperature from photometric color.

        Args:
            color_value: Color index value
            color_type: Type of color ('B-V', 'BP-RP', 'G-Ks')
            metallicity: [Fe/H] for metallicity correction

        Returns:
            Tuple of (Teff, uncertainty)
        """
        relation = self.COLOR_RELATIONS.get(color_type)
        if not relation:
            raise ValueError(f"Unknown color type: {color_type}")

        valid_min, valid_max = relation['valid_range']
        if not valid_min <= color_value <= valid_max:
            logger.warning(f"Color {color_value} outside valid range {relation['valid_range']}")

        # Polynomial evaluation
        coeffs = relation['coeffs']
        teff = sum(c * color_value**i for i, c in enumerate(coeffs))

        # Metallicity correction (approximate)
        teff += metallicity * 100

        # Constrain to physical range
        teff = max(2000, min(50000, teff))

        # Uncertainty estimate (typically 100-200 K)
        uncertainty = 150.0

        return teff, uncertainty

    def from_bp_rp(self, bp_rp: float, metallicity: float = 0.0) -> Tuple[float, float]:
        """Convenience method for Gaia BP-RP color."""
        return self.from_color(bp_rp, 'BP-RP', metallicity)


# ============================================================================
# Stellar Radius Estimator
# ============================================================================

class RadiusEstimator:
    """
    Estimate stellar radius from various methods.
    """

    # Solar constants
    STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4
    L_SUN = 3.828e26  # W
    R_SUN = 6.9634e8  # m
    T_SUN = 5778  # K

    def __init__(self):
        logger.info("RadiusEstimator initialized")

    def from_luminosity_teff(self, luminosity: float, teff: float) -> Tuple[float, float]:
        """
        Calculate radius from luminosity and temperature.

        R/R_sun = sqrt(L/L_sun) * (T_sun/T)^2

        Args:
            luminosity: Stellar luminosity in L_sun
            teff: Effective temperature in K

        Returns:
            Tuple of (radius in R_sun, uncertainty)
        """
        radius = np.sqrt(luminosity) * (self.T_SUN / teff)**2
        uncertainty = radius * 0.1  # ~10% uncertainty

        return radius, uncertainty

    def from_angular_diameter_distance(
        self,
        angular_diameter: float,  # milliarcseconds
        distance: float           # parsecs
    ) -> Tuple[float, float]:
        """
        Calculate radius from angular diameter and distance.

        R = 0.5 * angular_diameter(rad) * distance

        Args:
            angular_diameter: Angular diameter in milliarcseconds
            distance: Distance in parsecs

        Returns:
            Tuple of (radius in R_sun, uncertainty)
        """
        # Convert mas to radians
        theta_rad = angular_diameter * 1e-3 / 3600 * np.pi / 180

        # Calculate radius in AU, then convert to R_sun
        # 1 pc = 206265 AU, 1 R_sun = 0.00465 AU
        radius_au = 0.5 * theta_rad * distance * 206265
        radius = radius_au / 0.00465

        uncertainty = radius * 0.05  # ~5% uncertainty

        return radius, uncertainty

    def from_gaia_params(
        self,
        teff: float,
        logg: float,
        mass: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Estimate radius from Gaia parameters using logg.

        g = GM/R^2  =>  R = sqrt(GM/g)

        Args:
            teff: Effective temperature (K)
            logg: Log surface gravity (cgs)
            mass: Stellar mass in M_sun (estimated if not provided)

        Returns:
            Tuple of (radius in R_sun, uncertainty)
        """
        # Physical constants
        G = 6.674e-11  # m^3 kg^-1 s^-2
        M_SUN = 1.989e30  # kg
        R_SUN = 6.9634e8  # m

        # Estimate mass if not provided
        if mass is None:
            # Very rough estimate from Teff (main sequence)
            if teff > 10000:
                mass = 3.0
            elif teff > 7000:
                mass = 1.5
            elif teff > 5000:
                mass = 1.0
            elif teff > 4000:
                mass = 0.7
            else:
                mass = 0.3

        # Convert logg to g in m/s^2
        g = 10**(logg - 2)  # logg is in cgs, convert to SI

        # Calculate radius
        radius_m = np.sqrt(G * mass * M_SUN / g)
        radius = radius_m / R_SUN

        uncertainty = radius * 0.15  # ~15% uncertainty

        return radius, uncertainty


# ============================================================================
# Skill Registration
# ============================================================================

SKILL_INFO = {
    'STAR-001': {
        'id': 'STAR-001',
        'name': 'Stellar Classification',
        'command': 'stellar classify',
        'class': StellarClassifier,
        'description': 'Classify stars by spectral type (OBAFGKM)'
    },
    'STAR-002': {
        'id': 'STAR-002',
        'name': 'Effective Temperature',
        'command': 'stellar teff',
        'class': TemperatureEstimator,
        'description': 'Estimate stellar effective temperature from photometry'
    },
    'STAR-003': {
        'id': 'STAR-003',
        'name': 'Stellar Radius',
        'command': 'stellar radius',
        'class': RadiusEstimator,
        'description': 'Estimate stellar radius from various parameters'
    }
}


# ============================================================================
# CLI Functions
# ============================================================================

def classify_star(teff: float, logg: float = None, **kwargs) -> StellarParameters:
    """Classify a star by temperature (convenience function)."""
    classifier = StellarClassifier()
    return classifier.classify_from_teff(teff, logg, **kwargs)


def estimate_teff(color: float, color_type: str = 'BP-RP') -> Tuple[float, float]:
    """Estimate temperature from color (convenience function)."""
    estimator = TemperatureEstimator()
    return estimator.from_color(color, color_type)


def estimate_radius(teff: float, logg: float, mass: float = None) -> Tuple[float, float]:
    """Estimate stellar radius (convenience function)."""
    estimator = RadiusEstimator()
    return estimator.from_gaia_params(teff, logg, mass)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Stellar Classification...")
    print("=" * 50)

    # Test with solar parameters
    classifier = StellarClassifier()

    # Sun: G2V, Teff=5778K
    sun = classifier.classify_from_teff(5778, logg=4.44)
    print(f"Sun: {sun}")
    print(f"  Full type: {sun.spectral_type}")
    print(f"  Mass: {sun.mass:.2f} M_sun")
    print(f"  Radius: {sun.radius:.2f} R_sun")
    print(f"  Luminosity: {sun.luminosity:.4f} L_sun")
    print()

    # Test various stars
    test_stars = [
        ("Vega (A0V)", 9600, 4.0),
        ("Sirius (A1V)", 9940, 4.3),
        ("Arcturus (K1.5III)", 4286, 1.66),
        ("Betelgeuse (M2Ia)", 3600, 0.0),
        ("Proxima Centauri (M5.5V)", 3042, 5.2),
    ]

    for name, teff, logg in test_stars:
        result = classifier.classify_from_teff(teff, logg)
        print(f"{name}: {result.spectral_type}, Teff={result.teff:.0f}K, R={result.radius:.2f}R_sun")

    print()
    print("Temperature estimation from BP-RP color...")
    estimator = TemperatureEstimator()

    test_colors = [0.5, 0.82, 1.5, 2.5]  # BP-RP values
    for color in test_colors:
        teff, err = estimator.from_bp_rp(color)
        print(f"BP-RP={color:.2f} => Teff={teff:.0f} +/- {err:.0f} K")
