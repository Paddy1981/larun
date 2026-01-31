"""
LARUN Skills Module
===================
Skill implementations for astronomical data analysis.

Skills:
- periodogram: BLS and Lomb-Scargle periodogram analysis (ANAL-001, ANAL-010)
- stellar: Stellar classification and parameter estimation (STAR-001, STAR-002, STAR-003)
- gaia: Gaia DR3 integration and TIC cross-matching (MISSION-003)
"""

from .periodogram import (
    BLSPeriodogram,
    LombScarglePeriodogram,
    PeriodogramResult,
    TransitCandidate,
    phase_fold,
    bin_phase_curve,
    run_bls,
    run_lomb_scargle,
)

from .stellar import (
    StellarClassifier,
    StellarParameters,
    TemperatureEstimator,
    RadiusEstimator,
    classify_star,
    estimate_teff,
    estimate_radius,
    SPECTRAL_TYPES,
    LUMINOSITY_CLASSES,
)

from .gaia import (
    GaiaClient,
    GaiaSource,
    TICCrossmatcher,
    TICCrossmatch,
    query_gaia,
    get_stellar_params,
    crossmatch_tic,
)

from .planet import (
    PlanetRadiusCalculator,
    PlanetParameters,
    HabitableZone,
    calculate_planet_radius,
    calculate_habitable_zone,
    is_in_habitable_zone,
    PLANET_CLASSES,
)

from .transit_fit import (
    TransitFitter,
    TransitFitResult,
    fit_transit,
)

from .figures import (
    FigureGenerator,
    create_lightcurve_plot,
    create_periodogram_plot,
    create_phase_plot,
)

from .multiplanet import (
    MultiPlanetDetector,
    MultiPlanetResult,
    PlanetCandidate,
    detect_multiplanet,
)

__all__ = [
    # Periodogram
    'BLSPeriodogram',
    'LombScarglePeriodogram',
    'PeriodogramResult',
    'TransitCandidate',
    'phase_fold',
    'bin_phase_curve',
    'run_bls',
    'run_lomb_scargle',
    # Stellar
    'StellarClassifier',
    'StellarParameters',
    'TemperatureEstimator',
    'RadiusEstimator',
    'classify_star',
    'estimate_teff',
    'estimate_radius',
    'SPECTRAL_TYPES',
    'LUMINOSITY_CLASSES',
    # Gaia
    'GaiaClient',
    'GaiaSource',
    'TICCrossmatcher',
    'TICCrossmatch',
    'query_gaia',
    'get_stellar_params',
    'crossmatch_tic',
    # Planet
    'PlanetRadiusCalculator',
    'PlanetParameters',
    'HabitableZone',
    'calculate_planet_radius',
    'calculate_habitable_zone',
    'is_in_habitable_zone',
    'PLANET_CLASSES',
    # Transit Fitting
    'TransitFitter',
    'TransitFitResult',
    'fit_transit',
    # Figures
    'FigureGenerator',
    'create_lightcurve_plot',
    'create_periodogram_plot',
    'create_phase_plot',
    # Multi-Planet
    'MultiPlanetDetector',
    'MultiPlanetResult',
    'PlanetCandidate',
    'detect_multiplanet',
]
