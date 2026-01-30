"""
LARUN Skill: Transit Model Fitting
===================================
Precise transit model fitting using batman package.

Skill ID: PLANET-010
Commands: larun fit, /fit

Created by: Padmanaban Veeraragavalu (Larun Engineering)
Reference: docs/research/EXOPLANET_DETECTION.md
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

# Check for batman availability
try:
    import batman
    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False
    logger.warning("batman-package not installed. Install with: pip install batman-package")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TransitFitResult:
    """Result of transit model fitting."""
    # Fitted parameters
    rp_rs: float       # Planet/star radius ratio
    a_rs: float        # Semi-major axis / star radius
    inc: float         # Orbital inclination (degrees)
    t0: float          # Mid-transit time (BJD)
    period: float      # Orbital period (days)
    
    # Limb darkening
    u1: float = 0.3
    u2: float = 0.2
    
    # Fit quality
    chi2: float = 0.0
    chi2_reduced: float = 0.0
    bic: float = 0.0
    
    # Uncertainties (if available)
    rp_rs_err: float = 0.0
    a_rs_err: float = 0.0
    inc_err: float = 0.0
    t0_err: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rp_rs': round(self.rp_rs, 6),
            'a_rs': round(self.a_rs, 3),
            'inclination_deg': round(self.inc, 2),
            't0_bjd': round(self.t0, 6),
            'period_days': round(self.period, 6),
            'limb_dark': [round(self.u1, 3), round(self.u2, 3)],
            'chi2': round(self.chi2, 2),
            'chi2_reduced': round(self.chi2_reduced, 3),
            'bic': round(self.bic, 2),
            'depth_ppm': round(self.rp_rs**2 * 1e6, 1),
            'uncertainties': {
                'rp_rs': round(self.rp_rs_err, 6),
                'a_rs': round(self.a_rs_err, 3),
                'inc': round(self.inc_err, 2),
                't0': round(self.t0_err, 6)
            }
        }
    
    def planet_params(self, star_radius_solar: float = 1.0) -> Dict[str, Any]:
        """
        Estimate planet physical parameters.
        
        Args:
            star_radius_solar: Star radius in solar radii
            
        Returns:
            Dict with planet radius, semi-major axis
        """
        SOLAR_RADIUS_EARTH = 109.2
        AU_SOLAR_RADII = 215.0
        
        rp_earth = self.rp_rs * star_radius_solar * SOLAR_RADIUS_EARTH
        a_au = self.a_rs * star_radius_solar / AU_SOLAR_RADII
        
        return {
            'radius_earth': round(rp_earth, 2),
            'semi_major_axis_au': round(a_au, 4),
            'impact_parameter': round(self.a_rs * np.cos(np.radians(self.inc)), 3)
        }
    
    def summary(self) -> str:
        """Return human-readable summary."""
        depth_ppm = self.rp_rs**2 * 1e6
        lines = [
            f"Transit Fit Result:",
            f"  Rp/Rs: {self.rp_rs:.4f} (depth: {depth_ppm:.0f} ppm)",
            f"  a/Rs:  {self.a_rs:.2f}",
            f"  Inc:   {self.inc:.1f}°",
            f"  T0:    {self.t0:.4f} BJD",
            f"  χ² reduced: {self.chi2_reduced:.3f}"
        ]
        return "\n".join(lines)


# ============================================================================
# Transit Fitter
# ============================================================================

class TransitFitter:
    """
    Transit model fitting using batman.
    
    Fits a Mandel & Agol (2002) transit model to light curve data
    using scipy optimization.
    
    Based on: Kreidberg (2015) - batman
    Reference: docs/research/EXOPLANET_DETECTION.md
    
    Example:
        >>> fitter = TransitFitter()
        >>> result = fitter.fit(time, flux, period=3.5, t0_init=2458000.0)
        >>> print(result.summary())
    """

    def __init__(
        self,
        limb_dark_model: str = "quadratic",
        limb_dark_coeffs: Tuple[float, float] = (0.3, 0.2)
    ):
        """
        Initialize fitter.
        
        Args:
            limb_dark_model: Limb darkening law ("quadratic", "linear", etc.)
            limb_dark_coeffs: Limb darkening coefficients
        """
        self.limb_dark_model = limb_dark_model
        self.limb_dark_coeffs = limb_dark_coeffs

    def create_model(
        self,
        time: np.ndarray,
        t0: float,
        period: float,
        rp_rs: float,
        a_rs: float,
        inc: float,
        ecc: float = 0.0,
        omega: float = 90.0
    ) -> np.ndarray:
        """
        Create transit model using batman.
        
        Args:
            time: Time array
            t0: Mid-transit time
            period: Orbital period
            rp_rs: Planet/star radius ratio
            a_rs: Semi-major axis / star radius
            inc: Inclination (degrees)
            ecc: Eccentricity
            omega: Argument of periastron
            
        Returns:
            Model flux array
        """
        if not BATMAN_AVAILABLE:
            return self._create_model_fallback(
                time, t0, period, rp_rs, a_rs, inc
            )
        
        params = batman.TransitParams()
        params.t0 = t0
        params.per = period
        params.rp = rp_rs
        params.a = a_rs
        params.inc = inc
        params.ecc = ecc
        params.w = omega
        params.u = list(self.limb_dark_coeffs)
        params.limb_dark = self.limb_dark_model
        
        m = batman.TransitModel(params, time)
        return m.light_curve(params)

    def _create_model_fallback(
        self,
        time: np.ndarray,
        t0: float,
        period: float,
        rp_rs: float,
        a_rs: float,
        inc: float
    ) -> np.ndarray:
        """
        Simple box transit model (fallback when batman not available).
        """
        # Phase fold
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        
        # Transit duration estimate
        b = a_rs * np.cos(np.radians(inc))  # Impact parameter
        if b > 1:
            return np.ones_like(time)
        
        duration_phase = rp_rs / (np.pi * a_rs) * np.sqrt(1 - b**2)
        
        # Simple box model
        in_transit = np.abs(phase) < duration_phase / 2
        model = np.ones_like(time)
        model[in_transit] = 1.0 - rp_rs**2
        
        return model

    def fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0_init: float,
        flux_err: Optional[np.ndarray] = None,
        rp_init: float = 0.1,
        a_init: float = 10.0,
        inc_init: float = 89.0
    ) -> TransitFitResult:
        """
        Fit transit model to data.
        
        Args:
            time: Time array (BJD)
            flux: Normalized flux array
            period: Orbital period (days)
            t0_init: Initial mid-transit time
            flux_err: Optional flux uncertainties
            rp_init: Initial Rp/Rs guess
            a_init: Initial a/Rs guess
            inc_init: Initial inclination guess
            
        Returns:
            TransitFitResult
        """
        from scipy.optimize import minimize, differential_evolution
        
        logger.info(f"Fitting transit: P={period:.4f}d, t0_init={t0_init:.4f}")
        
        # Use median flux error if not provided
        if flux_err is None:
            flux_err = np.ones_like(flux) * np.std(flux) * 1.4826
        
        # Phase fold data for initial estimates
        phase = ((time - t0_init) % period) / period
        phase[phase > 0.5] -= 1.0
        
        # Estimate initial depth
        in_transit = np.abs(phase) < 0.03
        out_transit = np.abs(phase) > 0.1
        if np.sum(in_transit) > 5 and np.sum(out_transit) > 20:
            depth = 1.0 - np.mean(flux[in_transit]) / np.mean(flux[out_transit])
            rp_init = np.sqrt(max(depth, 0.0001))
        
        def neg_log_likelihood(theta):
            t0, rp, a, inc = theta
            
            # Physical bounds
            if rp <= 0 or rp > 0.5 or a < 1 or a > 100 or inc < 60 or inc > 90:
                return 1e10
            
            model = self.create_model(time, t0, period, rp, a, inc)
            residuals = (flux - model) / flux_err
            return 0.5 * np.sum(residuals**2)
        
        # Initial guess
        x0 = [t0_init, rp_init, a_init, inc_init]
        
        # Bounds
        bounds = [
            (t0_init - 0.1, t0_init + 0.1),
            (0.001, 0.3),
            (2, 50),
            (70, 90)
        ]
        
        # Try differential evolution for global optimization
        try:
            result = differential_evolution(
                neg_log_likelihood,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-5,
                workers=1
            )
            best_theta = result.x
            chi2 = result.fun * 2
        except:
            # Fall back to local minimization
            result = minimize(
                neg_log_likelihood,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000}
            )
            best_theta = result.x
            chi2 = result.fun * 2
        
        t0_fit, rp_fit, a_fit, inc_fit = best_theta
        
        # Calculate reduced chi2
        n_params = 4
        n_data = len(time)
        chi2_reduced = chi2 / (n_data - n_params)
        
        # BIC
        bic = chi2 + n_params * np.log(n_data)
        
        # Estimate uncertainties from Hessian (simplified)
        try:
            from scipy.optimize import approx_fprime
            eps = 1e-6
            hess_diag = []
            for i in range(4):
                def func_i(x):
                    theta = best_theta.copy()
                    theta[i] = x
                    return neg_log_likelihood(theta)
                h = approx_fprime([best_theta[i]], func_i, eps)[0]
                hess_diag.append(abs(h) + 1e-10)
            
            t0_err = 1.0 / np.sqrt(hess_diag[0])
            rp_err = 1.0 / np.sqrt(hess_diag[1])
            a_err = 1.0 / np.sqrt(hess_diag[2])
            inc_err = 1.0 / np.sqrt(hess_diag[3])
        except:
            t0_err = rp_err = a_err = inc_err = 0.0
        
        logger.info(f"Fit complete: Rp/Rs={rp_fit:.4f}, a/Rs={a_fit:.1f}, inc={inc_fit:.1f}°")
        
        return TransitFitResult(
            rp_rs=rp_fit,
            a_rs=a_fit,
            inc=inc_fit,
            t0=t0_fit,
            period=period,
            u1=self.limb_dark_coeffs[0],
            u2=self.limb_dark_coeffs[1],
            chi2=chi2,
            chi2_reduced=chi2_reduced,
            bic=bic,
            rp_rs_err=rp_err,
            a_rs_err=a_err,
            inc_err=inc_err,
            t0_err=t0_err
        )

    def fit_phase_folded(
        self,
        phase: np.ndarray,
        flux: np.ndarray,
        period: float,
        flux_err: Optional[np.ndarray] = None
    ) -> TransitFitResult:
        """
        Fit transit model to phase-folded data.
        
        Args:
            phase: Phase array (-0.5 to 0.5)
            flux: Binned flux array
            period: Period for converting phase to time
            flux_err: Optional uncertainties
            
        Returns:
            TransitFitResult
        """
        # Convert phase to time-like array
        time = phase * period
        return self.fit(time, flux, period, 0.0, flux_err)


# ============================================================================
# Convenience Functions
# ============================================================================

def fit_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    **kwargs
) -> TransitFitResult:
    """
    Convenience function for transit fitting.
    """
    fitter = TransitFitter(**kwargs)
    return fitter.fit(time, flux, period, t0)


# ============================================================================
# Skill Registration
# ============================================================================

SKILL_INFO = {
    'PLANET-010': {
        'id': 'PLANET-010',
        'name': 'Transit Model Fitting',
        'command': 'fit',
        'class': TransitFitter,
        'description': 'Precise transit model fitting using batman'
    }
}


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Transit Fitter...")
    print(f"Batman available: {BATMAN_AVAILABLE}")
    
    np.random.seed(42)
    
    # Generate synthetic transit
    period = 3.5
    t0 = 1.0
    rp_true = 0.1   # 10% radius ratio
    a_true = 12.0
    inc_true = 88.5
    
    # Time array around transit
    time = np.linspace(t0 - 0.2, t0 + 0.2, 500)
    
    # Create model
    fitter = TransitFitter()
    model = fitter.create_model(time, t0, period, rp_true, a_true, inc_true)
    
    # Add noise
    noise = 0.001
    flux = model + np.random.normal(0, noise, len(time))
    
    # Fit
    result = fitter.fit(time, flux, period, t0 + 0.01)  # Slight offset in t0
    
    print(result.summary())
    print(f"\nTrue values: Rp/Rs={rp_true}, a/Rs={a_true}, inc={inc_true}")
    print(f"Fitted:      Rp/Rs={result.rp_rs:.4f}, a/Rs={result.a_rs:.1f}, inc={result.inc:.1f}")
