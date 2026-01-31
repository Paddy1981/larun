"""
LARUN Skill: Figure Generator
=============================
Create publication-quality plots for astronomical analysis.

Skill ID: RES-003
Command: larun research plot

Created by: Padmanaban Veeraragavalu (Larun Engineering)
With AI assistance from: Claude (Anthropic)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Plot Configuration
# ============================================================================

# Publication-quality defaults
PLOT_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
}

# Color schemes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'transit': '#d62728',
    'candidate': '#2ca02c',
    'noise': '#7f7f7f',
    'model': '#9467bd',
    'residual': '#8c564b',
    'hz_inner': '#ff9896',
    'hz_outer': '#98df8a'
}


# ============================================================================
# Figure Generator
# ============================================================================

class FigureGenerator:
    """
    Generate publication-quality astronomical figures.

    Supports:
    - Light curve plots
    - Periodogram plots
    - Phase-folded transit plots
    - Detection summary figures
    - Habitable zone diagrams

    Example:
        >>> gen = FigureGenerator()
        >>> gen.plot_lightcurve(time, flux, title="TIC 12345678")
        >>> gen.save("lightcurve.png")
    """

    def __init__(self, style: str = 'publication'):
        """
        Initialize figure generator.

        Args:
            style: Plot style ('publication', 'presentation', 'default')
        """
        self.style = style
        self._fig = None
        self._axes = None
        self._setup_style()
        logger.info(f"FigureGenerator initialized with style='{style}'")

    def _setup_style(self):
        """Setup matplotlib style."""
        import matplotlib.pyplot as plt

        if self.style == 'publication':
            plt.rcParams.update(PLOT_STYLE)
        elif self.style == 'presentation':
            style = PLOT_STYLE.copy()
            style['figure.figsize'] = (12, 7)
            style['font.size'] = 14
            style['axes.labelsize'] = 16
            plt.rcParams.update(style)
        else:
            plt.style.use('default')

    def plot_lightcurve(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        title: str = "Light Curve",
        xlabel: str = "Time (days)",
        ylabel: str = "Normalized Flux",
        highlight_transits: Optional[List[Tuple[float, float]]] = None,
        model_flux: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> 'FigureGenerator':
        """
        Plot a light curve.

        Args:
            time: Time array
            flux: Flux array
            flux_err: Optional flux uncertainties
            title: Plot title
            xlabel, ylabel: Axis labels
            highlight_transits: List of (start, end) times to highlight
            model_flux: Optional model to overlay
            figsize: Figure size

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        self._fig, self._axes = plt.subplots(figsize=figsize)
        ax = self._axes

        # Plot data
        if flux_err is not None:
            ax.errorbar(time, flux, yerr=flux_err, fmt='.', color=COLORS['primary'],
                       alpha=0.5, markersize=2, elinewidth=0.5, label='Data')
        else:
            ax.scatter(time, flux, s=1, color=COLORS['primary'], alpha=0.5, label='Data')

        # Plot model if provided
        if model_flux is not None:
            ax.plot(time, model_flux, color=COLORS['model'], lw=2, label='Model')

        # Highlight transit regions
        if highlight_transits:
            for t_start, t_end in highlight_transits:
                ax.axvspan(t_start, t_end, alpha=0.2, color=COLORS['transit'], label='Transit')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

        plt.tight_layout()
        return self

    def plot_periodogram(
        self,
        periods: np.ndarray,
        power: np.ndarray,
        best_period: Optional[float] = None,
        fap_levels: Optional[Dict[float, float]] = None,
        title: str = "BLS Periodogram",
        xlabel: str = "Period (days)",
        ylabel: str = "BLS Power",
        figsize: Tuple[int, int] = (10, 5)
    ) -> 'FigureGenerator':
        """
        Plot a periodogram.

        Args:
            periods: Period array
            power: Power array
            best_period: Best period to highlight
            fap_levels: Dict of {FAP: power_threshold} for significance lines
            title: Plot title
            figsize: Figure size

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        self._fig, self._axes = plt.subplots(figsize=figsize)
        ax = self._axes

        # Plot periodogram
        ax.plot(periods, power, color=COLORS['primary'], lw=1)

        # Mark best period
        if best_period is not None:
            ax.axvline(best_period, color=COLORS['transit'], ls='--', lw=2,
                      label=f'Best: {best_period:.4f} d')

            # Mark harmonics
            for harmonic in [0.5, 2.0]:
                ax.axvline(best_period * harmonic, color=COLORS['noise'], ls=':',
                          alpha=0.5)

        # FAP levels
        if fap_levels:
            for fap, threshold in fap_levels.items():
                ax.axhline(threshold, color=COLORS['secondary'], ls='--', alpha=0.7,
                          label=f'FAP={fap}')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.set_xlim(periods.min(), periods.max())

        plt.tight_layout()
        return self

    def plot_phase_curve(
        self,
        phase: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        bin_phase: Optional[np.ndarray] = None,
        bin_flux: Optional[np.ndarray] = None,
        bin_err: Optional[np.ndarray] = None,
        model_phase: Optional[np.ndarray] = None,
        model_flux: Optional[np.ndarray] = None,
        period: Optional[float] = None,
        depth_ppm: Optional[float] = None,
        title: str = "Phase-Folded Light Curve",
        figsize: Tuple[int, int] = (10, 6)
    ) -> 'FigureGenerator':
        """
        Plot phase-folded light curve.

        Args:
            phase: Phase array (-0.5 to 0.5)
            flux: Flux array
            bin_phase, bin_flux, bin_err: Binned data
            model_phase, model_flux: Transit model
            period: Orbital period for label
            depth_ppm: Transit depth for label
            title: Plot title
            figsize: Figure size

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        self._fig, self._axes = plt.subplots(figsize=figsize)
        ax = self._axes

        # Plot unbinned data
        ax.scatter(phase, flux, s=1, color=COLORS['noise'], alpha=0.3, label='Data')

        # Plot binned data
        if bin_phase is not None and bin_flux is not None:
            if bin_err is not None:
                ax.errorbar(bin_phase, bin_flux, yerr=bin_err, fmt='o',
                           color=COLORS['primary'], markersize=5, capsize=2,
                           label='Binned')
            else:
                ax.plot(bin_phase, bin_flux, 'o', color=COLORS['primary'],
                       markersize=5, label='Binned')

        # Plot model
        if model_phase is not None and model_flux is not None:
            sort_idx = np.argsort(model_phase)
            ax.plot(model_phase[sort_idx], model_flux[sort_idx],
                   color=COLORS['model'], lw=2, label='Model')

        # Labels
        subtitle = ""
        if period:
            subtitle += f"P = {period:.4f} d"
        if depth_ppm:
            if subtitle:
                subtitle += ", "
            subtitle += f"Depth = {depth_ppm:.0f} ppm"

        ax.set_xlabel("Orbital Phase")
        ax.set_ylabel("Normalized Flux")
        ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
        ax.set_xlim(-0.5, 0.5)
        ax.legend(loc='best')

        plt.tight_layout()
        return self

    def plot_transit_fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        model_flux: np.ndarray,
        residuals: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        title: str = "Transit Fit",
        figsize: Tuple[int, int] = (10, 8)
    ) -> 'FigureGenerator':
        """
        Plot transit fit with residuals.

        Args:
            time: Time array
            flux: Observed flux
            model_flux: Model flux
            residuals: Fit residuals
            flux_err: Flux uncertainties
            title: Plot title
            figsize: Figure size

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        self._fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                              height_ratios=[3, 1],
                                              sharex=True)
        self._axes = (ax1, ax2)

        # Top panel: data and model
        ax1.scatter(time, flux, s=2, color=COLORS['primary'], alpha=0.5, label='Data')
        ax1.plot(time, model_flux, color=COLORS['model'], lw=2, label='Best fit')
        ax1.set_ylabel("Normalized Flux")
        ax1.set_title(title)
        ax1.legend(loc='best')

        # Bottom panel: residuals
        if flux_err is not None:
            ax2.errorbar(time, residuals * 1e6, yerr=flux_err * 1e6, fmt='.',
                        color=COLORS['residual'], markersize=2, elinewidth=0.5)
        else:
            ax2.scatter(time, residuals * 1e6, s=2, color=COLORS['residual'])

        ax2.axhline(0, color='k', ls='-', lw=0.5)
        ax2.set_xlabel("Time (days)")
        ax2.set_ylabel("Residuals (ppm)")

        # Calculate and show RMS
        rms = np.std(residuals) * 1e6
        ax2.text(0.98, 0.95, f'RMS = {rms:.1f} ppm', transform=ax2.transAxes,
                ha='right', va='top', fontsize=10)

        plt.tight_layout()
        return self

    def plot_detection_summary(
        self,
        target: str,
        period: float,
        depth_ppm: float,
        radius_earth: float,
        planet_class: str,
        teff: Optional[float] = None,
        in_hz: Optional[bool] = None,
        snr: Optional[float] = None,
        phase: Optional[np.ndarray] = None,
        flux: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> 'FigureGenerator':
        """
        Create a detection summary figure.

        Args:
            target: Target name
            period: Orbital period (days)
            depth_ppm: Transit depth (ppm)
            radius_earth: Planet radius (Earth radii)
            planet_class: Planet classification
            teff: Stellar effective temperature
            in_hz: Whether in habitable zone
            snr: Detection SNR
            phase, flux: Optional phase curve data
            figsize: Figure size

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        self._fig = plt.figure(figsize=figsize)

        # Layout: phase curve on left, parameters on right
        if phase is not None and flux is not None:
            ax1 = self._fig.add_subplot(121)
            ax2 = self._fig.add_subplot(122)

            # Phase curve
            ax1.scatter(phase, flux, s=1, color=COLORS['primary'], alpha=0.5)
            ax1.set_xlabel("Orbital Phase")
            ax1.set_ylabel("Normalized Flux")
            ax1.set_title("Phase-Folded Transit")
            ax1.set_xlim(-0.2, 0.2)
        else:
            ax2 = self._fig.add_subplot(111)

        # Parameters text
        ax2.axis('off')

        hz_str = "Yes" if in_hz else ("No" if in_hz is False else "N/A")
        hz_color = COLORS['candidate'] if in_hz else COLORS['transit']

        text = f"""
        Target: {target}

        Orbital Period: {period:.4f} days
        Transit Depth: {depth_ppm:.0f} ppm

        Planet Radius: {radius_earth:.2f} R_Earth
        Classification: {planet_class}

        Detection SNR: {snr:.1f if snr else 'N/A'}
        Stellar Teff: {teff:.0f} K if teff else 'N/A'

        In Habitable Zone: {hz_str}
        """

        ax2.text(0.1, 0.9, text.strip(), transform=ax2.transAxes,
                fontsize=14, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.set_title(f"Detection Summary: {target}", fontsize=16, fontweight='bold')

        self._fig.tight_layout()
        self._axes = ax2
        return self

    def plot_habitable_zone(
        self,
        stellar_teff: float,
        stellar_luminosity: float,
        hz_inner: float,
        hz_outer: float,
        planets: Optional[List[Dict[str, float]]] = None,
        title: str = "Habitable Zone",
        figsize: Tuple[int, int] = (10, 6)
    ) -> 'FigureGenerator':
        """
        Plot habitable zone diagram.

        Args:
            stellar_teff: Stellar effective temperature (K)
            stellar_luminosity: Stellar luminosity (L_sun)
            hz_inner: Inner HZ boundary (AU)
            hz_outer: Outer HZ boundary (AU)
            planets: List of {'name': str, 'a': float} for planets to plot
            title: Plot title
            figsize: Figure size

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        self._fig, self._axes = plt.subplots(figsize=figsize)
        ax = self._axes

        # Plot habitable zone
        ax.axvspan(hz_inner, hz_outer, alpha=0.3, color=COLORS['candidate'],
                  label='Habitable Zone')

        # Plot star
        ax.scatter([0], [0], s=200, color='yellow', edgecolor='orange',
                  marker='*', label=f'Star (Teff={stellar_teff:.0f}K)', zorder=5)

        # Plot planets
        if planets:
            for i, p in enumerate(planets):
                ax.scatter([p['a']], [0], s=100, color=COLORS['primary'],
                          edgecolor='black', zorder=5)
                ax.annotate(p.get('name', f'Planet {i+1}'),
                           (p['a'], 0.02), ha='center', fontsize=10)

        # Solar system reference (if scale allows)
        if hz_outer > 0.5:
            ax.axvline(1.0, color=COLORS['noise'], ls=':', alpha=0.5,
                      label='Earth orbit (1 AU)')

        ax.set_xlabel("Distance from Star (AU)")
        ax.set_xlim(0, max(hz_outer * 1.5, 2.0))
        ax.set_ylim(-0.2, 0.2)
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.set_yticks([])

        plt.tight_layout()
        return self

    def save(
        self,
        filename: Union[str, Path],
        dpi: int = 150,
        format: str = None
    ) -> str:
        """
        Save current figure.

        Args:
            filename: Output filename
            dpi: Resolution (dots per inch)
            format: Output format (auto-detected from extension if None)

        Returns:
            Path to saved file
        """
        import matplotlib.pyplot as plt

        if self._fig is None:
            raise ValueError("No figure to save. Create a plot first.")

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._fig.savefig(str(path), dpi=dpi, format=format)
        logger.info(f"Figure saved: {path}")

        return str(path)

    def show(self):
        """Display current figure."""
        import matplotlib.pyplot as plt

        if self._fig is None:
            raise ValueError("No figure to show. Create a plot first.")

        plt.show()

    def close(self):
        """Close current figure."""
        import matplotlib.pyplot as plt

        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None


# ============================================================================
# Skill Registration
# ============================================================================

SKILL_INFO = {
    'RES-003': {
        'id': 'RES-003',
        'name': 'Figure Generator',
        'command': 'research plot',
        'class': FigureGenerator,
        'description': 'Create publication-quality plots'
    }
}


# ============================================================================
# Convenience Functions
# ============================================================================

def create_lightcurve_plot(time, flux, **kwargs) -> str:
    """Create and save light curve plot."""
    gen = FigureGenerator()
    gen.plot_lightcurve(time, flux, **kwargs)
    filename = kwargs.get('filename', 'output/lightcurve.png')
    return gen.save(filename)


def create_periodogram_plot(periods, power, **kwargs) -> str:
    """Create and save periodogram plot."""
    gen = FigureGenerator()
    gen.plot_periodogram(periods, power, **kwargs)
    filename = kwargs.get('filename', 'output/periodogram.png')
    return gen.save(filename)


def create_phase_plot(phase, flux, **kwargs) -> str:
    """Create and save phase curve plot."""
    gen = FigureGenerator()
    gen.plot_phase_curve(phase, flux, **kwargs)
    filename = kwargs.get('filename', 'output/phase_curve.png')
    return gen.save(filename)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Figure Generator...")
    print("=" * 50)

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing

    # Generate test data
    np.random.seed(42)
    time = np.linspace(0, 27, 5000)
    flux = 1.0 + np.random.normal(0, 0.001, len(time))

    # Add transit
    period = 3.5
    phase = ((time - 1.0) % period) / period
    in_transit = phase < 0.03
    flux[in_transit] -= 0.01

    # Test light curve
    gen = FigureGenerator()
    gen.plot_lightcurve(time, flux, title="Test Light Curve")
    gen.save("output/test_lightcurve.png")
    print("Created: output/test_lightcurve.png")
    gen.close()

    # Test periodogram
    periods = np.linspace(1, 10, 1000)
    power = np.random.rand(len(periods))
    power[350] = 5.0  # Peak at P=3.5

    gen.plot_periodogram(periods, power, best_period=3.5, title="Test Periodogram")
    gen.save("output/test_periodogram.png")
    print("Created: output/test_periodogram.png")
    gen.close()

    # Test phase curve
    try:
        from skills.periodogram import phase_fold, bin_phase_curve
    except ImportError:
        from periodogram import phase_fold, bin_phase_curve
    phase_folded, flux_folded = phase_fold(time, flux, period, t0=1.0)
    bin_phase, bin_flux, bin_err = bin_phase_curve(phase_folded, flux_folded)

    gen.plot_phase_curve(
        phase_folded, flux_folded,
        bin_phase=bin_phase, bin_flux=bin_flux, bin_err=bin_err,
        period=period, depth_ppm=10000,
        title="Test Phase Curve"
    )
    gen.save("output/test_phase.png")
    print("Created: output/test_phase.png")
    gen.close()

    print("\nAll figures created successfully!")
