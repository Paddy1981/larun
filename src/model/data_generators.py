"""
Synthetic Data Generators for Astronomical TinyML Models
=========================================================
Generates realistic training data for each specialized model.
Based on actual astronomical phenomena and physics.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    n_samples: int = 1000
    noise_level: float = 0.01
    seed: int = 42


class ExoplanetDataGenerator:
    """
    Generates synthetic light curves for exoplanet transit detection.
    Classes: noise, stellar_signal, planetary_transit, eclipsing_binary,
             instrument_artifact, unknown_anomaly
    """

    def __init__(self, n_points: int = 1024):
        self.n_points = n_points
        self.class_labels = [
            "noise", "stellar_signal", "planetary_transit",
            "eclipsing_binary", "instrument_artifact", "unknown_anomaly"
        ]

    def generate_transit(self, period: float, depth: float, duration: float,
                        t0: float = 0.5) -> np.ndarray:
        """Generate a planetary transit signal."""
        time = np.linspace(0, 1, self.n_points)
        flux = np.ones(self.n_points)

        # Calculate transit phases
        phase = (time - t0) % (period / self.n_points)
        in_transit = np.abs(phase - 0.5) < (duration / 2)

        # Apply transit with limb darkening approximation
        flux[in_transit] = 1.0 - depth * (1.0 - 0.3 * (np.abs(phase[in_transit] - 0.5) / (duration / 2)) ** 2)

        return flux

    def generate_eclipsing_binary(self, period: float, depth1: float, depth2: float) -> np.ndarray:
        """Generate eclipsing binary signal with primary and secondary eclipses."""
        time = np.linspace(0, 1, self.n_points)
        flux = np.ones(self.n_points)

        phase = time % period
        # Primary eclipse (deeper)
        in_primary = np.abs(phase - 0.25 * period) < 0.05 * period
        flux[in_primary] = 1.0 - depth1

        # Secondary eclipse (shallower)
        in_secondary = np.abs(phase - 0.75 * period) < 0.04 * period
        flux[in_secondary] = 1.0 - depth2

        return flux

    def generate_stellar_signal(self, period: float, amplitude: float) -> np.ndarray:
        """Generate stellar variability (rotation, spots)."""
        time = np.linspace(0, 1, self.n_points)
        # Quasi-periodic stellar signal
        signal = amplitude * np.sin(2 * np.pi * time / period)
        signal += 0.3 * amplitude * np.sin(4 * np.pi * time / period + 0.5)
        return 1.0 + signal

    def generate_instrument_artifact(self) -> np.ndarray:
        """Generate instrumental artifacts (discontinuities, trends)."""
        flux = np.ones(self.n_points)

        # Random discontinuities
        n_jumps = np.random.randint(1, 4)
        for _ in range(n_jumps):
            jump_idx = np.random.randint(100, self.n_points - 100)
            jump_size = np.random.uniform(-0.02, 0.02)
            flux[jump_idx:] += jump_size

        # Add trend
        trend = np.linspace(0, np.random.uniform(-0.01, 0.01), self.n_points)
        flux += trend

        return flux

    def generate_unknown_anomaly(self) -> np.ndarray:
        """Generate unusual patterns (outbursts, dips, etc.)."""
        flux = np.ones(self.n_points)

        anomaly_type = np.random.choice(['dip', 'burst', 'oscillation'])

        if anomaly_type == 'dip':
            # Single deep dip
            center = np.random.randint(200, self.n_points - 200)
            width = np.random.randint(20, 100)
            depth = np.random.uniform(0.05, 0.3)
            x = np.arange(self.n_points)
            flux -= depth * np.exp(-((x - center) ** 2) / (2 * width ** 2))

        elif anomaly_type == 'burst':
            # Brightness increase
            center = np.random.randint(200, self.n_points - 200)
            width = np.random.randint(10, 50)
            height = np.random.uniform(0.02, 0.1)
            x = np.arange(self.n_points)
            flux += height * np.exp(-((x - center) ** 2) / (2 * width ** 2))

        else:  # oscillation
            freq = np.random.uniform(5, 20)
            amp = np.random.uniform(0.01, 0.05)
            time = np.linspace(0, 1, self.n_points)
            flux += amp * np.sin(2 * np.pi * freq * time)

        return flux

    def generate_dataset(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full training dataset."""
        np.random.seed(config.seed)

        X = []
        y = []
        samples_per_class = config.n_samples // len(self.class_labels)

        for class_idx, class_name in enumerate(self.class_labels):
            for _ in range(samples_per_class):
                if class_name == "noise":
                    flux = np.ones(self.n_points)
                elif class_name == "stellar_signal":
                    period = np.random.uniform(0.1, 0.5)
                    amplitude = np.random.uniform(0.005, 0.03)
                    flux = self.generate_stellar_signal(period, amplitude)
                elif class_name == "planetary_transit":
                    period = np.random.uniform(0.2, 0.8)
                    depth = np.random.uniform(0.001, 0.02)
                    duration = np.random.uniform(0.02, 0.1)
                    flux = self.generate_transit(period, depth, duration)
                elif class_name == "eclipsing_binary":
                    period = np.random.uniform(0.15, 0.6)
                    depth1 = np.random.uniform(0.05, 0.3)
                    depth2 = np.random.uniform(0.01, depth1 * 0.5)
                    flux = self.generate_eclipsing_binary(period, depth1, depth2)
                elif class_name == "instrument_artifact":
                    flux = self.generate_instrument_artifact()
                else:  # unknown_anomaly
                    flux = self.generate_unknown_anomaly()

                # Add noise
                flux += np.random.normal(0, config.noise_level, self.n_points)

                X.append(flux)
                y.append(class_idx)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Shuffle
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]


class VariableStarDataGenerator:
    """
    Generates synthetic light curves for variable star classification.
    Classes: cepheid, rr_lyrae, delta_scuti, eclipsing_binary, rotational, irregular, constant
    """

    def __init__(self, n_points: int = 512):
        self.n_points = n_points
        self.class_labels = [
            "cepheid", "rr_lyrae", "delta_scuti", "eclipsing_binary",
            "rotational", "irregular", "constant"
        ]

    def generate_cepheid(self, period: float, amplitude: float) -> np.ndarray:
        """Cepheid variable with characteristic sawtooth shape."""
        time = np.linspace(0, 3 * period, self.n_points)
        phase = (time % period) / period
        # Cepheids have fast rise, slow decline
        flux = amplitude * (1 - phase + 0.2 * np.sin(2 * np.pi * phase))
        return 1.0 + flux - np.mean(flux)

    def generate_rr_lyrae(self, period: float, amplitude: float) -> np.ndarray:
        """RR Lyrae with sharper, more sinusoidal variation."""
        time = np.linspace(0, 3 * period, self.n_points)
        phase = (time % period) / period
        # RR Lyrae have rapid rise
        flux = amplitude * (np.sin(2 * np.pi * phase) + 0.3 * np.sin(4 * np.pi * phase))
        return 1.0 + flux

    def generate_delta_scuti(self, period: float, amplitude: float) -> np.ndarray:
        """Delta Scuti with multi-mode pulsations."""
        time = np.linspace(0, 10 * period, self.n_points)
        # Multiple pulsation modes
        flux = amplitude * np.sin(2 * np.pi * time / period)
        flux += 0.5 * amplitude * np.sin(2 * np.pi * time / (period * 0.77))
        flux += 0.3 * amplitude * np.sin(2 * np.pi * time / (period * 1.23))
        return 1.0 + flux

    def generate_rotational(self, period: float, amplitude: float) -> np.ndarray:
        """Rotational modulation from starspots."""
        time = np.linspace(0, 2 * period, self.n_points)
        phase = (time % period) / period
        # Spot-induced variation
        flux = -amplitude * (0.7 * np.cos(2 * np.pi * phase) + 0.3 * np.cos(4 * np.pi * phase))
        return 1.0 + flux

    def generate_irregular(self) -> np.ndarray:
        """Irregular variable with random variations."""
        time = np.linspace(0, 1, self.n_points)
        # Superposition of random frequencies
        flux = np.zeros(self.n_points)
        for _ in range(np.random.randint(3, 8)):
            freq = np.random.uniform(2, 20)
            amp = np.random.uniform(0.005, 0.02)
            phase = np.random.uniform(0, 2 * np.pi)
            flux += amp * np.sin(2 * np.pi * freq * time + phase)
        return 1.0 + flux

    def generate_eclipsing_binary(self, period: float, depth1: float, depth2: float) -> np.ndarray:
        """Eclipsing binary light curve."""
        time = np.linspace(0, 2 * period, self.n_points)
        flux = np.ones(self.n_points)
        phase = (time % period) / period

        # Primary eclipse
        in_primary = np.abs(phase - 0.0) < 0.08
        flux[in_primary] = 1.0 - depth1 * (1 - (np.abs(phase[in_primary]) / 0.08) ** 2)

        # Secondary eclipse
        in_secondary = np.abs(phase - 0.5) < 0.06
        flux[in_secondary] = 1.0 - depth2 * (1 - (np.abs(phase[in_secondary] - 0.5) / 0.06) ** 2)

        return flux

    def generate_dataset(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full training dataset."""
        np.random.seed(config.seed)

        X = []
        y = []
        samples_per_class = config.n_samples // len(self.class_labels)

        for class_idx, class_name in enumerate(self.class_labels):
            for _ in range(samples_per_class):
                if class_name == "cepheid":
                    period = np.random.uniform(0.1, 0.3)
                    amplitude = np.random.uniform(0.1, 0.5)
                    flux = self.generate_cepheid(period, amplitude)
                elif class_name == "rr_lyrae":
                    period = np.random.uniform(0.05, 0.15)
                    amplitude = np.random.uniform(0.2, 0.8)
                    flux = self.generate_rr_lyrae(period, amplitude)
                elif class_name == "delta_scuti":
                    period = np.random.uniform(0.02, 0.08)
                    amplitude = np.random.uniform(0.01, 0.05)
                    flux = self.generate_delta_scuti(period, amplitude)
                elif class_name == "eclipsing_binary":
                    period = np.random.uniform(0.1, 0.4)
                    depth1 = np.random.uniform(0.1, 0.5)
                    depth2 = np.random.uniform(0.02, depth1 * 0.4)
                    flux = self.generate_eclipsing_binary(period, depth1, depth2)
                elif class_name == "rotational":
                    period = np.random.uniform(0.2, 0.6)
                    amplitude = np.random.uniform(0.01, 0.05)
                    flux = self.generate_rotational(period, amplitude)
                elif class_name == "irregular":
                    flux = self.generate_irregular()
                else:  # constant
                    flux = np.ones(self.n_points)

                flux += np.random.normal(0, config.noise_level, self.n_points)
                X.append(flux)
                y.append(class_idx)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]


class FlareDataGenerator:
    """
    Generates synthetic light curves for stellar flare detection.
    Classes: no_flare, weak_flare, moderate_flare, strong_flare, superflare
    """

    def __init__(self, n_points: int = 256):
        self.n_points = n_points
        self.class_labels = ["no_flare", "weak_flare", "moderate_flare", "strong_flare", "superflare"]

    def generate_flare(self, peak_amplitude: float, rise_time: float, decay_time: float,
                      position: float = 0.5) -> np.ndarray:
        """Generate a flare with fast rise and exponential decay."""
        time = np.linspace(0, 1, self.n_points)
        flux = np.ones(self.n_points)

        # Find flare region
        flare_start = int(position * self.n_points)
        rise_samples = int(rise_time * self.n_points)
        decay_samples = int(decay_time * self.n_points)

        # Rise phase (linear)
        rise_end = min(flare_start + rise_samples, self.n_points)
        if flare_start < rise_end:
            rise = np.linspace(0, peak_amplitude, rise_end - flare_start)
            flux[flare_start:rise_end] += rise

        # Decay phase (exponential)
        decay_start = rise_end
        decay_end = min(decay_start + decay_samples, self.n_points)
        if decay_start < decay_end:
            decay_time_arr = np.arange(decay_end - decay_start)
            decay = peak_amplitude * np.exp(-decay_time_arr / (decay_samples / 3))
            flux[decay_start:decay_end] += decay

        return flux

    def generate_dataset(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full training dataset."""
        np.random.seed(config.seed)

        X = []
        y = []
        samples_per_class = config.n_samples // len(self.class_labels)

        # Flare amplitude ranges by class
        amplitude_ranges = {
            "no_flare": (0, 0),
            "weak_flare": (0.01, 0.05),
            "moderate_flare": (0.05, 0.15),
            "strong_flare": (0.15, 0.4),
            "superflare": (0.4, 1.5)
        }

        for class_idx, class_name in enumerate(self.class_labels):
            for _ in range(samples_per_class):
                if class_name == "no_flare":
                    flux = np.ones(self.n_points)
                else:
                    amp_min, amp_max = amplitude_ranges[class_name]
                    amplitude = np.random.uniform(amp_min, amp_max)
                    rise_time = np.random.uniform(0.01, 0.05)
                    decay_time = np.random.uniform(0.05, 0.2)
                    position = np.random.uniform(0.2, 0.7)
                    flux = self.generate_flare(amplitude, rise_time, decay_time, position)

                flux += np.random.normal(0, config.noise_level, self.n_points)
                X.append(flux)
                y.append(class_idx)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]


class AsteroseismologyDataGenerator:
    """
    Generates synthetic power spectra for asteroseismology analysis.
    Classes: no_oscillation, solar_like, red_giant, delta_scuti, gamma_dor, hybrid
    """

    def __init__(self, n_points: int = 512):
        self.n_points = n_points
        self.class_labels = ["no_oscillation", "solar_like", "red_giant", "delta_scuti", "gamma_dor", "hybrid"]

    def generate_oscillation_pattern(self, nu_max: float, delta_nu: float,
                                    n_modes: int, mode_height: float) -> np.ndarray:
        """Generate oscillation pattern with p-modes."""
        freq = np.linspace(0, 1, self.n_points)  # Normalized frequency
        power = np.zeros(self.n_points)

        # Gaussian envelope centered at nu_max
        envelope = np.exp(-((freq - nu_max) ** 2) / (2 * (0.1) ** 2))

        # Add individual modes
        for n in range(-n_modes // 2, n_modes // 2 + 1):
            mode_freq = nu_max + n * delta_nu
            if 0 < mode_freq < 1:
                mode_width = 0.005 + 0.002 * np.random.randn()
                mode_power = mode_height * envelope[int(mode_freq * self.n_points)] * np.random.uniform(0.5, 1.5)
                power += mode_power * np.exp(-((freq - mode_freq) ** 2) / (2 * mode_width ** 2))

        return power

    def generate_dataset(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full training dataset."""
        np.random.seed(config.seed)

        X = []
        y = []
        samples_per_class = config.n_samples // len(self.class_labels)

        for class_idx, class_name in enumerate(self.class_labels):
            for _ in range(samples_per_class):
                if class_name == "no_oscillation":
                    power = np.random.exponential(0.1, self.n_points)
                elif class_name == "solar_like":
                    nu_max = np.random.uniform(0.4, 0.6)
                    delta_nu = np.random.uniform(0.03, 0.05)
                    power = self.generate_oscillation_pattern(nu_max, delta_nu, 10, 1.0)
                elif class_name == "red_giant":
                    nu_max = np.random.uniform(0.1, 0.25)
                    delta_nu = np.random.uniform(0.01, 0.02)
                    power = self.generate_oscillation_pattern(nu_max, delta_nu, 15, 2.0)
                elif class_name == "delta_scuti":
                    nu_max = np.random.uniform(0.6, 0.8)
                    delta_nu = np.random.uniform(0.02, 0.04)
                    power = self.generate_oscillation_pattern(nu_max, delta_nu, 8, 1.5)
                elif class_name == "gamma_dor":
                    nu_max = np.random.uniform(0.15, 0.3)
                    delta_nu = np.random.uniform(0.02, 0.03)
                    power = self.generate_oscillation_pattern(nu_max, delta_nu, 6, 0.8)
                else:  # hybrid
                    # Combined delta_scuti + gamma_dor
                    power1 = self.generate_oscillation_pattern(0.7, 0.03, 6, 1.0)
                    power2 = self.generate_oscillation_pattern(0.2, 0.02, 5, 0.7)
                    power = power1 + power2

                # Add noise
                power += np.abs(np.random.normal(0, config.noise_level * 0.5, self.n_points))
                power = np.maximum(power, 0)  # Ensure non-negative

                X.append(power)
                y.append(class_idx)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]


class SupernovaDataGenerator:
    """
    Generates synthetic light curves for supernova/transient detection.
    Classes: no_transient, sn_ia, sn_ii, sn_ibc, kilonova, tde, other_transient
    """

    def __init__(self, n_points: int = 128):
        self.n_points = n_points
        self.class_labels = ["no_transient", "sn_ia", "sn_ii", "sn_ibc", "kilonova", "tde", "other_transient"]

    def generate_sn_ia(self, peak_mag: float, t_rise: float, t_decline: float) -> np.ndarray:
        """Type Ia supernova with characteristic rise and decline."""
        time = np.linspace(0, 1, self.n_points)
        flux = np.zeros(self.n_points)

        peak_idx = int(0.3 * self.n_points)

        # Rise
        rise_idx = np.arange(peak_idx)
        flux[rise_idx] = peak_mag * (1 - np.exp(-(rise_idx / peak_idx) / t_rise))

        # Decline (two-component)
        decline_idx = np.arange(peak_idx, self.n_points)
        t = (decline_idx - peak_idx) / (self.n_points - peak_idx)
        flux[decline_idx] = peak_mag * (0.6 * np.exp(-t / t_decline) + 0.4 * np.exp(-t / (3 * t_decline)))

        return 1.0 + flux

    def generate_sn_ii(self, peak_mag: float, plateau_length: float) -> np.ndarray:
        """Type II supernova with plateau phase."""
        time = np.linspace(0, 1, self.n_points)
        flux = np.zeros(self.n_points)

        peak_idx = int(0.2 * self.n_points)
        plateau_end = int((0.2 + plateau_length) * self.n_points)

        # Rise
        flux[:peak_idx] = peak_mag * np.linspace(0, 1, peak_idx)

        # Plateau
        flux[peak_idx:plateau_end] = peak_mag * (1 - 0.1 * (np.arange(plateau_end - peak_idx) / (plateau_end - peak_idx)))

        # Drop and tail
        flux[plateau_end:] = flux[plateau_end - 1] * np.exp(-(np.arange(self.n_points - plateau_end) / 20))

        return 1.0 + flux

    def generate_kilonova(self, peak_mag: float) -> np.ndarray:
        """Kilonova with rapid evolution."""
        time = np.linspace(0, 1, self.n_points)
        flux = np.zeros(self.n_points)

        peak_idx = int(0.15 * self.n_points)

        # Very fast rise
        flux[:peak_idx] = peak_mag * (np.arange(peak_idx) / peak_idx) ** 0.5

        # Rapid decline
        decline = np.arange(self.n_points - peak_idx)
        flux[peak_idx:] = peak_mag * np.exp(-decline / 15)

        return 1.0 + flux

    def generate_tde(self, peak_mag: float) -> np.ndarray:
        """Tidal disruption event with slow evolution."""
        time = np.linspace(0, 1, self.n_points)
        flux = np.zeros(self.n_points)

        peak_idx = int(0.25 * self.n_points)

        # Gradual rise
        flux[:peak_idx] = peak_mag * (np.arange(peak_idx) / peak_idx) ** 2

        # Power-law decline (t^-5/3)
        t = 1 + np.arange(self.n_points - peak_idx)
        flux[peak_idx:] = peak_mag * t ** (-5 / 3)

        return 1.0 + flux

    def generate_dataset(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full training dataset."""
        np.random.seed(config.seed)

        X = []
        y = []
        samples_per_class = config.n_samples // len(self.class_labels)

        for class_idx, class_name in enumerate(self.class_labels):
            for _ in range(samples_per_class):
                if class_name == "no_transient":
                    flux = np.ones(self.n_points)
                elif class_name == "sn_ia":
                    peak = np.random.uniform(0.3, 1.0)
                    flux = self.generate_sn_ia(peak, 0.15, 0.3)
                elif class_name == "sn_ii":
                    peak = np.random.uniform(0.2, 0.8)
                    plateau = np.random.uniform(0.3, 0.5)
                    flux = self.generate_sn_ii(peak, plateau)
                elif class_name == "sn_ibc":
                    # Similar to Ia but faster
                    peak = np.random.uniform(0.3, 0.9)
                    flux = self.generate_sn_ia(peak, 0.1, 0.2)
                elif class_name == "kilonova":
                    peak = np.random.uniform(0.2, 0.6)
                    flux = self.generate_kilonova(peak)
                elif class_name == "tde":
                    peak = np.random.uniform(0.3, 0.8)
                    flux = self.generate_tde(peak)
                else:  # other_transient
                    # Random transient shape
                    peak = np.random.uniform(0.1, 0.5)
                    peak_idx = np.random.randint(20, 80)
                    flux = np.ones(self.n_points)
                    width = np.random.randint(10, 40)
                    x = np.arange(self.n_points)
                    flux += peak * np.exp(-((x - peak_idx) ** 2) / (2 * width ** 2))

                flux += np.random.normal(0, config.noise_level, self.n_points)
                X.append(flux)
                y.append(class_idx)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]


class GalaxyDataGenerator:
    """
    Generates synthetic galaxy images for morphology classification.
    Classes: elliptical, spiral, barred_spiral, irregular, merger, edge_on, unknown
    """

    def __init__(self, image_size: int = 64):
        self.image_size = image_size
        self.class_labels = ["elliptical", "spiral", "barred_spiral", "irregular", "merger", "edge_on", "unknown"]

    def generate_elliptical(self, ellipticity: float, size: float) -> np.ndarray:
        """Generate elliptical galaxy."""
        y, x = np.ogrid[:self.image_size, :self.image_size]
        center = self.image_size // 2

        # Elliptical profile
        angle = np.random.uniform(0, np.pi)
        x_rot = (x - center) * np.cos(angle) + (y - center) * np.sin(angle)
        y_rot = -(x - center) * np.sin(angle) + (y - center) * np.cos(angle)

        r = np.sqrt(x_rot ** 2 + (y_rot / (1 - ellipticity)) ** 2)
        image = np.exp(-r / (size * self.image_size / 10))

        return image

    def generate_spiral(self, n_arms: int, arm_tightness: float) -> np.ndarray:
        """Generate spiral galaxy."""
        y, x = np.ogrid[:self.image_size, :self.image_size]
        center = self.image_size // 2

        r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        theta = np.arctan2(y - center, x - center)

        # Bulge
        bulge = 0.5 * np.exp(-r / (self.image_size / 15))

        # Spiral arms
        arm_pattern = np.zeros((self.image_size, self.image_size))
        for i in range(n_arms):
            arm_theta = theta - arm_tightness * np.log(r + 1) + i * 2 * np.pi / n_arms
            arm_pattern += 0.5 * np.exp(-((np.sin(arm_theta)) ** 2) / 0.3) * np.exp(-r / (self.image_size / 4))

        # Disk
        disk = 0.3 * np.exp(-r / (self.image_size / 6))

        image = bulge + arm_pattern + disk
        return image / image.max()

    def generate_barred_spiral(self) -> np.ndarray:
        """Generate barred spiral galaxy."""
        y, x = np.ogrid[:self.image_size, :self.image_size]
        center = self.image_size // 2

        r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        theta = np.arctan2(y - center, x - center)

        # Bar
        bar_angle = np.random.uniform(0, np.pi)
        bar_x = np.abs((x - center) * np.cos(bar_angle) + (y - center) * np.sin(bar_angle))
        bar_y = np.abs(-(x - center) * np.sin(bar_angle) + (y - center) * np.cos(bar_angle))
        bar = 0.6 * np.exp(-bar_x / (self.image_size / 8)) * np.exp(-(bar_y ** 2) / (self.image_size / 3))

        # Spiral arms from bar ends
        arm_pattern = np.zeros((self.image_size, self.image_size))
        for sign in [-1, 1]:
            arm_theta = theta - sign * 0.5 * np.log(r + 1) + bar_angle
            arm_pattern += 0.4 * np.exp(-((np.sin(arm_theta)) ** 2) / 0.4) * np.exp(-r / (self.image_size / 5))

        image = bar + arm_pattern
        return image / image.max()

    def generate_irregular(self) -> np.ndarray:
        """Generate irregular galaxy."""
        image = np.zeros((self.image_size, self.image_size))

        # Multiple blobs
        n_blobs = np.random.randint(3, 8)
        for _ in range(n_blobs):
            cx = np.random.randint(10, self.image_size - 10)
            cy = np.random.randint(10, self.image_size - 10)
            size = np.random.uniform(3, 10)
            brightness = np.random.uniform(0.3, 1.0)

            y, x = np.ogrid[:self.image_size, :self.image_size]
            blob = brightness * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * size ** 2))
            image += blob

        return image / image.max()

    def generate_merger(self) -> np.ndarray:
        """Generate merging galaxies."""
        # Two overlapping galaxies
        image1 = self.generate_elliptical(np.random.uniform(0.2, 0.5), np.random.uniform(0.3, 0.5))
        image2 = self.generate_elliptical(np.random.uniform(0.2, 0.5), np.random.uniform(0.3, 0.5))

        # Shift second galaxy
        shift = np.random.randint(-15, 15, 2)
        image2 = np.roll(np.roll(image2, shift[0], axis=0), shift[1], axis=1)

        # Add tidal features
        y, x = np.ogrid[:self.image_size, :self.image_size]
        center = self.image_size // 2
        tidal = 0.1 * np.sin(0.1 * (x + y)) * np.exp(-((x - center) ** 2 + (y - center) ** 2) / (self.image_size ** 2))

        image = image1 + 0.7 * image2 + np.abs(tidal)
        return image / image.max()

    def generate_edge_on(self) -> np.ndarray:
        """Generate edge-on disk galaxy."""
        y, x = np.ogrid[:self.image_size, :self.image_size]
        center = self.image_size // 2

        # Thin disk
        angle = np.random.uniform(-0.3, 0.3)
        x_rot = (x - center) * np.cos(angle) + (y - center) * np.sin(angle)
        y_rot = -(x - center) * np.sin(angle) + (y - center) * np.cos(angle)

        disk = np.exp(-np.abs(y_rot) / 2) * np.exp(-np.abs(x_rot) / (self.image_size / 4))

        # Central bulge
        bulge = 0.5 * np.exp(-(x_rot ** 2 + y_rot ** 2) / (self.image_size / 5) ** 2)

        image = disk + bulge
        return image / image.max()

    def generate_dataset(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full training dataset."""
        np.random.seed(config.seed)

        X = []
        y = []
        samples_per_class = config.n_samples // len(self.class_labels)

        for class_idx, class_name in enumerate(self.class_labels):
            for _ in range(samples_per_class):
                if class_name == "elliptical":
                    image = self.generate_elliptical(np.random.uniform(0.1, 0.6), np.random.uniform(0.3, 0.6))
                elif class_name == "spiral":
                    image = self.generate_spiral(np.random.randint(2, 5), np.random.uniform(0.3, 0.8))
                elif class_name == "barred_spiral":
                    image = self.generate_barred_spiral()
                elif class_name == "irregular":
                    image = self.generate_irregular()
                elif class_name == "merger":
                    image = self.generate_merger()
                elif class_name == "edge_on":
                    image = self.generate_edge_on()
                else:  # unknown
                    # Random noise pattern
                    image = np.random.exponential(0.1, (self.image_size, self.image_size))

                # Add noise
                image += np.abs(np.random.normal(0, config.noise_level, (self.image_size, self.image_size)))
                image = image / image.max()

                X.append(image)
                y.append(class_idx)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]


class SpectralTypeDataGenerator:
    """
    Generates synthetic photometric features for spectral type classification.
    Classes: O, B, A, F, G, K, M, L
    Features: 8 photometric colors/indices
    """

    def __init__(self, n_features: int = 8):
        self.n_features = n_features
        self.class_labels = ["O", "B", "A", "F", "G", "K", "M", "L"]

        # Typical color indices for each spectral type (simplified)
        self.type_features = {
            "O": [-0.3, -0.15, -0.1, -0.05, 0.0, 0.1, 0.2, 0.3],
            "B": [-0.2, -0.1, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
            "A": [-0.05, 0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
            "F": [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7],
            "G": [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9],
            "K": [0.5, 0.65, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6],
            "M": [0.8, 1.0, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6],
            "L": [1.2, 1.5, 1.8, 2.1, 2.4, 2.8, 3.2, 3.6]
        }

    def generate_dataset(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full training dataset."""
        np.random.seed(config.seed)

        X = []
        y = []
        samples_per_class = config.n_samples // len(self.class_labels)

        for class_idx, class_name in enumerate(self.class_labels):
            base_features = np.array(self.type_features[class_name])

            for _ in range(samples_per_class):
                # Add realistic scatter
                scatter = np.random.normal(0, 0.1, self.n_features)
                features = base_features + scatter

                # Add measurement errors
                features += np.random.normal(0, config.noise_level, self.n_features)

                X.append(features)
                y.append(class_idx)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]


class MicrolensingDataGenerator:
    """
    Generates synthetic light curves for microlensing event detection.
    Classes: no_event, single_lens, binary_lens, planetary, parallax, unclear
    """

    def __init__(self, n_points: int = 512):
        self.n_points = n_points
        self.class_labels = ["no_event", "single_lens", "binary_lens", "planetary", "parallax", "unclear"]

    def paczynski_curve(self, t: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
        """Standard Paczynski microlensing curve."""
        u = np.sqrt(u0 ** 2 + ((t - t0) / tE) ** 2)
        A = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
        return A

    def generate_single_lens(self, t0: float, tE: float, u0: float) -> np.ndarray:
        """Generate single-lens microlensing event."""
        time = np.linspace(0, 1, self.n_points)
        magnification = self.paczynski_curve(time, t0, tE, u0)
        return magnification

    def generate_binary_lens(self, t0: float, tE: float, u0: float) -> np.ndarray:
        """Generate binary-lens event with caustic crossing."""
        time = np.linspace(0, 1, self.n_points)
        base = self.paczynski_curve(time, t0, tE, u0)

        # Add caustic crossing features
        caustic_t = t0 + np.random.uniform(-0.05, 0.05)
        caustic_width = 0.01
        caustic_height = np.random.uniform(1.5, 3.0)

        caustic = caustic_height * np.exp(-((time - caustic_t) ** 2) / (2 * caustic_width ** 2))

        return base + caustic

    def generate_planetary(self, t0: float, tE: float, u0: float) -> np.ndarray:
        """Generate planetary microlensing event."""
        time = np.linspace(0, 1, self.n_points)
        base = self.paczynski_curve(time, t0, tE, u0)

        # Small planetary deviation
        planet_t = t0 + np.random.uniform(-0.1, 0.1)
        planet_width = 0.02
        planet_effect = np.random.uniform(0.1, 0.4)

        deviation = planet_effect * np.exp(-((time - planet_t) ** 2) / (2 * planet_width ** 2))

        return base + deviation

    def generate_parallax(self, t0: float, tE: float, u0: float) -> np.ndarray:
        """Generate event with parallax signature."""
        time = np.linspace(0, 1, self.n_points)
        base = self.paczynski_curve(time, t0, tE, u0)

        # Add asymmetry from parallax
        parallax_amp = np.random.uniform(0.02, 0.1)
        asymmetry = parallax_amp * (time - t0) / tE

        return base * (1 + asymmetry)

    def generate_dataset(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full training dataset."""
        np.random.seed(config.seed)

        X = []
        y = []
        samples_per_class = config.n_samples // len(self.class_labels)

        for class_idx, class_name in enumerate(self.class_labels):
            for _ in range(samples_per_class):
                t0 = np.random.uniform(0.3, 0.7)
                tE = np.random.uniform(0.05, 0.2)
                u0 = np.random.uniform(0.01, 0.5)

                if class_name == "no_event":
                    flux = np.ones(self.n_points)
                elif class_name == "single_lens":
                    flux = self.generate_single_lens(t0, tE, u0)
                elif class_name == "binary_lens":
                    flux = self.generate_binary_lens(t0, tE, u0)
                elif class_name == "planetary":
                    flux = self.generate_planetary(t0, tE, u0)
                elif class_name == "parallax":
                    flux = self.generate_parallax(t0, tE, u0)
                else:  # unclear
                    # Ambiguous event
                    flux = self.generate_single_lens(t0, tE, u0)
                    flux += 0.1 * np.sin(2 * np.pi * 5 * np.linspace(0, 1, self.n_points))

                flux += np.random.normal(0, config.noise_level, self.n_points)
                X.append(flux)
                y.append(class_idx)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]


# Generator registry
DATA_GENERATORS = {
    "EXOPLANET-001": ExoplanetDataGenerator,
    "VSTAR-001": VariableStarDataGenerator,
    "FLARE-001": FlareDataGenerator,
    "ASTERO-001": AsteroseismologyDataGenerator,
    "SUPERNOVA-001": SupernovaDataGenerator,
    "GALAXY-001": GalaxyDataGenerator,
    "SPECTYPE-001": SpectralTypeDataGenerator,
    "MICROLENS-001": MicrolensingDataGenerator,
}


def get_generator(node_id: str):
    """Factory function to get a data generator by node ID."""
    if node_id not in DATA_GENERATORS:
        raise ValueError(f"Unknown generator: {node_id}. Available: {list(DATA_GENERATORS.keys())}")
    return DATA_GENERATORS[node_id]()


if __name__ == "__main__":
    print("Testing all data generators...\n")

    config = DatasetConfig(n_samples=100, noise_level=0.01, seed=42)

    for node_id, GeneratorClass in DATA_GENERATORS.items():
        generator = GeneratorClass()
        X, y = generator.generate_dataset(config)

        print(f"{node_id}:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {np.unique(y)}")
        print(f"  Class distribution: {np.bincount(y)}")
        print()
