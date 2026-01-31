"""
Tests for LARUN Stellar Classification Skills
==============================================

Tests for stellar classification and temperature estimation.

Run with: pytest tests/test_stellar.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from skills.stellar import (
    StellarClassifier,
    StellarParameters,
    TemperatureEstimator,
    RadiusEstimator,
    SPECTRAL_TYPES,
    LUMINOSITY_CLASSES
)


class TestStellarClassifier:
    """Tests for stellar classification."""

    @pytest.fixture
    def classifier(self):
        return StellarClassifier()

    def test_sun_classification(self, classifier):
        """Test classification of the Sun (G2V)."""
        result = classifier.classify_from_teff(5778, logg=4.44)

        assert result.spectral_class == 'G'
        assert result.luminosity_class == 'V'
        assert 0 <= result.subtype <= 9
        assert 'G' in result.spectral_type
        assert 'V' in result.spectral_type

    def test_hot_star_classification(self, classifier):
        """Test O/B star classification."""
        # Vega-like (A0V)
        result = classifier.classify_from_teff(9600, logg=4.0)
        assert result.spectral_class == 'A'

        # O star
        result = classifier.classify_from_teff(35000, logg=4.0)
        assert result.spectral_class == 'O'

        # B star
        result = classifier.classify_from_teff(15000, logg=4.0)
        assert result.spectral_class == 'B'

    def test_cool_star_classification(self, classifier):
        """Test K/M star classification."""
        # K star
        result = classifier.classify_from_teff(4500, logg=4.5)
        assert result.spectral_class == 'K'

        # M dwarf
        result = classifier.classify_from_teff(3000, logg=5.0)
        assert result.spectral_class == 'M'

    def test_luminosity_class_giants(self, classifier):
        """Test luminosity class for giants."""
        # Giant (low logg)
        result = classifier.classify_from_teff(4500, logg=2.0)
        assert result.luminosity_class in ['II', 'III']

        # Supergiant
        result = classifier.classify_from_teff(4000, logg=0.5)
        assert result.luminosity_class in ['Ia', 'Ib']

    def test_luminosity_class_dwarfs(self, classifier):
        """Test luminosity class for main sequence stars."""
        # Main sequence
        result = classifier.classify_from_teff(5778, logg=4.44)
        assert result.luminosity_class == 'V'

        # Subgiant
        result = classifier.classify_from_teff(5500, logg=3.7)
        assert result.luminosity_class == 'IV'

    def test_mass_estimation(self, classifier):
        """Test mass estimation."""
        result = classifier.classify_from_teff(5778, logg=4.44)
        assert result.mass is not None
        assert 0.5 < result.mass < 2.0  # Sun-like mass

        # Hot massive star
        result = classifier.classify_from_teff(20000, logg=4.0)
        assert result.mass > 2.0

        # M dwarf
        result = classifier.classify_from_teff(3000, logg=5.0)
        assert result.mass < 0.5

    def test_radius_estimation(self, classifier):
        """Test radius estimation."""
        result = classifier.classify_from_teff(5778, logg=4.44)
        assert result.radius is not None
        assert 0.8 < result.radius < 1.5  # Sun-like radius

        # Giant
        result = classifier.classify_from_teff(4500, logg=2.0)
        assert result.radius > 5.0

    def test_luminosity_calculation(self, classifier):
        """Test luminosity calculation from Teff and radius."""
        result = classifier.classify_from_teff(5778, logg=4.44)
        assert result.luminosity is not None
        # Should be close to 1 L_sun for Sun-like star
        assert 0.5 < result.luminosity < 2.0

    def test_invalid_temperature(self, classifier):
        """Test handling of invalid temperature."""
        with pytest.raises(ValueError):
            classifier.classify_from_teff(0)

        with pytest.raises(ValueError):
            classifier.classify_from_teff(-1000)

    def test_to_dict(self, classifier):
        """Test dictionary conversion."""
        result = classifier.classify_from_teff(5778, logg=4.44)
        d = result.to_dict()

        assert 'teff_K' in d
        assert 'spectral_type' in d
        assert 'spectral_class' in d
        assert 'luminosity_class' in d
        assert d['teff_K'] == 5778


class TestTemperatureEstimator:
    """Tests for temperature estimation from colors."""

    @pytest.fixture
    def estimator(self):
        return TemperatureEstimator()

    def test_bp_rp_color(self, estimator):
        """Test temperature estimation from BP-RP color."""
        # Solar-like star (BP-RP ~ 0.82)
        teff, err = estimator.from_bp_rp(0.82)
        assert 5000 < teff < 6500
        assert err > 0

        # Hot star (small BP-RP)
        teff, err = estimator.from_bp_rp(0.2)
        assert teff > 7000

        # Cool star (large BP-RP)
        teff, err = estimator.from_bp_rp(2.5)
        assert teff < 4500

    def test_color_out_of_range(self, estimator):
        """Test warning for colors outside valid range."""
        # Should still return a value but may warn
        teff, err = estimator.from_bp_rp(5.0)  # Very red
        assert 2000 <= teff < 50000  # Within physical bounds (clamped to min)

    def test_metallicity_correction(self, estimator):
        """Test metallicity correction."""
        teff_solar, _ = estimator.from_bp_rp(0.82, metallicity=0.0)
        teff_metal_poor, _ = estimator.from_bp_rp(0.82, metallicity=-1.0)
        teff_metal_rich, _ = estimator.from_bp_rp(0.82, metallicity=0.5)

        # Metal-poor stars appear hotter at same color
        assert teff_metal_poor < teff_solar
        # Metal-rich stars appear cooler
        assert teff_metal_rich > teff_solar


class TestRadiusEstimator:
    """Tests for radius estimation."""

    @pytest.fixture
    def estimator(self):
        return RadiusEstimator()

    def test_from_luminosity_teff(self, estimator):
        """Test radius from luminosity and Teff."""
        # Sun: L=1, Teff=5778K, R=1 R_sun
        radius, err = estimator.from_luminosity_teff(1.0, 5778)
        assert 0.9 < radius < 1.1
        assert err > 0

        # Hot luminous star
        radius, err = estimator.from_luminosity_teff(100.0, 10000)
        assert radius > 2.0

    def test_from_gaia_params(self, estimator):
        """Test radius from Gaia parameters."""
        # Sun-like
        radius, err = estimator.from_gaia_params(5778, 4.44, mass=1.0)
        assert 0.5 < radius < 2.0
        assert err > 0

        # Giant (low logg)
        radius, err = estimator.from_gaia_params(4500, 2.0, mass=1.5)
        assert radius > 5.0


class TestSpectralTypes:
    """Tests for spectral type data."""

    def test_spectral_type_ordering(self):
        """Test that spectral types are ordered by temperature."""
        types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

        for i in range(len(types) - 1):
            t1 = SPECTRAL_TYPES[types[i]]
            t2 = SPECTRAL_TYPES[types[i + 1]]
            # Hotter type should have higher Teff
            assert t1['teff_min'] > t2['teff_min']

    def test_spectral_type_properties(self):
        """Test spectral type property ranges."""
        for spec_type, props in SPECTRAL_TYPES.items():
            assert props['teff_min'] > 0
            assert props['teff_max'] > props['teff_min']
            assert 'mass_solar' in props
            assert 'radius_solar' in props


class TestLuminosityClasses:
    """Tests for luminosity class data."""

    def test_all_classes_defined(self):
        """Test that all luminosity classes are defined."""
        expected = ['Ia', 'Ib', 'II', 'III', 'IV', 'V', 'VI', 'VII']
        for lc in expected:
            assert lc in LUMINOSITY_CLASSES


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
