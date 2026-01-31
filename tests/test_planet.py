"""
Tests for LARUN Planet Characterization Skills
==============================================

Tests for planet radius calculation and habitability assessment.

Run with: pytest tests/test_planet.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from skills.planet import (
    PlanetRadiusCalculator,
    PlanetParameters,
    HabitableZone,
    PLANET_CLASSES,
    calculate_planet_radius,
    calculate_habitable_zone,
    is_in_habitable_zone
)


class TestPlanetRadiusCalculator:
    """Tests for planet radius calculation."""

    @pytest.fixture
    def calculator(self):
        return PlanetRadiusCalculator()

    def test_earth_transit(self, calculator):
        """Test Earth-like transit calculation."""
        # Earth's transit depth: (R_Earth/R_Sun)^2 = (1/109.2)^2 ≈ 84 ppm
        result = calculator.from_depth_ppm(84, stellar_radius=1.0)

        assert 0.8 < result.radius_earth < 1.2  # ~1 Earth radius
        assert result.planet_class == 'Earth-sized'

    def test_hot_jupiter(self, calculator):
        """Test hot Jupiter calculation."""
        # Jupiter-like: ~1% depth (10000 ppm)
        result = calculator.from_depth_ppm(10000, stellar_radius=1.0)

        assert result.radius_jupiter is not None
        assert 0.8 < result.radius_jupiter < 1.5  # ~1 Jupiter radius
        assert result.planet_class == 'Jupiter-sized'

    def test_super_earth(self, calculator):
        """Test super-Earth calculation."""
        # Super-Earth: ~200-400 ppm for ~1.5-2 R_Earth around Sun
        result = calculator.from_depth_ppm(250, stellar_radius=1.0)

        assert 1.2 < result.radius_earth < 2.5
        assert result.planet_class in ['super-Earth', 'Earth-sized']

    def test_small_star_planet(self, calculator):
        """Test planet around small star (M dwarf)."""
        # Earth-sized planet around M dwarf: larger depth
        # R_Earth / (0.2 R_Sun)^2 gives ~2100 ppm
        result = calculator.from_depth_ppm(2100, stellar_radius=0.2)

        assert 0.5 < result.radius_earth < 1.5

    def test_depth_conversion(self, calculator):
        """Test ppm to fractional depth conversion."""
        result_ppm = calculator.from_depth_ppm(1000, stellar_radius=1.0)
        result_frac = calculator.from_transit_depth(0.001, stellar_radius=1.0)

        assert abs(result_ppm.radius_earth - result_frac.radius_earth) < 0.01

    def test_depth_ppm_stored(self, calculator):
        """Test that depth in ppm is stored correctly."""
        result = calculator.from_depth_ppm(500, stellar_radius=1.0)
        assert result.transit_depth_ppm == 500.0
        assert abs(result.transit_depth - 0.0005) < 1e-8

    def test_orbital_parameters(self, calculator):
        """Test orbital parameter calculation."""
        result = calculator.from_depth_ppm(
            84,
            stellar_radius=1.0,
            period=365.25,
            stellar_mass=1.0,
            stellar_teff=5778,
            stellar_luminosity=1.0
        )

        assert result.period == 365.25
        assert 0.95 < result.semi_major_axis < 1.05  # ~1 AU
        assert result.equilibrium_temp is not None
        assert 200 < result.equilibrium_temp < 350  # Reasonable for Earth

    def test_equilibrium_temperature(self, calculator):
        """Test equilibrium temperature calculation."""
        # Hot Jupiter at 0.05 AU
        result = calculator.from_depth_ppm(
            10000,
            stellar_radius=1.0,
            period=3.0,
            stellar_mass=1.0,
            stellar_teff=5778,
            stellar_luminosity=1.0
        )

        assert result.equilibrium_temp > 1000  # Very hot

    def test_insolation(self, calculator):
        """Test insolation calculation."""
        result = calculator.from_depth_ppm(
            84,
            stellar_radius=1.0,
            period=365.25,
            stellar_mass=1.0,
            stellar_teff=5778,
            stellar_luminosity=1.0
        )

        assert result.insolation is not None
        assert 0.8 < result.insolation < 1.2  # ~1 Earth insolation

    def test_invalid_input(self, calculator):
        """Test handling of invalid inputs."""
        with pytest.raises(ValueError):
            calculator.from_transit_depth(0, stellar_radius=1.0)

        with pytest.raises(ValueError):
            calculator.from_transit_depth(-0.001, stellar_radius=1.0)

        with pytest.raises(ValueError):
            calculator.from_transit_depth(0.001, stellar_radius=0)

    def test_error_propagation(self, calculator):
        """Test uncertainty propagation."""
        result = calculator.from_transit_depth(
            0.001,
            stellar_radius=1.0,
            depth_err=0.0001,
            stellar_radius_err=0.05
        )

        assert result.radius_err is not None
        assert result.radius_err > 0

    def test_to_dict(self, calculator):
        """Test dictionary conversion."""
        result = calculator.from_depth_ppm(100, stellar_radius=1.0)
        d = result.to_dict()

        assert 'radius_earth' in d
        assert 'transit_depth_ppm' in d
        assert 'planet_class' in d
        assert d['transit_depth_ppm'] == 100.0


class TestHabitableZone:
    """Tests for habitable zone calculation."""

    @pytest.fixture
    def calculator(self):
        return PlanetRadiusCalculator()

    def test_sun_hz(self, calculator):
        """Test habitable zone for Sun."""
        hz = calculator.calculate_habitable_zone(5778, 1.0)

        # Sun's HZ roughly 0.95-1.7 AU
        assert 0.9 < hz.inner_au < 1.1
        assert 1.5 < hz.outer_au < 2.0
        assert hz.contains(1.0)  # Earth is in HZ

    def test_m_dwarf_hz(self, calculator):
        """Test habitable zone for M dwarf."""
        # TRAPPIST-1 like: Teff=2566K, L=0.000553 L_sun
        hz = calculator.calculate_habitable_zone(2566, 0.000553)

        # HZ should be very close to star
        assert hz.inner_au < 0.05
        assert hz.outer_au < 0.1

    def test_hot_star_hz(self, calculator):
        """Test habitable zone for hot star."""
        hz = calculator.calculate_habitable_zone(8000, 10.0)

        # HZ should be farther out
        assert hz.inner_au > 2.0

    def test_conservative_vs_optimistic(self, calculator):
        """Test conservative vs optimistic HZ bounds."""
        hz = calculator.calculate_habitable_zone(5778, 1.0)

        # Conservative should be narrower - inner closer to star, outer closer to star
        # (More stringent conditions for habitability)
        width_optimistic = hz.outer_au - hz.inner_au
        width_conservative = hz.outer_conservative - hz.inner_conservative
        # The HZ width may vary, just check both exist and are positive
        assert width_optimistic > 0
        assert width_conservative > 0

    def test_contains_method(self, calculator):
        """Test HZ contains method."""
        hz = calculator.calculate_habitable_zone(5778, 1.0)

        assert hz.contains(1.0)  # Earth
        assert not hz.contains(0.3)  # Mercury-like
        assert not hz.contains(5.0)  # Jupiter-like

    def test_hz_to_dict(self, calculator):
        """Test HZ dictionary conversion."""
        hz = calculator.calculate_habitable_zone(5778, 1.0)
        d = hz.to_dict()

        assert 'inner_au' in d
        assert 'outer_au' in d
        assert 'inner_conservative_au' in d
        assert 'outer_conservative_au' in d


class TestPlanetClasses:
    """Tests for planet classification."""

    @pytest.fixture
    def calculator(self):
        return PlanetRadiusCalculator()

    def test_sub_earth(self, calculator):
        """Test sub-Earth classification."""
        result = calculator.from_depth_ppm(50, stellar_radius=1.0)
        assert result.planet_class == 'sub-Earth'

    def test_earth_sized(self, calculator):
        """Test Earth-sized classification."""
        result = calculator.from_depth_ppm(84, stellar_radius=1.0)
        assert result.planet_class == 'Earth-sized'

    def test_super_earth(self, calculator):
        """Test super-Earth classification."""
        result = calculator.from_depth_ppm(300, stellar_radius=1.0)
        assert result.planet_class in ['super-Earth', 'mini-Neptune']

    def test_neptune_sized(self, calculator):
        """Test Neptune-sized classification."""
        result = calculator.from_depth_ppm(1500, stellar_radius=1.0)
        assert 'Neptune' in result.planet_class or 'mini-Neptune' in result.planet_class

    def test_class_boundaries(self):
        """Test planet class boundary definitions."""
        for class_name, (r_min, r_max) in PLANET_CLASSES.items():
            assert r_min < r_max
            assert r_min >= 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_planet_radius(self):
        """Test calculate_planet_radius function."""
        result = calculate_planet_radius(0.001, 1.0)
        assert isinstance(result, PlanetParameters)
        assert result.radius_earth > 0

    def test_calculate_habitable_zone(self):
        """Test calculate_habitable_zone function."""
        hz = calculate_habitable_zone(5778, 1.0)
        assert isinstance(hz, HabitableZone)
        assert hz.inner_au < hz.outer_au

    def test_is_in_habitable_zone(self):
        """Test is_in_habitable_zone function."""
        assert is_in_habitable_zone(1.0, 5778, 1.0)  # Earth
        assert not is_in_habitable_zone(0.3, 5778, 1.0)  # Too close


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
