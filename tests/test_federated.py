"""
Unit Tests for LARUN Federated Multi-Model System
=================================================
Tests for FPP Calculator, Model Registry, and Gaia Integration.

Run with: python -m pytest tests/test_federated.py -v
        or: python tests/test_federated.py
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestFPPCalculator(unittest.TestCase):
    """Tests for False Positive Probability Calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        from skills.fpp import FPPCalculator, StellarParams
        self.calc = FPPCalculator()
        self.stellar = StellarParams()
    
    def test_basic_calculation(self):
        """Test basic FPP calculation without vetting results."""
        from skills.fpp import FPPCalculator
        calc = FPPCalculator()
        
        result = calc.calculate(
            period=3.5,
            depth_ppm=1200,
            target_name="Test"
        )
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.fpp, 0.0)
        self.assertLessEqual(result.fpp, 1.0)
        self.assertIn(result.disposition, ["VALIDATED", "CANDIDATE", "LIKELY_FP"])
    
    def test_planet_like_signal(self):
        """Test that shallow transits favor planet hypothesis."""
        from skills.fpp import FPPCalculator, StellarParams
        calc = FPPCalculator()
        
        # Small depth (1000 ppm = 0.1%), typical of planet
        result = calc.calculate(
            period=5.0,
            depth_ppm=1000,
            stellar_params=StellarParams(),
            target_name="Planet Test"
        )
        
        # Should have reasonable planet probability
        self.assertGreater(result.fpp, 0.0)
        self.assertLess(result.fpp, 0.5)  # More likely planet than not
    
    def test_eb_like_signal(self):
        """Test that deep eclipses favor EB hypothesis."""
        from skills.fpp import FPPCalculator
        calc = FPPCalculator()
        
        # Very deep eclipse (5% depth), typical of EB
        result = calc.calculate(
            period=2.0,
            depth_ppm=50000,
            target_name="EB Test"
        )
        
        # Should have high FPP (likely EB)
        self.assertGreater(result.fpp, 0.9)
        self.assertEqual(result.disposition, "LIKELY_FP")
    
    def test_scenarios_sum_to_one(self):
        """Test that scenario probabilities sum to 1."""
        from skills.fpp import FPPCalculator
        calc = FPPCalculator()
        
        result = calc.calculate(period=3.0, depth_ppm=2000)
        
        total = sum(s.posterior for s in result.scenarios)
        self.assertAlmostEqual(total, 1.0, places=5)
    
    def test_to_dict(self):
        """Test serialization to dict."""
        from skills.fpp import FPPCalculator
        calc = FPPCalculator()
        
        result = calc.calculate(period=3.0, depth_ppm=2000)
        result_dict = result.to_dict()
        
        self.assertIn('fpp', result_dict)
        self.assertIn('disposition', result_dict)
        self.assertIn('scenarios', result_dict)


class TestModelRegistry(unittest.TestCase):
    """Tests for Federated Model Registry."""
    
    def setUp(self):
        """Set up test fixtures."""
        from federated.registry import ModelRegistry, ModelMetadata
        self.test_path = Path("test_registry_temp.json")
        self.registry = ModelRegistry(self.test_path)
        
        # Clean up any existing test file
        if self.test_path.exists():
            self.test_path.unlink()
    
    def tearDown(self):
        """Clean up test files."""
        if self.test_path.exists():
            self.test_path.unlink()
    
    def test_register_model(self):
        """Test model registration."""
        from federated.registry import ModelMetadata
        
        meta = ModelMetadata(
            model_id="test_model_v1",
            version="1.0.0",
            task="transit_detection",
            accuracy=0.85,
            input_shape=(1024, 1),
            created_at="2026-01-31T00:00:00Z",
            file_path="models/test.tflite"
        )
        
        self.registry.register(meta, save=False)
        
        self.assertIn("test_model_v1", self.registry.models)
        self.assertEqual(len(self.registry.models), 1)
    
    def test_get_by_task(self):
        """Test querying models by task."""
        from federated.registry import ModelMetadata
        
        # Register models for different tasks
        self.registry.register(ModelMetadata(
            model_id="transit_v1", version="1.0.0",
            task="transit_detection", accuracy=0.82,
            input_shape=(1024, 1), created_at="2026-01-31",
            file_path="test.h5"
        ), save=False)
        
        self.registry.register(ModelMetadata(
            model_id="binary_v1", version="1.0.0",
            task="binary_discrimination", accuracy=0.78,
            input_shape=(1024, 1), created_at="2026-01-31",
            file_path="test.h5"
        ), save=False)
        
        transit_models = self.registry.get_by_task("transit_detection")
        binary_models = self.registry.get_by_task("binary_discrimination")
        
        self.assertEqual(len(transit_models), 1)
        self.assertEqual(len(binary_models), 1)
        self.assertEqual(transit_models[0].model_id, "transit_v1")
    
    def test_get_best_for_task(self):
        """Test getting best model for a task."""
        from federated.registry import ModelMetadata
        
        self.registry.register(ModelMetadata(
            model_id="model_low", version="1.0.0",
            task="test_task", accuracy=0.70,
            input_shape=(1024, 1), created_at="2026-01-31",
            file_path="test.h5"
        ), save=False)
        
        self.registry.register(ModelMetadata(
            model_id="model_high", version="1.0.0",
            task="test_task", accuracy=0.95,
            input_shape=(1024, 1), created_at="2026-01-31",
            file_path="test.h5"
        ), save=False)
        
        best = self.registry.get_best_for_task("test_task")
        
        self.assertIsNotNone(best)
        self.assertEqual(best.model_id, "model_high")
        self.assertEqual(best.accuracy, 0.95)
    
    def test_unregister(self):
        """Test model unregistration."""
        from federated.registry import ModelMetadata
        
        self.registry.register(ModelMetadata(
            model_id="temp_model", version="1.0.0",
            task="test", accuracy=0.8,
            input_shape=(100, 1), created_at="2026-01-31",
            file_path="test.h5"
        ), save=False)
        
        self.assertIn("temp_model", self.registry.models)
        
        result = self.registry.unregister("temp_model", save=False)
        
        self.assertTrue(result)
        self.assertNotIn("temp_model", self.registry.models)


class TestGaiaIntegration(unittest.TestCase):
    """Tests for Gaia DR3 Integration."""
    
    def test_stellar_params_creation(self):
        """Test StellarParams dataclass."""
        from integrations.gaia import StellarParams
        
        params = StellarParams(
            teff=5778.0,
            logg=4.44,
            radius=1.0
        )
        
        self.assertEqual(params.teff, 5778.0)
        self.assertEqual(params.logg, 4.44)
        self.assertTrue(params.is_valid)
    
    def test_spectral_type_classification(self):
        """Test spectral type estimation."""
        from integrations.gaia import StellarParams
        
        # G-type star (Sun-like)
        g_star = StellarParams(teff=5778.0)
        self.assertEqual(g_star.spectral_type(), "G")
        
        # M-type star
        m_star = StellarParams(teff=3500.0)
        self.assertEqual(m_star.spectral_type(), "M")
        
        # A-type star
        a_star = StellarParams(teff=8000.0)
        self.assertEqual(a_star.spectral_type(), "A")
    
    def test_luminosity_class(self):
        """Test luminosity class estimation."""
        from integrations.gaia import StellarParams
        
        # Dwarf (main sequence)
        dwarf = StellarParams(logg=4.5)
        self.assertEqual(dwarf.luminosity_class(), "V")
        
        # Giant
        giant = StellarParams(logg=2.5)
        self.assertEqual(giant.luminosity_class(), "III")
    
    def test_stellar_params_to_dict(self):
        """Test serialization."""
        from integrations.gaia import StellarParams
        
        params = StellarParams(
            tic_id="123456",
            teff=6000.0,
            logg=4.3,
            radius=1.1
        )
        
        result = params.to_dict()
        
        self.assertIn('teff_k', result)
        self.assertIn('logg', result)
        self.assertEqual(result['tic_id'], "123456")
    
    def test_gaia_client_fallback(self):
        """Test GaiaClient fallback for unavailable targets."""
        from integrations.gaia import GaiaClient
        
        client = GaiaClient()
        fallback = client._fallback_params("TEST123")
        
        self.assertEqual(fallback.teff, 5778.0)  # Sun-like defaults
        self.assertFalse(fallback.is_valid)
        self.assertIn('fallback', fallback.quality_flags)


class TestProtocol(unittest.TestCase):
    """Tests for Federated Inference Protocol."""
    
    def test_inference_request_creation(self):
        """Test InferenceRequest creation."""
        from federated.protocol import InferenceRequest
        
        request = InferenceRequest(
            task="transit_detection",
            models=["model_a", "model_b"]
        )
        
        self.assertEqual(request.task, "transit_detection")
        self.assertEqual(len(request.models), 2)
        self.assertIsNotNone(request.request_id)
    
    def test_inference_request_serialization(self):
        """Test request serialization round-trip."""
        from federated.protocol import InferenceRequest
        
        data = np.random.randn(1024, 1).astype(np.float32)
        request = InferenceRequest(
            task="test",
            data=data
        )
        
        # Serialize
        req_dict = request.to_dict()
        
        # Deserialize
        restored = InferenceRequest.from_dict(req_dict)
        
        self.assertEqual(restored.request_id, request.request_id)
        self.assertEqual(restored.task, request.task)
        np.testing.assert_array_almost_equal(restored.data, data, decimal=4)
    
    def test_inference_response_error(self):
        """Test error response creation."""
        from federated.protocol import InferenceResponse
        
        response = InferenceResponse.error_response(
            request_id="test-123",
            error="Model not found"
        )
        
        self.assertFalse(response.success)
        self.assertEqual(response.error, "Model not found")
        self.assertEqual(response.request_id, "test-123")


class TestOrchestrator(unittest.TestCase):
    """Tests for Model Orchestrator."""
    
    def test_empty_registry_prediction(self):
        """Test prediction with empty registry."""
        from federated.registry import ModelRegistry
        from federated.orchestrator import ModelOrchestrator
        
        registry = ModelRegistry(Path("empty_test.json"))
        orchestrator = ModelOrchestrator(registry)
        
        data = np.random.randn(1024, 1).astype(np.float32)
        result = orchestrator.predict(data, task="nonexistent_task")
        
        self.assertEqual(len(result.individual_predictions), 0)
        self.assertEqual(result.confidence, 0.0)
        
        # Cleanup
        Path("empty_test.json").unlink(missing_ok=True)
    
    def test_ensemble_prediction_structure(self):
        """Test EnsemblePrediction dataclass structure."""
        from federated.orchestrator import EnsemblePrediction, ModelPrediction
        
        pred = ModelPrediction(
            model_id="test_model",
            probabilities=np.array([0.2, 0.3, 0.5]),
            predicted_class=2,
            confidence=0.5,
            latency_ms=10.0
        )
        
        ensemble = EnsemblePrediction(
            predicted_class=2,
            confidence=0.5,
            probabilities=np.array([0.2, 0.3, 0.5]),
            individual_predictions=[pred],
            model_agreement=1.0,
            total_latency_ms=10.0
        )
        
        result_dict = ensemble.to_dict()
        
        self.assertIn('predicted_class', result_dict)
        self.assertIn('individual_predictions', result_dict)
        self.assertEqual(result_dict['num_models'], 1)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestFPPCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestModelRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestGaiaIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestProtocol))
    suite.addTests(loader.loadTestsFromTestCase(TestOrchestrator))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("LARUN Federated System Unit Tests")
    print("=" * 70)
    success = run_tests()
    print("\n" + "=" * 70)
    print(f"Overall: {'PASSED' if success else 'FAILED'}")
    print("=" * 70)
    sys.exit(0 if success else 1)
