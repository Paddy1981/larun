"""
LARUN Specialized Models Package
=================================
Specialized TinyML models for astronomical classification.

Models:
- StellarClassifier: Classify stellar types and luminosity classes
- BinaryDiscriminator: Distinguish EBs from planets
- HabitabilityAssessor: Assess planetary habitability
"""

from .stellar_classifier import StellarClassifier, StellarClassifierResult
from .binary_discriminator import BinaryDiscriminator, BinaryResult
from .habitability_assessor import HabitabilityAssessor, HabitabilityResult

__all__ = [
    'StellarClassifier',
    'StellarClassifierResult',
    'BinaryDiscriminator',
    'BinaryResult',
    'HabitabilityAssessor',
    'HabitabilityResult',
]
