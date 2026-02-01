"""
LARUN Educational Trainer Node - ASTRA
=======================================

Interactive learning system for space science education powered by
ASTRA (Astronomical Science Tutor and Research Assistant).

Features:
- RAG-based AI tutor with knowledge from NASA, ESA, ISRO, JAXA, SpaceX, CNSA
- Short-term and long-term memory for personalized learning
- Web crawler for indexing space science sources
- Hot-swap model deployment with automatic updates
- Interactive lessons and quizzes

Usage:
    from nodes.trainer.src import ASTRA, AstraTutor

    tutor = AstraTutor()
    response = tutor.ask("What is an exoplanet?")
    print(response.answer)
"""

# Core trainer
from .trainer import EducationalTrainer
from .lesson_loader import LessonLoader
from .quiz_engine import QuizEngine
from .explainer import PredictionExplainer

# AI-powered tutor
from .llm_tutor import SpaceScienceTutor, TutorResponse
from .persona import AstraPersona, ASTRA, Personality
from .memory import MemoryManager, ShortTermMemory, LongTermMemory
from .crawler import KnowledgeIndexer, CrawlConfig
from .model_manager import ModelManager, DeploymentConfig, DeploymentStrategy

__all__ = [
    # Core
    'EducationalTrainer',
    'LessonLoader',
    'QuizEngine',
    'PredictionExplainer',

    # AI Tutor (ASTRA)
    'SpaceScienceTutor',
    'TutorResponse',
    'AstraPersona',
    'ASTRA',
    'Personality',

    # Memory
    'MemoryManager',
    'ShortTermMemory',
    'LongTermMemory',

    # Crawler
    'KnowledgeIndexer',
    'CrawlConfig',

    # Model Management
    'ModelManager',
    'DeploymentConfig',
    'DeploymentStrategy',
]


# Convenience alias
AstraTutor = SpaceScienceTutor
