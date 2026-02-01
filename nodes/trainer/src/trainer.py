"""
Educational Trainer for LARUN
=============================

Main trainer class that orchestrates lessons, quizzes, and explainers.
Provides an interactive learning experience for space science concepts.

Usage:
    trainer = EducationalTrainer()
    trainer.start_lesson("exoplanets_intro")
    trainer.take_quiz("basics")
    trainer.explain_prediction(prediction_result)
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from .lesson_loader import LessonLoader, Lesson
from .quiz_engine import QuizEngine, QuizResult
from .explainer import PredictionExplainer


@dataclass
class LearningProgress:
    """Tracks a user's learning progress."""
    user_id: str = "default"
    lessons_completed: List[str] = field(default_factory=list)
    quiz_scores: Dict[str, float] = field(default_factory=dict)
    topics_mastered: List[str] = field(default_factory=list)
    total_time_minutes: float = 0.0
    last_activity: str = ""
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'lessons_completed': self.lessons_completed,
            'quiz_scores': self.quiz_scores,
            'topics_mastered': self.topics_mastered,
            'total_time_minutes': self.total_time_minutes,
            'last_activity': self.last_activity,
            'started_at': self.started_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningProgress':
        return cls(**data)


class EducationalTrainer:
    """
    Interactive educational trainer for space science.

    Provides:
    - Structured lessons on astronomical concepts
    - Interactive quizzes with scoring
    - Prediction explainers for ML models
    - Progress tracking
    """

    def __init__(self, content_dir: Optional[Path] = None):
        """
        Initialize the trainer.

        Args:
            content_dir: Path to content directory (defaults to node's content/)
        """
        if content_dir is None:
            content_dir = Path(__file__).parent.parent / 'content'

        self.content_dir = content_dir
        self.lesson_loader = LessonLoader(content_dir / 'lessons')
        self.quiz_engine = QuizEngine(content_dir / 'quizzes')
        self.explainer = PredictionExplainer()

        # Load or create progress
        self.progress_file = content_dir.parent / 'progress.json'
        self.progress = self._load_progress()

    def _load_progress(self) -> LearningProgress:
        """Load progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    return LearningProgress.from_dict(json.load(f))
            except Exception:
                pass
        return LearningProgress()

    def _save_progress(self) -> None:
        """Save progress to file."""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress.to_dict(), f, indent=2)

    # =========================================================================
    # Lesson Methods
    # =========================================================================

    def list_topics(self) -> List[Dict[str, Any]]:
        """
        List all available learning topics.

        Returns:
            List of topic dictionaries with metadata
        """
        topics = []

        for lesson in self.lesson_loader.list_lessons():
            completed = lesson.id in self.progress.lessons_completed
            topics.append({
                'id': lesson.id,
                'title': lesson.title,
                'difficulty': lesson.difficulty,
                'estimated_time': lesson.estimated_time_min,
                'completed': completed,
            })

        return topics

    def start_lesson(self, topic_id: str) -> Dict[str, Any]:
        """
        Start a lesson on the given topic.

        Args:
            topic_id: Lesson identifier

        Returns:
            Lesson content and metadata
        """
        lesson = self.lesson_loader.load_lesson(topic_id)

        if lesson is None:
            return {
                'success': False,
                'error': f"Lesson '{topic_id}' not found",
                'available': [l.id for l in self.lesson_loader.list_lessons()],
            }

        self.progress.last_activity = datetime.utcnow().isoformat()
        self._save_progress()

        return {
            'success': True,
            'lesson': {
                'id': lesson.id,
                'title': lesson.title,
                'difficulty': lesson.difficulty,
                'content': lesson.content,
                'sections': lesson.sections,
                'key_points': lesson.key_points,
                'further_reading': lesson.further_reading,
            },
        }

    def complete_lesson(self, topic_id: str, time_spent_min: float = 0) -> Dict[str, Any]:
        """
        Mark a lesson as completed.

        Args:
            topic_id: Lesson identifier
            time_spent_min: Time spent on lesson

        Returns:
            Completion status and next recommended lesson
        """
        if topic_id not in self.progress.lessons_completed:
            self.progress.lessons_completed.append(topic_id)
            self.progress.total_time_minutes += time_spent_min

        self.progress.last_activity = datetime.utcnow().isoformat()
        self._save_progress()

        # Find next recommended lesson
        all_lessons = self.lesson_loader.list_lessons()
        next_lesson = None

        for lesson in all_lessons:
            if lesson.id not in self.progress.lessons_completed:
                next_lesson = lesson.id
                break

        return {
            'success': True,
            'message': f"Lesson '{topic_id}' completed!",
            'total_completed': len(self.progress.lessons_completed),
            'next_recommended': next_lesson,
        }

    # =========================================================================
    # Quiz Methods
    # =========================================================================

    def list_quizzes(self) -> List[Dict[str, Any]]:
        """List available quizzes."""
        quizzes = []

        for quiz in self.quiz_engine.list_quizzes():
            score = self.progress.quiz_scores.get(quiz.id)
            quizzes.append({
                'id': quiz.id,
                'title': quiz.title,
                'questions': quiz.num_questions,
                'passing_score': quiz.passing_score,
                'best_score': score,
                'passed': score is not None and score >= quiz.passing_score,
            })

        return quizzes

    def take_quiz(self, quiz_id: str) -> Dict[str, Any]:
        """
        Start a quiz.

        Args:
            quiz_id: Quiz identifier

        Returns:
            Quiz questions (answers not included)
        """
        quiz = self.quiz_engine.load_quiz(quiz_id)

        if quiz is None:
            return {
                'success': False,
                'error': f"Quiz '{quiz_id}' not found",
                'available': [q.id for q in self.quiz_engine.list_quizzes()],
            }

        # Return questions without answers
        questions = []
        for i, q in enumerate(quiz.questions):
            questions.append({
                'number': i + 1,
                'question': q.question,
                'options': q.options,
                'type': q.question_type,
            })

        return {
            'success': True,
            'quiz_id': quiz.id,
            'title': quiz.title,
            'questions': questions,
            'passing_score': quiz.passing_score,
        }

    def submit_quiz(
        self,
        quiz_id: str,
        answers: Dict[int, str],
    ) -> Dict[str, Any]:
        """
        Submit quiz answers and get results.

        Args:
            quiz_id: Quiz identifier
            answers: Dict mapping question number to answer

        Returns:
            Quiz results with score and feedback
        """
        result = self.quiz_engine.grade_quiz(quiz_id, answers)

        if result is None:
            return {
                'success': False,
                'error': f"Quiz '{quiz_id}' not found",
            }

        # Update progress
        current_best = self.progress.quiz_scores.get(quiz_id, 0)
        if result.score > current_best:
            self.progress.quiz_scores[quiz_id] = result.score

        if result.passed and quiz_id not in self.progress.topics_mastered:
            self.progress.topics_mastered.append(quiz_id)

        self.progress.last_activity = datetime.utcnow().isoformat()
        self._save_progress()

        return {
            'success': True,
            'score': result.score,
            'passed': result.passed,
            'correct': result.correct,
            'total': result.total,
            'feedback': result.feedback,
            'new_best': result.score > current_best,
        }

    # =========================================================================
    # Explainer Methods
    # =========================================================================

    def explain_prediction(
        self,
        prediction: Dict[str, Any],
        data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Explain why a model made a specific prediction.

        Args:
            prediction: Model prediction result
            data: Optional input data for visualization

        Returns:
            Explanation with features and comparison examples
        """
        return self.explainer.explain(prediction, data)

    # =========================================================================
    # Progress Methods
    # =========================================================================

    def get_progress(self) -> Dict[str, Any]:
        """Get current learning progress."""
        total_lessons = len(self.lesson_loader.list_lessons())
        total_quizzes = len(self.quiz_engine.list_quizzes())

        return {
            'lessons_completed': len(self.progress.lessons_completed),
            'total_lessons': total_lessons,
            'completion_percent': (
                len(self.progress.lessons_completed) / total_lessons * 100
                if total_lessons > 0 else 0
            ),
            'quizzes_passed': len(self.progress.topics_mastered),
            'total_quizzes': total_quizzes,
            'quiz_scores': self.progress.quiz_scores,
            'topics_mastered': self.progress.topics_mastered,
            'total_time_minutes': self.progress.total_time_minutes,
            'last_activity': self.progress.last_activity,
        }

    def reset_progress(self) -> Dict[str, Any]:
        """Reset all learning progress."""
        self.progress = LearningProgress()
        self._save_progress()
        return {'success': True, 'message': 'Progress reset'}

    # =========================================================================
    # Interactive Mode
    # =========================================================================

    def interactive_menu(self) -> str:
        """Generate interactive menu text."""
        lines = [
            "=" * 50,
            " LARUN Space Science Trainer",
            "=" * 50,
            "",
            "Commands:",
            "  /learn topics        - List available topics",
            "  /learn topic <id>    - Start a lesson",
            "  /learn quiz <id>     - Take a quiz",
            "  /learn explain       - Explain last prediction",
            "  /learn progress      - Show your progress",
            "",
            "Topics:",
        ]

        for topic in self.list_topics():
            status = "" if topic['completed'] else "  "
            lines.append(f"  {status} {topic['id']}: {topic['title']}")

        lines.append("")
        lines.append("Quizzes:")

        for quiz in self.list_quizzes():
            status = "" if quiz['passed'] else "  "
            lines.append(f"  {status} {quiz['id']}: {quiz['title']}")

        return "\n".join(lines)
