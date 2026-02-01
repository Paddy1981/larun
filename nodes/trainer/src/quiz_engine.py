"""
Quiz Engine for LARUN Educational Trainer
==========================================

Manages quizzes, grading, and feedback for the educational system.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import random
import json


@dataclass
class Question:
    """A single quiz question."""
    question: str
    options: List[str]
    correct_answer: str
    explanation: str = ""
    question_type: str = "multiple_choice"  # multiple_choice, true_false
    difficulty: str = "medium"

    def is_correct(self, answer: str) -> bool:
        """Check if an answer is correct."""
        return answer.strip().lower() == self.correct_answer.strip().lower()


@dataclass
class Quiz:
    """A complete quiz with questions."""
    id: str
    title: str
    questions: List[Question]
    passing_score: float = 70.0
    shuffle: bool = True
    description: str = ""

    @property
    def num_questions(self) -> int:
        return len(self.questions)


@dataclass
class QuizResult:
    """Result of a completed quiz."""
    quiz_id: str
    score: float
    correct: int
    total: int
    passed: bool
    feedback: List[Dict[str, Any]] = field(default_factory=list)


class QuizEngine:
    """
    Quiz management and grading system.

    Supports:
    - Multiple choice questions
    - True/false questions
    - Automatic grading
    - Detailed feedback
    """

    def __init__(self, quizzes_dir: Path):
        """
        Initialize the quiz engine.

        Args:
            quizzes_dir: Path to quizzes directory
        """
        self.quizzes_dir = quizzes_dir
        self._quizzes_cache: Dict[str, Quiz] = {}

    def list_quizzes(self) -> List[Quiz]:
        """List all available quizzes."""
        quizzes = []

        if self.quizzes_dir.exists():
            for quiz_file in sorted(self.quizzes_dir.glob('*.yaml')):
                quiz = self._load_quiz_from_file(quiz_file)
                if quiz:
                    quizzes.append(quiz)

        # Add built-in quizzes
        if not quizzes:
            quizzes = [
                Quiz(
                    id='basics',
                    title='Space Science Basics',
                    questions=[],
                    passing_score=70.0,
                ),
                Quiz(
                    id='transits',
                    title='Transit Detection',
                    questions=[],
                    passing_score=70.0,
                ),
            ]

        return quizzes

    def load_quiz(self, quiz_id: str) -> Optional[Quiz]:
        """Load a specific quiz by ID."""
        if quiz_id in self._quizzes_cache:
            return self._quizzes_cache[quiz_id]

        # Try to load from file
        quiz_file = self.quizzes_dir / f"quiz_{quiz_id}.yaml"
        if quiz_file.exists():
            quiz = self._load_quiz_from_file(quiz_file)
            if quiz:
                self._quizzes_cache[quiz_id] = quiz
                return quiz

        # Try built-in quizzes
        builtin = self._get_builtin_quiz(quiz_id)
        if builtin:
            self._quizzes_cache[quiz_id] = builtin
            return builtin

        return None

    def _load_quiz_from_file(self, file_path: Path) -> Optional[Quiz]:
        """Load a quiz from a YAML file."""
        try:
            import yaml
            with open(file_path) as f:
                data = yaml.safe_load(f)

            questions = []
            for q_data in data.get('questions', []):
                questions.append(Question(
                    question=q_data['question'],
                    options=q_data.get('options', []),
                    correct_answer=q_data['answer'],
                    explanation=q_data.get('explanation', ''),
                    question_type=q_data.get('type', 'multiple_choice'),
                ))

            return Quiz(
                id=data.get('id', file_path.stem),
                title=data.get('title', file_path.stem),
                questions=questions,
                passing_score=data.get('passing_score', 70.0),
                description=data.get('description', ''),
            )
        except Exception:
            return None

    def grade_quiz(
        self,
        quiz_id: str,
        answers: Dict[int, str],
    ) -> Optional[QuizResult]:
        """
        Grade a quiz submission.

        Args:
            quiz_id: Quiz identifier
            answers: Dict mapping question number (1-indexed) to answer

        Returns:
            QuizResult with score and feedback
        """
        quiz = self.load_quiz(quiz_id)
        if quiz is None:
            return None

        correct = 0
        feedback = []

        for i, question in enumerate(quiz.questions):
            q_num = i + 1
            user_answer = answers.get(q_num, "")
            is_correct = question.is_correct(user_answer)

            if is_correct:
                correct += 1

            feedback.append({
                'question_number': q_num,
                'question': question.question,
                'your_answer': user_answer,
                'correct_answer': question.correct_answer,
                'is_correct': is_correct,
                'explanation': question.explanation,
            })

        total = len(quiz.questions)
        score = (correct / total * 100) if total > 0 else 0
        passed = score >= quiz.passing_score

        return QuizResult(
            quiz_id=quiz_id,
            score=score,
            correct=correct,
            total=total,
            passed=passed,
            feedback=feedback,
        )

    def _get_builtin_quiz(self, quiz_id: str) -> Optional[Quiz]:
        """Get a built-in quiz."""
        builtin_quizzes = {
            'basics': self._quiz_basics(),
            'transits': self._quiz_transits(),
        }
        return builtin_quizzes.get(quiz_id)

    # =========================================================================
    # Built-in Quizzes
    # =========================================================================

    def _quiz_basics(self) -> Quiz:
        return Quiz(
            id='basics',
            title='Space Science Basics',
            description='Test your knowledge of fundamental space science concepts.',
            passing_score=70.0,
            questions=[
                Question(
                    question="What is an exoplanet?",
                    options=[
                        "A) A planet in our solar system",
                        "B) A planet orbiting a star other than the Sun",
                        "C) A large asteroid",
                        "D) A moon of Jupiter",
                    ],
                    correct_answer="B",
                    explanation="Exoplanets are planets that orbit stars other than our Sun.",
                ),
                Question(
                    question="How many exoplanets have been confirmed as of 2024?",
                    options=[
                        "A) About 500",
                        "B) About 1,000",
                        "C) Over 5,000",
                        "D) Over 50,000",
                    ],
                    correct_answer="C",
                    explanation="As of 2024, over 5,000 exoplanets have been confirmed.",
                ),
                Question(
                    question="What is the most successful method for detecting exoplanets?",
                    options=[
                        "A) Direct imaging",
                        "B) Transit method",
                        "C) Astrometry",
                        "D) Gravitational waves",
                    ],
                    correct_answer="B",
                    explanation="The transit method has discovered the most exoplanets, especially through Kepler and TESS missions.",
                ),
                Question(
                    question="What is a 'Hot Jupiter'?",
                    options=[
                        "A) Jupiter when it's closest to the Sun",
                        "B) A gas giant orbiting very close to its star",
                        "C) A star that looks like Jupiter",
                        "D) A volcanic moon",
                    ],
                    correct_answer="B",
                    explanation="Hot Jupiters are gas giants that orbit very close to their host stars, making them very hot.",
                ),
                Question(
                    question="Which space telescope discovered the most exoplanets?",
                    options=[
                        "A) Hubble Space Telescope",
                        "B) James Webb Space Telescope",
                        "C) Kepler Space Telescope",
                        "D) Spitzer Space Telescope",
                    ],
                    correct_answer="C",
                    explanation="The Kepler Space Telescope discovered over 2,700 confirmed exoplanets during its mission.",
                ),
                Question(
                    question="What does the 'habitable zone' refer to?",
                    options=[
                        "A) Where humans can live without spacesuits",
                        "B) The region where liquid water could exist on a planet's surface",
                        "C) Areas with no asteroids",
                        "D) The center of a galaxy",
                    ],
                    correct_answer="B",
                    explanation="The habitable zone is the region around a star where temperatures allow liquid water to exist on a planet's surface.",
                ),
                Question(
                    question="What is a light curve?",
                    options=[
                        "A) The shape of a galaxy",
                        "B) A graph of brightness versus time",
                        "C) The path light takes through space",
                        "D) A type of telescope",
                    ],
                    correct_answer="B",
                    explanation="A light curve is a graph showing how an object's brightness changes over time.",
                ),
                Question(
                    question="True or False: All stars have the same brightness.",
                    options=["A) True", "B) False"],
                    correct_answer="B",
                    question_type="true_false",
                    explanation="Stars vary greatly in brightness due to differences in size, temperature, and distance.",
                ),
                Question(
                    question="What mission is currently surveying bright, nearby stars for exoplanets?",
                    options=[
                        "A) Kepler",
                        "B) TESS",
                        "C) Voyager",
                        "D) Apollo",
                    ],
                    correct_answer="B",
                    explanation="TESS (Transiting Exoplanet Survey Satellite) is surveying bright, nearby stars since 2018.",
                ),
                Question(
                    question="What percentage of Sun-like stars are estimated to have Earth-like planets in the habitable zone?",
                    options=[
                        "A) Less than 1%",
                        "B) About 5-20%",
                        "C) About 50%",
                        "D) Over 90%",
                    ],
                    correct_answer="B",
                    explanation="Studies suggest about 5-20% of Sun-like stars may have Earth-sized planets in their habitable zones.",
                ),
            ],
        )

    def _quiz_transits(self) -> Quiz:
        return Quiz(
            id='transits',
            title='Transit Detection',
            description='Test your understanding of the transit method for detecting exoplanets.',
            passing_score=70.0,
            questions=[
                Question(
                    question="What happens during a planetary transit?",
                    options=[
                        "A) A planet moves behind its star",
                        "B) A planet passes in front of its star, causing a brightness dip",
                        "C) A planet emits light",
                        "D) A planet's orbit changes",
                    ],
                    correct_answer="B",
                    explanation="During a transit, a planet passes in front of its star from our viewpoint, blocking some light.",
                ),
                Question(
                    question="What does transit depth tell us about?",
                    options=[
                        "A) The planet's mass",
                        "B) The planet's color",
                        "C) The planet's size relative to the star",
                        "D) The planet's temperature",
                    ],
                    correct_answer="C",
                    explanation="Transit depth = (planet radius / star radius)², so it tells us the size ratio.",
                ),
                Question(
                    question="If a planet like Jupiter transits the Sun, approximately what depth would we see?",
                    options=[
                        "A) 0.001% (10 ppm)",
                        "B) 0.01% (100 ppm)",
                        "C) 0.1% (1000 ppm)",
                        "D) 1% (10,000 ppm)",
                    ],
                    correct_answer="D",
                    explanation="Jupiter's radius is about 0.1 times the Sun's radius, so depth ≈ 0.1² = 1%.",
                ),
                Question(
                    question="What is 'ingress' in a transit light curve?",
                    options=[
                        "A) When the planet is fully in front of the star",
                        "B) When the planet starts crossing the stellar disk",
                        "C) When the planet exits the stellar disk",
                        "D) When the planet is behind the star",
                    ],
                    correct_answer="B",
                    explanation="Ingress is the phase when the planet begins to cross in front of the star.",
                ),
                Question(
                    question="How can we determine an exoplanet's orbital period from transits?",
                    options=[
                        "A) By measuring the transit depth",
                        "B) By measuring the time between consecutive transits",
                        "C) By measuring the star's brightness",
                        "D) By measuring the planet's color",
                    ],
                    correct_answer="B",
                    explanation="The orbital period is the time between consecutive transits of the same planet.",
                ),
                Question(
                    question="What is limb darkening?",
                    options=[
                        "A) The edge of the galaxy getting darker",
                        "B) Stars appearing dimmer at their edges",
                        "C) Planets blocking more light at night",
                        "D) Telescopes getting dirty",
                    ],
                    correct_answer="B",
                    explanation="Limb darkening is the effect where stars appear dimmer at their edges (limb) than at their center.",
                ),
                Question(
                    question="What minimum signal-to-noise ratio (SNR) is typically needed for transit detection?",
                    options=[
                        "A) SNR > 1",
                        "B) SNR > 3",
                        "C) SNR > 7",
                        "D) SNR > 100",
                    ],
                    correct_answer="C",
                    explanation="Typically SNR > 7 is required for confident detection of transit signals.",
                ),
                Question(
                    question="Why might an eclipsing binary be mistaken for an exoplanet?",
                    options=[
                        "A) They have the same mass",
                        "B) Both cause periodic brightness dips",
                        "C) They emit the same light",
                        "D) They have the same orbital period",
                    ],
                    correct_answer="B",
                    explanation="Both exoplanet transits and eclipsing binaries cause periodic brightness dips, though binary dips are usually deeper.",
                ),
                Question(
                    question="What is a 'grazing transit'?",
                    options=[
                        "A) When a planet barely touches the star's edge",
                        "B) When a planet orbits very fast",
                        "C) When a planet is very large",
                        "D) When two planets transit at once",
                    ],
                    correct_answer="A",
                    explanation="A grazing transit occurs when the planet only crosses the edge of the stellar disk, producing a V-shaped light curve.",
                ),
                Question(
                    question="True or False: We can only detect transits if the orbital plane is aligned with our line of sight.",
                    options=["A) True", "B) False"],
                    correct_answer="A",
                    question_type="true_false",
                    explanation="Transits require geometric alignment - we can only see them when the planet passes directly between us and the star.",
                ),
            ],
        )
