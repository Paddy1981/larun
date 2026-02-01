"""
Memory System for LARUN Space Science Tutor
=============================================

Implements both short-term (conversation) and long-term (persistent) memory
for contextual, personalized learning experiences.

Memory Types:
- Short-term: Current conversation context, recent questions
- Long-term: User preferences, learning history, indexed knowledge
- Episodic: Specific learning sessions and interactions
- Semantic: Factual knowledge from crawled sources
"""

import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
import threading


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    memory_type: str  # short_term, long_term, episodic, semantic
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    last_accessed: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    decay_rate: float = 0.1  # For short-term memory decay

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        return cls(**data)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # user, assistant
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sources: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearningSession:
    """A complete learning session."""
    session_id: str
    user_id: str = "default"
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: Optional[str] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    topics_covered: List[str] = field(default_factory=list)
    quiz_scores: Dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def add_turn(self, role: str, content: str, **kwargs) -> None:
        self.turns.append(ConversationTurn(role=role, content=content, **kwargs))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['turns'] = [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.turns]
        return d


@dataclass
class UserProfile:
    """User learning profile for personalization."""
    user_id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    topics_interested: List[str] = field(default_factory=list)
    topics_mastered: List[str] = field(default_factory=list)
    difficulty_preference: str = "intermediate"  # beginner, intermediate, advanced
    learning_style: str = "visual"  # visual, textual, interactive
    total_sessions: int = 0
    total_questions: int = 0
    avg_quiz_score: float = 0.0
    favorite_sources: List[str] = field(default_factory=list)
    last_active: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Short-Term Memory (Conversation Context)
# =============================================================================

class ShortTermMemory:
    """
    Short-term memory for current conversation context.

    Features:
    - Sliding window of recent exchanges
    - Automatic decay of old entries
    - Topic tracking
    - Context summarization
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_tokens_estimate: int = 4000,
        decay_after_minutes: int = 30,
    ):
        self.max_turns = max_turns
        self.max_tokens = max_tokens_estimate
        self.decay_time = timedelta(minutes=decay_after_minutes)

        self._turns: deque = deque(maxlen=max_turns)
        self._topics: List[str] = []
        self._summary: str = ""
        self._last_update = datetime.utcnow()

    def add(self, role: str, content: str, **metadata) -> None:
        """Add a conversation turn."""
        turn = ConversationTurn(role=role, content=content, **metadata)
        self._turns.append(turn)
        self._last_update = datetime.utcnow()

        # Extract topics
        topics = metadata.get('topics', [])
        for topic in topics:
            if topic not in self._topics:
                self._topics.append(topic)

    def get_context(self, max_turns: Optional[int] = None) -> List[ConversationTurn]:
        """Get recent conversation context."""
        self._apply_decay()

        n = max_turns or len(self._turns)
        return list(self._turns)[-n:]

    def get_context_string(self, max_turns: Optional[int] = None) -> str:
        """Get context as a formatted string."""
        turns = self.get_context(max_turns)

        lines = []
        for turn in turns:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")

        return "\n".join(lines)

    def get_topics(self) -> List[str]:
        """Get topics discussed in this conversation."""
        return self._topics.copy()

    def get_last_question(self) -> Optional[str]:
        """Get the last user question."""
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn.content
        return None

    def get_last_answer(self) -> Optional[str]:
        """Get the last assistant answer."""
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.content
        return None

    def summarize(self) -> str:
        """Create a summary of the conversation."""
        if not self._turns:
            return ""

        questions = [t.content for t in self._turns if t.role == "user"]
        topics = self._topics[:5]

        summary = f"Discussed {len(questions)} questions"
        if topics:
            summary += f" about: {', '.join(topics)}"

        return summary

    def clear(self) -> None:
        """Clear short-term memory."""
        self._turns.clear()
        self._topics.clear()
        self._summary = ""

    def _apply_decay(self) -> None:
        """Remove old entries based on decay time."""
        if not self._turns:
            return

        cutoff = datetime.utcnow() - self.decay_time

        while self._turns:
            oldest = self._turns[0]
            try:
                oldest_time = datetime.fromisoformat(oldest.timestamp.replace('Z', '+00:00'))
                if oldest_time.replace(tzinfo=None) < cutoff:
                    self._turns.popleft()
                else:
                    break
            except (ValueError, AttributeError):
                break


# =============================================================================
# Long-Term Memory (Persistent Knowledge)
# =============================================================================

class LongTermMemory:
    """
    Long-term persistent memory using SQLite.

    Features:
    - Persistent storage across sessions
    - Semantic search with embeddings
    - User profiles and learning history
    - Knowledge base from crawled sources
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Knowledge entries
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT,
                    title TEXT,
                    url TEXT,
                    embedding BLOB,
                    created_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5
                );

                -- User profiles
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                -- Learning sessions
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_json TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

                -- Episodic memories (specific interactions)
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    content TEXT,
                    context TEXT,
                    timestamp TEXT,
                    importance REAL DEFAULT 0.5,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge(source);
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);
            """)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    # -------------------------------------------------------------------------
    # Knowledge Management
    # -------------------------------------------------------------------------

    def add_knowledge(
        self,
        content: str,
        source: str,
        title: str = "",
        url: str = "",
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
    ) -> str:
        """Add knowledge to long-term memory."""
        doc_id = hashlib.sha256(f"{source}:{content[:100]}".encode()).hexdigest()[:16]

        embedding_blob = None
        if embedding:
            import struct
            embedding_blob = struct.pack(f'{len(embedding)}f', *embedding)

        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO knowledge
                    (id, content, source, title, url, embedding, created_at, importance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id, content, source, title, url, embedding_blob,
                    datetime.utcnow().isoformat(), importance
                ))

        return doc_id

    def search_knowledge(
        self,
        query_embedding: List[float],
        source_filter: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search knowledge by embedding similarity."""
        with self._get_connection() as conn:
            if source_filter:
                placeholders = ','.join('?' * len(source_filter))
                rows = conn.execute(f"""
                    SELECT id, content, source, title, url, embedding
                    FROM knowledge
                    WHERE source IN ({placeholders})
                """, source_filter).fetchall()
            else:
                rows = conn.execute("""
                    SELECT id, content, source, title, url, embedding
                    FROM knowledge
                """).fetchall()

        # Score by cosine similarity
        results = []
        for row in rows:
            if row[5]:  # Has embedding
                import struct
                n = len(row[5]) // 4
                stored_embedding = list(struct.unpack(f'{n}f', row[5]))
                score = self._cosine_similarity(query_embedding, stored_embedding)
                results.append({
                    'id': row[0],
                    'content': row[1],
                    'source': row[2],
                    'title': row[3],
                    'url': row[4],
                    'score': score,
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    def get_knowledge_stats(self) -> Dict[str, int]:
        """Get knowledge base statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
            by_source = conn.execute("""
                SELECT source, COUNT(*) FROM knowledge GROUP BY source
            """).fetchall()

        return {
            'total': total,
            'by_source': dict(by_source),
        }

    # -------------------------------------------------------------------------
    # User Profile Management
    # -------------------------------------------------------------------------

    def get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create a user profile."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT profile_json FROM users WHERE user_id = ?",
                (user_id,)
            ).fetchone()

        if row:
            data = json.loads(row[0])
            return UserProfile(**data)

        # Create new profile
        profile = UserProfile(user_id=user_id)
        self.save_user_profile(profile)
        return profile

    def save_user_profile(self, profile: UserProfile) -> None:
        """Save a user profile."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO users (user_id, profile_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    profile.user_id,
                    json.dumps(profile.to_dict()),
                    profile.created_at,
                    datetime.utcnow().isoformat(),
                ))

    def update_user_stats(
        self,
        user_id: str,
        questions_asked: int = 0,
        quiz_score: Optional[float] = None,
        topics: Optional[List[str]] = None,
    ) -> None:
        """Update user statistics."""
        profile = self.get_user_profile(user_id)

        profile.total_questions += questions_asked
        profile.last_active = datetime.utcnow().isoformat()

        if topics:
            for topic in topics:
                if topic not in profile.topics_interested:
                    profile.topics_interested.append(topic)

        if quiz_score is not None:
            # Update rolling average
            n = profile.total_sessions or 1
            profile.avg_quiz_score = (
                (profile.avg_quiz_score * n + quiz_score) / (n + 1)
            )

        self.save_user_profile(profile)

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def save_session(self, session: LearningSession) -> None:
        """Save a learning session."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions
                    (session_id, user_id, session_json, started_at, ended_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.user_id,
                    json.dumps(session.to_dict()),
                    session.started_at,
                    session.ended_at,
                ))

    def get_recent_sessions(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[LearningSession]:
        """Get recent learning sessions for a user."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT session_json FROM sessions
                WHERE user_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()

        sessions = []
        for row in rows:
            data = json.loads(row[0])
            session = LearningSession(**{
                k: v for k, v in data.items()
                if k in LearningSession.__dataclass_fields__
            })
            sessions.append(session)

        return sessions

    # -------------------------------------------------------------------------
    # Episodic Memory
    # -------------------------------------------------------------------------

    def add_episode(
        self,
        user_id: str,
        content: str,
        context: str = "",
        importance: float = 0.5,
    ) -> str:
        """Add an episodic memory."""
        episode_id = hashlib.sha256(
            f"{user_id}:{content[:50]}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO episodes (id, user_id, content, context, timestamp, importance)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    episode_id, user_id, content, context,
                    datetime.utcnow().isoformat(), importance
                ))

        return episode_id

    def get_relevant_episodes(
        self,
        user_id: str,
        context: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get episodes relevant to current context."""
        with self._get_connection() as conn:
            # Simple keyword matching - in production use embeddings
            rows = conn.execute("""
                SELECT id, content, context, timestamp, importance
                FROM episodes
                WHERE user_id = ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """, (user_id, limit * 2)).fetchall()

        # Filter by context relevance (simple keyword match)
        context_words = set(context.lower().split())
        scored = []
        for row in rows:
            episode_words = set(row[1].lower().split())
            overlap = len(context_words & episode_words)
            scored.append((overlap, {
                'id': row[0],
                'content': row[1],
                'context': row[2],
                'timestamp': row[3],
                'importance': row[4],
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        return dot / (norm_a * norm_b + 1e-8)


# =============================================================================
# Unified Memory Manager
# =============================================================================

class MemoryManager:
    """
    Unified memory manager combining short-term and long-term memory.

    Provides a single interface for all memory operations.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        user_id: str = "default",
    ):
        if db_path is None:
            db_path = Path.home() / '.larun' / 'memory' / 'tutor.db'

        self.user_id = user_id
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(db_path)

        self._current_session: Optional[LearningSession] = None

    def start_session(self) -> str:
        """Start a new learning session."""
        session_id = hashlib.sha256(
            f"{self.user_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        self._current_session = LearningSession(
            session_id=session_id,
            user_id=self.user_id,
        )

        # Update user stats
        profile = self.long_term.get_user_profile(self.user_id)
        profile.total_sessions += 1
        self.long_term.save_user_profile(profile)

        return session_id

    def end_session(self) -> None:
        """End the current session."""
        if self._current_session:
            self._current_session.ended_at = datetime.utcnow().isoformat()
            self._current_session.summary = self.short_term.summarize()
            self._current_session.topics_covered = self.short_term.get_topics()

            self.long_term.save_session(self._current_session)
            self._current_session = None

        self.short_term.clear()

    def add_interaction(
        self,
        role: str,
        content: str,
        topics: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        importance: float = 0.5,
    ) -> None:
        """Add an interaction to memory."""
        # Add to short-term
        self.short_term.add(
            role=role,
            content=content,
            topics=topics or [],
            sources=sources or [],
        )

        # Add to current session
        if self._current_session:
            self._current_session.add_turn(
                role=role,
                content=content,
                topics=topics or [],
                sources=sources or [],
            )

        # Add important interactions to episodic memory
        if importance > 0.7 and role == "user":
            context = self.short_term.get_context_string(max_turns=3)
            self.long_term.add_episode(
                user_id=self.user_id,
                content=content,
                context=context,
                importance=importance,
            )

        # Update user stats
        if role == "user":
            self.long_term.update_user_stats(
                user_id=self.user_id,
                questions_asked=1,
                topics=topics,
            )

    def get_context(self, include_history: bool = True) -> Dict[str, Any]:
        """Get current context for the LLM."""
        context = {
            'recent_conversation': self.short_term.get_context_string(),
            'topics': self.short_term.get_topics(),
            'last_question': self.short_term.get_last_question(),
        }

        if include_history:
            # Get relevant episodes from long-term memory
            current_context = context['recent_conversation']
            if current_context:
                episodes = self.long_term.get_relevant_episodes(
                    self.user_id,
                    current_context,
                    limit=3,
                )
                context['relevant_history'] = episodes

            # Get user profile
            profile = self.long_term.get_user_profile(self.user_id)
            context['user_profile'] = {
                'interests': profile.topics_interested[:5],
                'mastered': profile.topics_mastered[:5],
                'difficulty': profile.difficulty_preference,
            }

        return context

    def add_knowledge(self, **kwargs) -> str:
        """Add knowledge to long-term memory."""
        return self.long_term.add_knowledge(**kwargs)

    def search_knowledge(self, **kwargs) -> List[Dict[str, Any]]:
        """Search long-term knowledge."""
        return self.long_term.search_knowledge(**kwargs)

    def get_user_profile(self) -> UserProfile:
        """Get current user profile."""
        return self.long_term.get_user_profile(self.user_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'short_term_turns': len(self.short_term._turns),
            'short_term_topics': self.short_term.get_topics(),
            'knowledge_base': self.long_term.get_knowledge_stats(),
            'user_profile': self.get_user_profile().to_dict(),
        }
