"""
LLM-Powered Space Science Tutor for LARUN
==========================================

RAG-based tutor using knowledge from NASA, ESA, ISRO, JAXA, SpaceX, CNSA,
astronomical papers, and educational content.

Features:
- Retrieval-Augmented Generation for accurate answers
- Multi-source knowledge base (space agencies, papers, textbooks)
- Fine-tuned open model support
- Interactive tutoring sessions
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Document:
    """A document in the knowledge base."""
    id: str
    content: str
    source: str  # nasa, esa, isro, jaxa, spacex, cnsa, arxiv, youtube, textbook
    title: str = ""
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'source': self.source,
            'title': self.title,
            'url': self.url,
            'metadata': self.metadata,
        }


@dataclass
class ChatMessage:
    """A message in a conversation."""
    role: str  # user, assistant, system
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sources: List[Document] = field(default_factory=list)


@dataclass
class TutorResponse:
    """Response from the tutor."""
    answer: str
    sources: List[Document]
    confidence: float = 0.0
    follow_up_questions: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)


# =============================================================================
# LLM Provider Interface
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(
        self,
        model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.embed_model = embed_model
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        try:
            import requests

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

        except Exception as e:
            return f"Error generating response: {e}"

    def embed(self, text: str) -> List[float]:
        try:
            import requests

            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["embedding"]

        except Exception as e:
            # Return zero vector on error
            return [0.0] * 384


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.embed_model = embed_model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating response: {e}"

    def embed(self, text: str) -> List[float]:
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                model=self.embed_model,
                input=text,
            )
            return response.data[0].embedding

        except Exception as e:
            return [0.0] * 1536


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except Exception as e:
            return f"Error generating response: {e}"

    def embed(self, text: str) -> List[float]:
        # Anthropic doesn't have embeddings API, use fallback
        # In production, use a separate embedding service
        return self._simple_embed(text)

    def _simple_embed(self, text: str) -> List[float]:
        """Simple hash-based embedding fallback."""
        # This is a placeholder - use a real embedding model in production
        hash_val = hashlib.md5(text.encode()).hexdigest()
        return [float(int(hash_val[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)] * 24


# =============================================================================
# Vector Store
# =============================================================================

class SimpleVectorStore:
    """
    Simple in-memory vector store for RAG.

    For production, use ChromaDB, Pinecone, Weaviate, or FAISS.
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self.documents: Dict[str, Document] = {}
        self.persist_path = persist_path

        if persist_path and persist_path.exists():
            self._load()

    def add(self, document: Document) -> None:
        """Add a document to the store."""
        self.documents[document.id] = document

    def add_many(self, documents: List[Document]) -> None:
        """Add multiple documents."""
        for doc in documents:
            self.add(doc)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        source_filter: Optional[List[str]] = None,
    ) -> List[Document]:
        """Search for similar documents."""
        if not self.documents:
            return []

        scores = []
        for doc_id, doc in self.documents.items():
            if source_filter and doc.source not in source_filter:
                continue

            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                scores.append((score, doc))

        # Sort by similarity
        scores.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scores[:top_k]]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not HAS_NUMPY:
            # Simple implementation without numpy
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b + 1e-8)

        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-8))

    def _save(self) -> None:
        """Save to disk."""
        if self.persist_path:
            data = {
                doc_id: {
                    'id': doc.id,
                    'content': doc.content,
                    'source': doc.source,
                    'title': doc.title,
                    'url': doc.url,
                    'metadata': doc.metadata,
                    'embedding': doc.embedding,
                }
                for doc_id, doc in self.documents.items()
            }
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'w') as f:
                json.dump(data, f)

    def _load(self) -> None:
        """Load from disk."""
        if self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path) as f:
                    data = json.load(f)
                for doc_id, doc_data in data.items():
                    self.documents[doc_id] = Document(**doc_data)
            except Exception:
                pass


# =============================================================================
# Knowledge Sources
# =============================================================================

class KnowledgeSource(ABC):
    """Abstract base class for knowledge sources."""

    @abstractmethod
    def fetch(self, query: Optional[str] = None, limit: int = 100) -> List[Document]:
        """Fetch documents from the source."""
        pass


class NASASource(KnowledgeSource):
    """NASA data source."""

    NASA_ENDPOINTS = {
        'apod': 'https://api.nasa.gov/planetary/apod',
        'neo': 'https://api.nasa.gov/neo/rest/v1/neo/browse',
        'exoplanet': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync',
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('NASA_API_KEY', 'DEMO_KEY')

    def fetch(self, query: Optional[str] = None, limit: int = 100) -> List[Document]:
        documents = []

        # Add built-in NASA educational content
        documents.extend(self._get_builtin_content())

        return documents

    def _get_builtin_content(self) -> List[Document]:
        """Get built-in NASA educational content."""
        return [
            Document(
                id='nasa-exoplanet-101',
                content="""
                NASA's Exoplanet Exploration Program searches for signs of life and habitable worlds
                beyond our solar system. The program uses multiple detection methods including:

                1. Transit Method: Measures the dimming of starlight when a planet passes in front
                2. Radial Velocity: Detects wobble in stars caused by orbiting planets
                3. Direct Imaging: Photographs exoplanets by blocking out starlight
                4. Microlensing: Uses gravitational lensing to detect distant planets

                Key missions include Kepler, TESS, and the upcoming Nancy Grace Roman Space Telescope.
                """,
                source='nasa',
                title='NASA Exoplanet Program Overview',
                url='https://exoplanets.nasa.gov/',
            ),
            Document(
                id='nasa-tess-mission',
                content="""
                TESS (Transiting Exoplanet Survey Satellite) is NASA's all-sky transit survey mission.
                Launched in 2018, TESS monitors bright, nearby stars for transiting exoplanets.

                Key Facts:
                - Observes stars 30-100 times brighter than Kepler targets
                - Covers 400x more sky than Kepler
                - Focus on Earth and Super-Earth sized planets
                - 2-minute and 20-second cadence observations
                - Has discovered over 400 confirmed planets
                """,
                source='nasa',
                title='TESS Mission Overview',
                url='https://tess.mit.edu/',
            ),
            Document(
                id='nasa-jwst-exoplanets',
                content="""
                The James Webb Space Telescope (JWST) revolutionizes exoplanet science through:

                1. Transmission Spectroscopy: Analyzing starlight filtered through exoplanet atmospheres
                2. Emission Spectroscopy: Detecting thermal emission from hot exoplanets
                3. Direct Imaging: Photographing young, distant exoplanets

                JWST has detected water vapor, CO2, and other molecules in exoplanet atmospheres,
                bringing us closer to identifying potentially habitable worlds.
                """,
                source='nasa',
                title='JWST and Exoplanet Atmospheres',
                url='https://webb.nasa.gov/',
            ),
        ]


class ESASource(KnowledgeSource):
    """European Space Agency data source."""

    def fetch(self, query: Optional[str] = None, limit: int = 100) -> List[Document]:
        return [
            Document(
                id='esa-gaia-mission',
                content="""
                Gaia is ESA's space mission to chart a three-dimensional map of our Galaxy.
                It provides unprecedented positional and radial velocity measurements for over
                1 billion stars in the Milky Way.

                Key contributions to exoplanet science:
                - Precise stellar parameters (distance, temperature, radius)
                - Detection of astrometric wobbles from massive planets
                - Identification of stellar companions and binaries
                - Gaia DR3 includes over 800,000 binary star solutions
                """,
                source='esa',
                title='Gaia Mission and Stellar Characterization',
                url='https://www.cosmos.esa.int/gaia',
            ),
            Document(
                id='esa-cheops-mission',
                content="""
                CHEOPS (CHaracterising ExOPlanet Satellite) is ESA's first exoplanet mission.
                Launched in 2019, it measures precise radii of known exoplanets.

                Science Goals:
                - Precise radius measurements (uncertainty <10%)
                - Identification of planets with significant atmospheres
                - Target selection for spectroscopic follow-up
                - Study of planetary system architectures
                """,
                source='esa',
                title='CHEOPS Exoplanet Characterization',
                url='https://cheops.unibe.ch/',
            ),
        ]


class ISROSource(KnowledgeSource):
    """Indian Space Research Organisation data source."""

    def fetch(self, query: Optional[str] = None, limit: int = 100) -> List[Document]:
        return [
            Document(
                id='isro-astrosat',
                content="""
                AstroSat is India's first dedicated multi-wavelength space telescope, launched in 2015.
                It observes celestial sources in X-ray, optical, and UV spectral bands simultaneously.

                Contributions to stellar and exoplanet science:
                - UV observations of hot stars and stellar activity
                - X-ray monitoring of stellar flares
                - Study of binary star systems
                - Timing observations of variable sources

                AstroSat complements ground-based observations for complete characterization
                of stellar hosts of exoplanets.
                """,
                source='isro',
                title='AstroSat Multi-wavelength Observatory',
                url='https://astrosat.iucaa.in/',
            ),
            Document(
                id='isro-exoworlds',
                content="""
                India's ExoWorlds initiative aims to discover and characterize exoplanets
                using both space-based and ground-based facilities.

                Key capabilities:
                - PRL 1.2m telescope for radial velocity measurements
                - Participation in international exoplanet surveys
                - Development of high-resolution spectrographs
                - Plans for dedicated exoplanet missions
                """,
                source='isro',
                title='Indian Exoplanet Research',
                url='https://www.isro.gov.in/',
            ),
        ]


class JAXASource(KnowledgeSource):
    """Japan Aerospace Exploration Agency data source."""

    def fetch(self, query: Optional[str] = None, limit: int = 100) -> List[Document]:
        return [
            Document(
                id='jaxa-jasmine',
                content="""
                JASMINE (Japan Astrometry Satellite Mission for INfrared Exploration) will
                perform high-precision infrared astrometry to detect exoplanets.

                Mission Goals:
                - Astrometric detection of Earth-like planets around nearby stars
                - Infrared observations to penetrate Galactic dust
                - Precise distance measurements for Galactic structure
                - Expected launch in the late 2020s
                """,
                source='jaxa',
                title='JASMINE Astrometry Mission',
                url='https://www.jasmine.nao.ac.jp/',
            ),
        ]


class SpaceXSource(KnowledgeSource):
    """SpaceX educational content."""

    def fetch(self, query: Optional[str] = None, limit: int = 100) -> List[Document]:
        return [
            Document(
                id='spacex-starship-mars',
                content="""
                SpaceX's Starship is designed to carry humans and cargo to the Moon, Mars, and beyond.
                While primarily a transportation system, Starship enables space science through:

                1. Large payload capacity for space telescopes
                2. On-orbit refueling for deep space missions
                3. Rapid reusability reducing launch costs
                4. Potential for space station construction

                The ability to launch large, powerful telescopes could revolutionize exoplanet
                detection and characterization in the coming decades.
                """,
                source='spacex',
                title='Starship and Space Science',
                url='https://www.spacex.com/vehicles/starship/',
            ),
        ]


class CNSASource(KnowledgeSource):
    """China National Space Administration data source."""

    def fetch(self, query: Optional[str] = None, limit: int = 100) -> List[Document]:
        return [
            Document(
                id='cnsa-earth2',
                content="""
                China's Earth 2.0 (ET) mission is designed to find Earth-like planets in
                habitable zones around Sun-like stars. Planned for launch around 2026.

                Mission Design:
                - 6 transit-detecting telescopes monitoring the Kepler field
                - 1 gravitational microlensing telescope
                - 4-year mission duration
                - Expected to find dozens of Earth-like planets

                This represents China's first dedicated exoplanet mission and will
                significantly expand the catalog of potentially habitable worlds.
                """,
                source='cnsa',
                title='Earth 2.0 Exoplanet Mission',
                url='https://www.cnsa.gov.cn/',
            ),
        ]


class ArxivSource(KnowledgeSource):
    """ArXiv astronomy papers source."""

    def fetch(self, query: Optional[str] = None, limit: int = 100) -> List[Document]:
        # In production, use arxiv API
        return [
            Document(
                id='arxiv-transit-method',
                content="""
                The transit method remains the most prolific technique for exoplanet detection.
                Key developments include:

                - Precision photometry at the parts-per-million level
                - Machine learning for automated transit detection
                - Bayesian methods for false positive rejection
                - Transit timing variations revealing additional planets
                - Phase curve analysis for atmospheric characterization

                Recent advances in data analysis have improved detection sensitivity for
                Earth-sized planets in longer-period orbits.
                """,
                source='arxiv',
                title='Advances in Transit Detection Methods',
            ),
            Document(
                id='arxiv-ml-exoplanets',
                content="""
                Machine learning has revolutionized exoplanet science:

                1. Neural Networks for Transit Detection
                   - CNNs achieve >95% recall on Kepler data
                   - Reduce false positive rates by 10x
                   - Enable real-time detection in TESS data

                2. Classification of Variable Stars
                   - Random forests distinguish planet signals from stellar variability
                   - Deep learning classifies eclipsing binary subtypes

                3. Atmospheric Retrieval
                   - Neural networks accelerate spectral fitting
                   - Ensemble methods provide uncertainty estimates

                TinyML brings these capabilities to edge devices for distributed analysis.
                """,
                source='arxiv',
                title='Machine Learning in Exoplanet Science',
            ),
        ]


# =============================================================================
# Space Science Tutor
# =============================================================================

# System prompt for the space science tutor
SPACE_TUTOR_SYSTEM_PROMPT = """You are LARUN's Space Science Tutor, an expert AI assistant specializing in astronomy, astrophysics, and space exploration. Your knowledge comes from NASA, ESA, ISRO, JAXA, SpaceX, CNSA, and academic sources.

Your role is to:
1. Answer questions about space science clearly and accurately
2. Explain complex concepts in accessible language
3. Provide context from multiple space agencies when relevant
4. Suggest related topics and follow-up questions
5. Correct misconceptions gently but firmly

Guidelines:
- Use analogies to explain complex concepts
- Cite specific missions, discoveries, or papers when possible
- Acknowledge uncertainty and ongoing research
- Encourage curiosity and further exploration
- Keep responses focused and informative

When using provided context:
- Prioritize information from the context
- Synthesize information from multiple sources
- Note when your knowledge extends beyond the context
- Be clear about what is well-established vs. cutting-edge research"""


class SpaceScienceTutor:
    """
    RAG-powered Space Science Tutor.

    Uses knowledge from multiple space agencies and astronomical sources
    to provide accurate, educational responses about space science.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        knowledge_dir: Optional[Path] = None,
    ):
        """
        Initialize the tutor.

        Args:
            llm_provider: LLM provider (defaults to Ollama)
            knowledge_dir: Path to knowledge base directory
        """
        self.llm = llm_provider or OllamaProvider()

        if knowledge_dir is None:
            knowledge_dir = Path(__file__).parent.parent / 'knowledge'
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        self.vector_store = SimpleVectorStore(
            persist_path=knowledge_dir / 'vectors.json'
        )

        self.conversation_history: List[ChatMessage] = []
        self._initialized = False

    def initialize_knowledge(self) -> None:
        """Load knowledge from all sources."""
        if self._initialized:
            return

        sources = [
            NASASource(),
            ESASource(),
            ISROSource(),
            JAXASource(),
            SpaceXSource(),
            CNSASource(),
            ArxivSource(),
        ]

        all_docs = []
        for source in sources:
            docs = source.fetch()
            all_docs.extend(docs)

        # Generate embeddings
        for doc in all_docs:
            doc.embedding = self.llm.embed(doc.content)

        self.vector_store.add_many(all_docs)
        self._initialized = True

    def ask(
        self,
        question: str,
        source_filter: Optional[List[str]] = None,
        include_sources: bool = True,
    ) -> TutorResponse:
        """
        Ask a question to the tutor.

        Args:
            question: User's question
            source_filter: Only use these sources (e.g., ['nasa', 'esa'])
            include_sources: Whether to include source documents

        Returns:
            TutorResponse with answer and sources
        """
        # Initialize knowledge if needed
        if not self._initialized:
            self.initialize_knowledge()

        # Embed the question
        query_embedding = self.llm.embed(question)

        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(
            query_embedding,
            top_k=5,
            source_filter=source_filter,
        )

        # Build context from documents
        context = self._build_context(relevant_docs)

        # Create prompt with RAG context
        prompt = self._build_prompt(question, context)

        # Generate response
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=SPACE_TUTOR_SYSTEM_PROMPT,
            temperature=0.7,
        )

        # Update conversation history
        self.conversation_history.append(ChatMessage(
            role='user',
            content=question,
        ))
        self.conversation_history.append(ChatMessage(
            role='assistant',
            content=answer,
            sources=relevant_docs,
        ))

        # Generate follow-up questions
        follow_ups = self._generate_follow_ups(question, answer)

        return TutorResponse(
            answer=answer,
            sources=relevant_docs if include_sources else [],
            confidence=self._estimate_confidence(relevant_docs),
            follow_up_questions=follow_ups,
            related_topics=self._extract_topics(relevant_docs),
        )

    def explain_more(self, topic: Optional[str] = None) -> TutorResponse:
        """
        Ask for more explanation on the last topic or a specific topic.
        """
        if not self.conversation_history:
            return TutorResponse(
                answer="No previous conversation to explain further. Please ask a question first.",
                sources=[],
            )

        last_answer = self.conversation_history[-1].content

        if topic:
            question = f"Please explain more about {topic} in the context of our discussion about: {last_answer[:200]}"
        else:
            question = f"Can you explain this in more detail and simpler terms? {last_answer[:200]}"

        return self.ask(question)

    def quiz_me(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a quiz question on a topic.
        """
        if topic:
            prompt = f"Generate a multiple choice quiz question about {topic} in space science."
        else:
            prompt = "Generate a multiple choice quiz question about an interesting space science topic."

        prompt += """

Format your response as:
QUESTION: [Your question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
CORRECT: [Letter]
EXPLANATION: [Brief explanation]"""

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=SPACE_TUTOR_SYSTEM_PROMPT,
            temperature=0.8,
        )

        # Parse the response
        return {
            'raw_response': response,
            'topic': topic or 'general space science',
        }

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.conversation_history:
            return "No conversation yet."

        topics = []
        for msg in self.conversation_history:
            if msg.role == 'user':
                topics.append(msg.content[:100])

        return f"Topics discussed: {', '.join(topics[:5])}"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from documents."""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents):
            source_label = doc.source.upper()
            context_parts.append(
                f"[Source {i+1}: {source_label} - {doc.title}]\n{doc.content}\n"
            )

        return "\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the full prompt for the LLM."""
        if context:
            return f"""Based on the following information from space agencies and scientific sources:

{context}

Please answer this question: {question}

Provide a clear, educational response that cites relevant sources when appropriate."""
        else:
            return question

    def _generate_follow_ups(self, question: str, answer: str) -> List[str]:
        """Generate follow-up questions."""
        # Simple keyword-based follow-ups
        follow_ups = []

        keywords = {
            'exoplanet': "How do scientists detect exoplanet atmospheres?",
            'transit': "What is limb darkening and how does it affect transits?",
            'kepler': "How did Kepler's mission end and what comes next?",
            'tess': "What types of stars does TESS observe?",
            'habitable': "What factors determine if a planet is habitable?",
            'jwst': "What can JWST detect that other telescopes cannot?",
        }

        question_lower = question.lower()
        for keyword, follow_up in keywords.items():
            if keyword in question_lower and len(follow_ups) < 3:
                follow_ups.append(follow_up)

        return follow_ups

    def _extract_topics(self, documents: List[Document]) -> List[str]:
        """Extract related topics from documents."""
        topics = set()
        for doc in documents:
            if 'exoplanet' in doc.content.lower():
                topics.add('Exoplanets')
            if 'transit' in doc.content.lower():
                topics.add('Transit Detection')
            if 'spectroscopy' in doc.content.lower():
                topics.add('Spectroscopy')
            if 'atmosphere' in doc.content.lower():
                topics.add('Planetary Atmospheres')
            if 'mission' in doc.content.lower():
                topics.add('Space Missions')

        return list(topics)[:5]

    def _estimate_confidence(self, documents: List[Document]) -> float:
        """Estimate confidence based on source quality."""
        if not documents:
            return 0.3

        # More sources = higher confidence
        source_count = len(set(doc.source for doc in documents))
        doc_count = len(documents)

        confidence = min(0.5 + (source_count * 0.1) + (doc_count * 0.05), 0.95)
        return confidence
