"""
Web Crawler for LARUN Space Science Knowledge Base
===================================================

Crawls and indexes content from space agencies, arXiv, YouTube,
and educational sources to build the knowledge base.

Sources:
- NASA (news, missions, educational content)
- ESA (news, missions, science)
- ISRO (missions, achievements)
- JAXA (missions, research)
- SpaceX (updates, missions)
- CNSA (missions, programs)
- arXiv (astronomy papers)
- YouTube (educational videos - transcripts)
"""

import os
import re
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin, urlparse
from abc import ABC, abstractmethod
import threading
from queue import Queue
import logging

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrawlConfig:
    """Configuration for web crawling."""
    max_pages_per_source: int = 100
    request_delay_seconds: float = 1.0
    timeout_seconds: int = 30
    max_content_length: int = 50000
    user_agent: str = "LARUN-SpaceCrawler/1.0 (Educational Research)"
    respect_robots_txt: bool = True
    max_retries: int = 3


@dataclass
class CrawledPage:
    """A crawled web page."""
    url: str
    title: str
    content: str
    source: str
    crawled_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


# =============================================================================
# Base Crawler
# =============================================================================

class BaseCrawler(ABC):
    """Abstract base class for source-specific crawlers."""

    def __init__(self, config: Optional[CrawlConfig] = None):
        self.config = config or CrawlConfig()
        self._session = None

    @property
    def session(self):
        if self._session is None and HAS_REQUESTS:
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': self.config.user_agent,
            })
        return self._session

    @abstractmethod
    def get_seed_urls(self) -> List[str]:
        """Get initial URLs to crawl."""
        pass

    @abstractmethod
    def parse_page(self, url: str, html: str) -> CrawledPage:
        """Parse a page and extract content."""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Get the source identifier."""
        pass

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page's HTML content."""
        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return None

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()

                # Respect content length limit
                if len(response.text) > self.config.max_content_length:
                    return response.text[:self.config.max_content_length]

                return response.text

            except Exception as e:
                logger.warning(f"Failed to fetch {url} (attempt {attempt + 1}): {e}")
                time.sleep(self.config.request_delay_seconds * (attempt + 1))

        return None

    def crawl(self, max_pages: Optional[int] = None) -> Generator[CrawledPage, None, None]:
        """Crawl pages from this source."""
        max_pages = max_pages or self.config.max_pages_per_source
        crawled = set()
        to_crawl = self.get_seed_urls()

        while to_crawl and len(crawled) < max_pages:
            url = to_crawl.pop(0)

            if url in crawled:
                continue

            html = self.fetch_page(url)
            if html:
                try:
                    page = self.parse_page(url, html)
                    if page and page.content:
                        yield page
                        crawled.add(url)

                        # Extract more URLs from page
                        new_urls = self.extract_urls(url, html)
                        for new_url in new_urls:
                            if new_url not in crawled and new_url not in to_crawl:
                                to_crawl.append(new_url)

                except Exception as e:
                    logger.error(f"Failed to parse {url}: {e}")

            time.sleep(self.config.request_delay_seconds)

    def extract_urls(self, base_url: str, html: str) -> List[str]:
        """Extract URLs from a page."""
        if not HAS_BS4:
            return []

        urls = []
        soup = BeautifulSoup(html, 'html.parser')

        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)

            # Only follow links to same domain
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                urls.append(full_url)

        return urls[:20]  # Limit URL extraction


# =============================================================================
# Source-Specific Crawlers
# =============================================================================

class NASACrawler(BaseCrawler):
    """Crawler for NASA content."""

    def get_source_name(self) -> str:
        return "nasa"

    def get_seed_urls(self) -> List[str]:
        return [
            "https://science.nasa.gov/exoplanets/",
            "https://exoplanets.nasa.gov/news/",
            "https://www.nasa.gov/missions/",
            "https://science.nasa.gov/astrophysics/",
        ]

    def parse_page(self, url: str, html: str) -> CrawledPage:
        if not HAS_BS4:
            return CrawledPage(url=url, title="", content="", source=self.get_source_name())

        soup = BeautifulSoup(html, 'html.parser')

        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()

        # Extract main content
        content_parts = []

        # Try common content containers
        for selector in ['article', 'main', '.content', '#content']:
            container = soup.select_one(selector)
            if container:
                for p in container.find_all(['p', 'h1', 'h2', 'h3', 'li']):
                    text = p.get_text().strip()
                    if text and len(text) > 20:
                        content_parts.append(text)
                break

        content = "\n".join(content_parts)

        return CrawledPage(
            url=url,
            title=title,
            content=content,
            source=self.get_source_name(),
            metadata={'domain': 'nasa.gov'},
        )


class ESACrawler(BaseCrawler):
    """Crawler for ESA content."""

    def get_source_name(self) -> str:
        return "esa"

    def get_seed_urls(self) -> List[str]:
        return [
            "https://www.esa.int/Science_Exploration/Space_Science/Exoplanets",
            "https://www.cosmos.esa.int/web/gaia",
            "https://sci.esa.int/web/cheops",
        ]

    def parse_page(self, url: str, html: str) -> CrawledPage:
        if not HAS_BS4:
            return CrawledPage(url=url, title="", content="", source=self.get_source_name())

        soup = BeautifulSoup(html, 'html.parser')

        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()

        content_parts = []
        for p in soup.find_all(['p', 'h1', 'h2', 'h3']):
            text = p.get_text().strip()
            if text and len(text) > 20:
                content_parts.append(text)

        return CrawledPage(
            url=url,
            title=title,
            content="\n".join(content_parts[:50]),
            source=self.get_source_name(),
        )


class ArxivCrawler(BaseCrawler):
    """Crawler for arXiv astronomy papers."""

    ARXIV_API = "http://export.arxiv.org/api/query"

    def get_source_name(self) -> str:
        return "arxiv"

    def get_seed_urls(self) -> List[str]:
        # ArXiv uses API, not traditional crawling
        return []

    def parse_page(self, url: str, html: str) -> CrawledPage:
        # Handled by search_papers
        return CrawledPage(url=url, title="", content="", source=self.get_source_name())

    def search_papers(
        self,
        query: str = "exoplanet",
        max_results: int = 50,
    ) -> Generator[CrawledPage, None, None]:
        """Search arXiv for papers."""
        if not HAS_REQUESTS:
            return

        categories = ["astro-ph.EP", "astro-ph.SR", "astro-ph.IM"]
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)

        params = {
            "search_query": f"({cat_query}) AND ({query})",
            "start": 0,
            "max_results": max_results,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending",
        }

        try:
            response = self.session.get(self.ARXIV_API, params=params, timeout=30)
            response.raise_for_status()

            # Parse Atom feed
            if HAS_BS4:
                soup = BeautifulSoup(response.text, 'xml')

                for entry in soup.find_all('entry'):
                    title = entry.find('title').get_text().strip() if entry.find('title') else ""
                    abstract = entry.find('summary').get_text().strip() if entry.find('summary') else ""
                    arxiv_id = entry.find('id').get_text().strip() if entry.find('id') else ""

                    if title and abstract:
                        yield CrawledPage(
                            url=arxiv_id,
                            title=title,
                            content=f"{title}\n\n{abstract}",
                            source=self.get_source_name(),
                            metadata={'arxiv_id': arxiv_id},
                        )

        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")

    def crawl(self, max_pages: Optional[int] = None) -> Generator[CrawledPage, None, None]:
        """Override to use API-based search."""
        queries = [
            "exoplanet transit detection",
            "variable star classification",
            "stellar flare",
            "gravitational microlensing",
            "galaxy morphology machine learning",
        ]

        for query in queries:
            yield from self.search_papers(query, max_results=10)
            time.sleep(3)  # Respect arXiv rate limits


class YouTubeCrawler(BaseCrawler):
    """
    Crawler for YouTube educational videos.

    Note: Requires YouTube Data API key for full functionality.
    Falls back to curated content list without API.
    """

    def get_source_name(self) -> str:
        return "youtube"

    def get_seed_urls(self) -> List[str]:
        return []

    def parse_page(self, url: str, html: str) -> CrawledPage:
        return CrawledPage(url=url, title="", content="", source=self.get_source_name())

    def get_curated_content(self) -> List[CrawledPage]:
        """Get curated educational video content summaries."""
        # In production, use YouTube API to fetch transcripts
        # For now, provide curated summaries of popular space science channels

        content = [
            CrawledPage(
                url="https://youtube.com/NASA",
                title="NASA Official Channel - Exoplanet Content",
                content="""
                NASA's YouTube channel provides educational content about exoplanet exploration:

                Key Topics Covered:
                - Kepler and TESS mission discoveries
                - How scientists find exoplanets
                - Habitable zone explanations
                - James Webb Space Telescope exoplanet observations
                - Artist concepts of alien worlds

                Notable Series:
                - "NASA Explorers" featuring exoplanet scientists
                - Mission briefings and launch coverage
                - Educational animations explaining detection methods
                """,
                source=self.get_source_name(),
                metadata={'channel': 'NASA'},
            ),
            CrawledPage(
                url="https://youtube.com/SpaceX",
                title="SpaceX - Launch and Mission Content",
                content="""
                SpaceX's channel documents rocket launches and space technology:

                Topics:
                - Starship development for Mars missions
                - Satellite deployment missions
                - Crew Dragon missions to ISS
                - Reusable rocket technology

                Educational Value:
                - Engineering explanations
                - Mission profiles and orbital mechanics
                - Future of human spaceflight
                """,
                source=self.get_source_name(),
                metadata={'channel': 'SpaceX'},
            ),
            CrawledPage(
                url="https://youtube.com/DrBecky",
                title="Dr. Becky - Astrophysics Education",
                content="""
                Dr. Becky Smethurst is an astrophysicist creating educational content:

                Topics Covered:
                - Latest exoplanet discoveries explained
                - Black holes and galaxy formation
                - How astronomical research works
                - Paper reviews and science news

                Style: Accessible explanations of complex topics with enthusiasm
                and accuracy. Great for understanding current research.
                """,
                source=self.get_source_name(),
                metadata={'channel': 'DrBecky'},
            ),
        ]

        return content

    def crawl(self, max_pages: Optional[int] = None) -> Generator[CrawledPage, None, None]:
        """Return curated content."""
        for page in self.get_curated_content():
            yield page


# =============================================================================
# Knowledge Indexer
# =============================================================================

class KnowledgeIndexer:
    """
    Orchestrates crawling and indexing of space science knowledge.

    Manages multiple crawlers and stores results in the memory system.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config: Optional[CrawlConfig] = None,
    ):
        if output_dir is None:
            output_dir = Path.home() / '.larun' / 'knowledge'
        output_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
        self.config = config or CrawlConfig()

        self.crawlers = {
            'nasa': NASACrawler(self.config),
            'esa': ESACrawler(self.config),
            'arxiv': ArxivCrawler(self.config),
            'youtube': YouTubeCrawler(self.config),
        }

        self.index_file = output_dir / 'index.json'
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load the content index."""
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {'pages': {}, 'stats': {}, 'last_crawl': None}

    def _save_index(self) -> None:
        """Save the content index."""
        with open(self.index_file, 'w') as f:
            json.dump(self._index, f, indent=2)

    def crawl_source(
        self,
        source: str,
        max_pages: Optional[int] = None,
        embed_fn: Optional[callable] = None,
    ) -> int:
        """
        Crawl a specific source.

        Args:
            source: Source name (nasa, esa, arxiv, youtube)
            max_pages: Maximum pages to crawl
            embed_fn: Optional function to generate embeddings

        Returns:
            Number of pages indexed
        """
        if source not in self.crawlers:
            logger.error(f"Unknown source: {source}")
            return 0

        crawler = self.crawlers[source]
        count = 0

        logger.info(f"Starting crawl of {source}...")

        for page in crawler.crawl(max_pages):
            # Skip if already indexed
            if page.content_hash in self._index['pages']:
                continue

            # Save page content
            page_file = self.output_dir / source / f"{page.content_hash}.json"
            page_file.parent.mkdir(parents=True, exist_ok=True)

            page_data = {
                'url': page.url,
                'title': page.title,
                'content': page.content,
                'source': page.source,
                'crawled_at': page.crawled_at,
                'metadata': page.metadata,
            }

            # Add embedding if function provided
            if embed_fn:
                try:
                    page_data['embedding'] = embed_fn(page.content)
                except Exception as e:
                    logger.warning(f"Failed to embed: {e}")

            with open(page_file, 'w') as f:
                json.dump(page_data, f)

            # Update index
            self._index['pages'][page.content_hash] = {
                'file': str(page_file.relative_to(self.output_dir)),
                'source': source,
                'title': page.title,
                'url': page.url,
            }

            count += 1
            logger.info(f"Indexed: {page.title[:50]}...")

        # Update stats
        self._index['stats'][source] = self._index['stats'].get(source, 0) + count
        self._index['last_crawl'] = datetime.utcnow().isoformat()
        self._save_index()

        logger.info(f"Crawled {count} new pages from {source}")
        return count

    def crawl_all(
        self,
        max_pages_per_source: Optional[int] = None,
        embed_fn: Optional[callable] = None,
    ) -> Dict[str, int]:
        """Crawl all sources."""
        results = {}

        for source in self.crawlers:
            results[source] = self.crawl_source(
                source,
                max_pages_per_source,
                embed_fn,
            )

        return results

    def search(
        self,
        query: str,
        source_filter: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search indexed content.

        Simple keyword search - use embeddings for better results.
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for content_hash, meta in self._index['pages'].items():
            if source_filter and meta['source'] not in source_filter:
                continue

            # Load page content
            page_file = self.output_dir / meta['file']
            if not page_file.exists():
                continue

            try:
                with open(page_file) as f:
                    page_data = json.load(f)

                content_lower = page_data['content'].lower()
                title_lower = page_data['title'].lower()

                # Score by keyword matches
                content_words = set(content_lower.split())
                matches = len(query_words & content_words)
                title_match = any(w in title_lower for w in query_words)

                if matches > 0 or title_match:
                    score = matches + (10 if title_match else 0)
                    results.append({
                        'score': score,
                        'id': content_hash,
                        **page_data,
                    })

            except Exception:
                continue

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        return {
            'total_pages': len(self._index['pages']),
            'by_source': self._index['stats'],
            'last_crawl': self._index['last_crawl'],
        }
