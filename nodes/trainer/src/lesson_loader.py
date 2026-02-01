"""
Lesson Loader for LARUN Educational Trainer
============================================

Loads and parses markdown-based lessons for the educational system.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re


@dataclass
class Lesson:
    """Represents a single lesson."""
    id: str
    title: str
    difficulty: str = "beginner"
    estimated_time_min: int = 15
    content: str = ""
    sections: List[Dict[str, str]] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    further_reading: List[Dict[str, str]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'difficulty': self.difficulty,
            'estimated_time_min': self.estimated_time_min,
            'content': self.content,
            'sections': self.sections,
            'key_points': self.key_points,
            'further_reading': self.further_reading,
            'prerequisites': self.prerequisites,
        }


class LessonLoader:
    """
    Loads lessons from markdown files.

    Lesson format:
    ```markdown
    ---
    title: Lesson Title
    difficulty: beginner|intermediate|advanced
    time: 15
    ---

    # Section 1
    Content...

    # Section 2
    Content...

    ## Key Points
    - Point 1
    - Point 2

    ## Further Reading
    - [Title](url)
    ```
    """

    def __init__(self, lessons_dir: Path):
        """
        Initialize the lesson loader.

        Args:
            lessons_dir: Path to lessons directory
        """
        self.lessons_dir = lessons_dir
        self._lessons_cache: Dict[str, Lesson] = {}

    def list_lessons(self) -> List[Lesson]:
        """
        List all available lessons.

        Returns:
            List of Lesson objects with metadata (content not loaded)
        """
        lessons = []

        if not self.lessons_dir.exists():
            # Return built-in lessons if directory doesn't exist
            return self._get_builtin_lessons()

        for lesson_file in sorted(self.lessons_dir.glob('*.md')):
            lesson_id = lesson_file.stem
            lesson = self._parse_lesson_metadata(lesson_file)
            if lesson:
                lessons.append(lesson)

        # Add built-in lessons if no files found
        if not lessons:
            lessons = self._get_builtin_lessons()

        return lessons

    def load_lesson(self, lesson_id: str) -> Optional[Lesson]:
        """
        Load a specific lesson by ID.

        Args:
            lesson_id: Lesson identifier

        Returns:
            Lesson object or None if not found
        """
        # Check cache
        if lesson_id in self._lessons_cache:
            return self._lessons_cache[lesson_id]

        # Try to load from file
        lesson_file = self.lessons_dir / f"{lesson_id}.md"

        if lesson_file.exists():
            lesson = self._parse_lesson_file(lesson_file)
            if lesson:
                self._lessons_cache[lesson_id] = lesson
                return lesson

        # Try built-in lessons
        builtin = self._get_builtin_lesson(lesson_id)
        if builtin:
            self._lessons_cache[lesson_id] = builtin
            return builtin

        return None

    def _parse_lesson_metadata(self, file_path: Path) -> Optional[Lesson]:
        """Parse only lesson metadata without full content."""
        try:
            content = file_path.read_text(encoding='utf-8')
            return self._parse_lesson_content(file_path.stem, content, metadata_only=True)
        except Exception:
            return None

    def _parse_lesson_file(self, file_path: Path) -> Optional[Lesson]:
        """Parse a complete lesson file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            return self._parse_lesson_content(file_path.stem, content, metadata_only=False)
        except Exception:
            return None

    def _parse_lesson_content(
        self,
        lesson_id: str,
        content: str,
        metadata_only: bool = False,
    ) -> Optional[Lesson]:
        """Parse lesson content from markdown."""
        lesson = Lesson(id=lesson_id, title=lesson_id.replace('_', ' ').title())

        # Parse YAML front matter
        front_matter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)

        if front_matter_match:
            front_matter = front_matter_match.group(1)

            for line in front_matter.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key == 'title':
                        lesson.title = value
                    elif key == 'difficulty':
                        lesson.difficulty = value
                    elif key in ('time', 'estimated_time'):
                        try:
                            lesson.estimated_time_min = int(value)
                        except ValueError:
                            pass

            # Remove front matter from content
            content = content[front_matter_match.end():]

        if metadata_only:
            return lesson

        # Parse content
        lesson.content = content.strip()

        # Extract sections
        sections = re.split(r'\n#\s+', content)
        for section in sections[1:]:  # Skip content before first header
            lines = section.split('\n', 1)
            if len(lines) >= 2:
                lesson.sections.append({
                    'title': lines[0].strip(),
                    'content': lines[1].strip(),
                })

        # Extract key points
        key_points_match = re.search(
            r'##\s*Key\s*Points?\s*\n((?:-\s*.+\n?)+)',
            content,
            re.IGNORECASE,
        )
        if key_points_match:
            points_text = key_points_match.group(1)
            lesson.key_points = [
                line.strip('- ').strip()
                for line in points_text.split('\n')
                if line.strip().startswith('-')
            ]

        # Extract further reading
        reading_match = re.search(
            r'##\s*Further\s*Reading\s*\n((?:-\s*.+\n?)+)',
            content,
            re.IGNORECASE,
        )
        if reading_match:
            reading_text = reading_match.group(1)
            for line in reading_text.split('\n'):
                link_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', line)
                if link_match:
                    lesson.further_reading.append({
                        'title': link_match.group(1),
                        'url': link_match.group(2),
                    })

        return lesson

    def _get_builtin_lessons(self) -> List[Lesson]:
        """Get list of built-in lessons."""
        return [
            Lesson(
                id='exoplanets_intro',
                title='Introduction to Exoplanets',
                difficulty='beginner',
                estimated_time_min=15,
            ),
            Lesson(
                id='transit_method',
                title='The Transit Method',
                difficulty='beginner',
                estimated_time_min=20,
            ),
            Lesson(
                id='light_curves',
                title='Understanding Light Curves',
                difficulty='intermediate',
                estimated_time_min=25,
            ),
            Lesson(
                id='variable_stars',
                title='Types of Variable Stars',
                difficulty='intermediate',
                estimated_time_min=20,
            ),
            Lesson(
                id='stellar_flares',
                title='Stellar Flares and Activity',
                difficulty='intermediate',
                estimated_time_min=15,
            ),
        ]

    def _get_builtin_lesson(self, lesson_id: str) -> Optional[Lesson]:
        """Get a specific built-in lesson with full content."""
        builtin_content = {
            'exoplanets_intro': self._lesson_exoplanets_intro(),
            'transit_method': self._lesson_transit_method(),
            'light_curves': self._lesson_light_curves(),
            'variable_stars': self._lesson_variable_stars(),
            'stellar_flares': self._lesson_stellar_flares(),
        }

        return builtin_content.get(lesson_id)

    # =========================================================================
    # Built-in Lesson Content
    # =========================================================================

    def _lesson_exoplanets_intro(self) -> Lesson:
        return Lesson(
            id='exoplanets_intro',
            title='Introduction to Exoplanets',
            difficulty='beginner',
            estimated_time_min=15,
            content="""
# What is an Exoplanet?

An **exoplanet** (extrasolar planet) is a planet that orbits a star other than our Sun.
The first confirmed exoplanet around a Sun-like star was discovered in 1995, and since
then we've found over 5,000 confirmed exoplanets!

## Why Study Exoplanets?

1. **Search for Life**: Some exoplanets may have conditions suitable for life
2. **Understand Planet Formation**: Learn how planetary systems form and evolve
3. **Find Earth-like Worlds**: Discover planets similar to our own
4. **Test Physics**: Exoplanets provide laboratories for testing theories

## Types of Exoplanets

- **Hot Jupiters**: Gas giants orbiting very close to their stars
- **Super-Earths**: Rocky planets larger than Earth but smaller than Neptune
- **Mini-Neptunes**: Small gas-rich planets
- **Earth-like**: Rocky planets in the habitable zone

## Detection Methods

There are several ways to detect exoplanets:

1. **Transit Method**: Watch for the planet passing in front of its star
2. **Radial Velocity**: Measure the star's wobble due to the planet's gravity
3. **Direct Imaging**: Take pictures of the planet (very difficult!)
4. **Microlensing**: Use gravitational lensing effects

LARUN uses the **Transit Method** for exoplanet detection.
            """,
            sections=[
                {'title': 'What is an Exoplanet?', 'content': 'Definition and overview'},
                {'title': 'Why Study Exoplanets?', 'content': 'Scientific motivations'},
                {'title': 'Types of Exoplanets', 'content': 'Classification of planets'},
                {'title': 'Detection Methods', 'content': 'How we find exoplanets'},
            ],
            key_points=[
                'Exoplanets orbit stars other than our Sun',
                'Over 5,000 exoplanets have been discovered',
                'The transit method detects planets by light dips',
                'Different planet types include Hot Jupiters and Super-Earths',
            ],
            further_reading=[
                {'title': 'NASA Exoplanet Archive', 'url': 'https://exoplanetarchive.ipac.caltech.edu/'},
                {'title': 'Exoplanet Exploration', 'url': 'https://exoplanets.nasa.gov/'},
            ],
        )

    def _lesson_transit_method(self) -> Lesson:
        return Lesson(
            id='transit_method',
            title='The Transit Method',
            difficulty='beginner',
            estimated_time_min=20,
            content="""
# The Transit Method

The **transit method** is the most successful technique for finding exoplanets.
It works by detecting the tiny dimming that occurs when a planet passes in
front of its host star.

## How It Works

1. **Observation**: Continuously monitor a star's brightness
2. **Detection**: Look for periodic dips in brightness
3. **Confirmation**: Verify the signal is from a planet, not other sources
4. **Characterization**: Measure planet size and orbit

## The Light Curve

A **light curve** is a graph of brightness vs. time. When a planet transits:

```
Brightness
    │
1.0 ├──────┐      ┌──────
    │      │      │
0.99├      └──────┘
    │
    └────────────────────▶ Time
           Transit
```

## What We Can Learn

From a transit, we can determine:

- **Planet Size**: Larger planets block more light
- **Orbital Period**: Time between transits
- **Orbital Distance**: From Kepler's laws
- **Inclination**: Angle of the orbit

## Transit Depth

The **transit depth** tells us the planet size:

```
depth = (R_planet / R_star)²
```

For example:
- Jupiter transiting the Sun: ~1% depth
- Earth transiting the Sun: ~0.01% depth

## Missions Using Transit Method

- **Kepler** (2009-2018): Found 2,700+ confirmed planets
- **TESS** (2018-present): Surveying bright, nearby stars
- **CHEOPS** (2019-present): Precise radius measurements
            """,
            key_points=[
                'Transits cause periodic brightness dips',
                'Transit depth reveals planet size relative to star',
                'Kepler and TESS are major transit-finding missions',
                'We can determine period, size, and orbit from transits',
            ],
            further_reading=[
                {'title': 'TESS Mission', 'url': 'https://tess.mit.edu/'},
                {'title': 'Kepler Mission', 'url': 'https://www.nasa.gov/kepler'},
            ],
        )

    def _lesson_light_curves(self) -> Lesson:
        return Lesson(
            id='light_curves',
            title='Understanding Light Curves',
            difficulty='intermediate',
            estimated_time_min=25,
            content="""
# Understanding Light Curves

A **light curve** is a fundamental tool in astronomy showing how an object's
brightness changes over time. LARUN analyzes light curves to detect various
astronomical phenomena.

## Anatomy of a Transit Light Curve

```
        Ingress      Transit     Egress
           │            │           │
           ▼            ▼           ▼
    ─────┐   ┌──────────────────┐   ┌─────
         │   │                  │   │
          ╲ │                    │ ╱
           ╲│                    │╱
            └────────────────────┘
            ◄──── Duration ─────►
```

### Key Features:
- **Ingress**: Planet starts crossing the star
- **Flat bottom**: Planet fully in front of star
- **Egress**: Planet exits the stellar disk
- **Duration**: Total time of transit

## Light Curve Characteristics

### Transit Depth
```
depth = ΔF/F = (R_p/R_*)²
```
- Tells us the planet-to-star radius ratio

### Transit Duration
```
T ≈ 13 hours × (P/year)^(1/3) × (R_*/R_sun)
```
- Depends on orbital period and star size

### Contact Points
- **T1**: First contact (ingress begins)
- **T2**: Second contact (ingress ends)
- **T3**: Third contact (egress begins)
- **T4**: Fourth contact (egress ends)

## Common Light Curve Shapes

1. **Box-shaped**: Uniform stellar disk
2. **Rounded**: Limb darkening effect
3. **V-shaped**: Grazing transit
4. **Asymmetric**: Eccentric orbit or starspots

## Signal-to-Noise

The transit depth must exceed the noise level:

```
SNR = depth × √(number of transits) / noise
```

Typically need SNR > 7 for detection.
            """,
            key_points=[
                'Light curves show brightness vs time',
                'Transit depth = (planet radius / star radius)²',
                'Duration depends on orbital period and stellar size',
                'SNR > 7 typically required for detection',
            ],
        )

    def _lesson_variable_stars(self) -> Lesson:
        return Lesson(
            id='variable_stars',
            title='Types of Variable Stars',
            difficulty='intermediate',
            estimated_time_min=20,
            content="""
# Types of Variable Stars

**Variable stars** are stars whose brightness changes over time. Understanding
variable stars helps us distinguish them from exoplanet transits and study
stellar physics.

## Classification

### Intrinsic Variables
The star itself changes brightness.

#### Pulsating Variables
- **Cepheids**: Regular pulsations, period-luminosity relation
- **RR Lyrae**: Short period (~0.5 days), used as distance indicators
- **Delta Scuti**: Short periods (hours), low amplitude
- **Mira Variables**: Long periods (months), large amplitude

#### Eruptive Variables
- **Flare Stars**: Sudden brightness increases
- **T Tauri Stars**: Young, pre-main sequence stars
- **R Coronae Borealis**: Sudden fading events

### Extrinsic Variables
Brightness changes due to external factors.

#### Eclipsing Binaries
- **Algol-type (EA)**: Distinct eclipses, constant between
- **Beta Lyrae (EB)**: Continuous light variation
- **W UMa (EW)**: Contact binaries, short periods

#### Rotating Variables
- **BY Draconis**: Starspot variations
- **Ellipsoidal**: Tidally distorted stars

## Why It Matters for LARUN

Variable stars can mimic or mask exoplanet signals:
- Eclipsing binaries look like deep transits
- Starspots create similar periodic dips
- Pulsations add noise to light curves

LARUN's VSTAR-001 node classifies variable star types to:
1. Remove false positives from planet searches
2. Study stellar physics
3. Measure distances (Cepheids, RR Lyrae)
            """,
            key_points=[
                'Variable stars change brightness over time',
                'Pulsating variables: Cepheids, RR Lyrae, Delta Scuti',
                'Eclipsing binaries can mimic planet transits',
                'Classification helps remove false positives',
            ],
        )

    def _lesson_stellar_flares(self) -> Lesson:
        return Lesson(
            id='stellar_flares',
            title='Stellar Flares and Activity',
            difficulty='intermediate',
            estimated_time_min=15,
            content="""
# Stellar Flares and Activity

**Stellar flares** are sudden, intense releases of energy from a star's
surface. They're important for understanding stellar physics and
assessing habitability of exoplanets.

## What is a Stellar Flare?

A flare occurs when magnetic field lines reconnect, releasing stored
energy. This causes:
- Sudden brightness increase (minutes to hours)
- X-ray and UV emission
- Particle acceleration

## Flare Characteristics

### Light Curve Shape
```
        Peak
         │
    ─────│─
        ╱│╲
       ╱ │ ╲
      ╱  │  ╲╲
     ╱   │    ╲╲╲
────╱    │       ╲╲────
    ◄───►│◄────────────►
    Rise  │   Decay
```

- **Rise**: Fast (seconds to minutes)
- **Decay**: Slower exponential decay
- **Duration**: Minutes to hours

### Flare Energy Classes

| Class | Energy (ergs) | Example |
|-------|---------------|---------|
| Nanoflare | 10²⁴ | Solar microflares |
| Small | 10²⁶-10²⁸ | Typical solar flares |
| Large | 10²⁹-10³¹ | Major solar flares |
| Superflare | 10³²-10³⁶ | M-dwarf flares |

## Stars That Flare

- **M Dwarfs**: Most active, frequent superflares
- **K Dwarfs**: Moderate activity
- **G Dwarfs**: Like our Sun, occasional flares
- **Young Stars**: More active than old stars

## Importance for Habitability

Flares affect exoplanet habitability:
- UV radiation can sterilize surfaces
- But may also drive prebiotic chemistry
- Atmospheric erosion from particle events

LARUN's FLARE-001 node detects flares to:
1. Characterize stellar activity
2. Assess habitability of orbiting planets
3. Study magnetic field dynamics
            """,
            key_points=[
                'Flares are sudden energy releases from magnetic reconnection',
                'Characteristic fast rise, slow decay light curve',
                'M dwarfs are most active, with frequent superflares',
                'Flare activity affects exoplanet habitability',
            ],
        )
