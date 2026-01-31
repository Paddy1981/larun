#!/usr/bin/env python3
"""
LARUN Chat - Conversational AI for Astronomical Analysis
=========================================================
Natural language interface for interacting with LARUN TinyML.

Usage:
    python larun_chat.py              # Start chat mode
    python larun_chat.py --api        # Start API server mode

Created by: Padmanaban Veeraragavalu (Larun Engineering)
With AI assistance from: Claude (Anthropic)

Project: LARUN TinyML × Astrodata
License: MIT
"""

import sys
import json
import re
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "1.0.0"
CHAT_NAME = "LARUN"
MODEL_PATH = Path("models/real/astro_tinyml.h5")
DATA_PATH = Path("data/real/training_data.npz")

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

# ============================================================================
# CHAT CONTEXT & MEMORY
# ============================================================================

@dataclass
class Message:
    """A single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """Manages chat session state."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    current_target: Optional[str] = None
    last_analysis: Optional[Dict] = None

    def add_message(self, role: str, content: str, metadata: Dict = None):
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        return msg

    def get_history(self, limit: int = 10) -> List[Message]:
        return self.messages[-limit:]

    def clear(self):
        self.messages.clear()
        self.context.clear()


# ============================================================================
# INTENT RECOGNITION
# ============================================================================

class IntentRecognizer:
    """Recognize user intents from natural language."""

    INTENTS = {
        'greeting': {
            'patterns': [
                r'\b(hi|hello|hey|howdy|greetings)\b',
                r'\bgood\s+(morning|afternoon|evening)\b',
                r"^(what'?s up|sup)\b"
            ],
            'priority': 1
        },
        'farewell': {
            'patterns': [
                r'\b(bye|goodbye|see you|farewell|quit|exit)\b',
                r'\btalk later\b'
            ],
            'priority': 1
        },
        'help': {
            'patterns': [
                r'\b(help|assist|guide|how do i|what can you)\b',
                r'\bwhat are your (capabilities|features|skills)\b'
            ],
            'priority': 2
        },
        'fetch_data': {
            'patterns': [
                r'\b(fetch|get|download|retrieve|pull)\b.*\b(data|lightcurve|light curve)\b',
                r'\bdata (for|from|of)\b',
                r'\bshow me.*light\s*curve\b',
                r'\bfind.*\b(star|planet|tic|kepler)\b'
            ],
            'priority': 3
        },
        'analyze': {
            'patterns': [
                r'\b(analyze|analysis|examine|study|investigate)\b',
                r'\bwhat (does|do) .* (show|indicate|mean)\b',
                r'\bcan you (look at|check|review)\b'
            ],
            'priority': 3
        },
        'detect_transit': {
            'patterns': [
                r'\b(detect|find|search|look for)\b.*\b(transit|planet|exoplanet)\b',
                r'\btransit (detection|search|analysis)\b',
                r'\bare there.*planets?\b',
                r'\bplanet.*signal\b'
            ],
            'priority': 4
        },
        'detect_anomaly': {
            'patterns': [
                r'\b(anomal|unusual|strange|weird|odd)\b',
                r'\b(detect|find).*\b(anomal|outlier)\b',
                r'\banything (unusual|strange|interesting)\b'
            ],
            'priority': 4
        },
        'train_model': {
            'patterns': [
                r'\b(train|retrain|learn|fit)\b.*\bmodel\b',
                r'\bmodel training\b',
                r'\bimprove (the )?(model|accuracy)\b'
            ],
            'priority': 3
        },
        'generate_report': {
            'patterns': [
                r'\b(generate|create|make|produce)\b.*\breport\b',
                r'\breport (on|for|about)\b',
                r'\bsummar(y|ize)\b'
            ],
            'priority': 3
        },
        'explain': {
            'patterns': [
                r'\b(what is|what are|explain|tell me about|describe)\b',
                r'\bwhat does .* mean\b',
                r'\bhow (does|do) .* work\b'
            ],
            'priority': 2
        },
        'status': {
            'patterns': [
                r'\b(status|state|current|progress)\b',
                r'\bwhat.*going on\b',
                r'\bwhere are we\b'
            ],
            'priority': 2
        },
        'list_skills': {
            'patterns': [
                r'\b(list|show|display)\b.*\b(skills|capabilities|features)\b',
                r'\bwhat can you do\b',
                r'\bavailable (skills|features|options)\b'
            ],
            'priority': 2
        },
        'target_star': {
            'patterns': [
                r'\b(tic|toi|kepler|koi|kic|epic)\s*[\d\-]+\b',
                r'\btarget\s+\w+',
                r'\bstar\s+(named?\s+)?\w+'
            ],
            'priority': 5
        },
        'habitability': {
            'patterns': [
                r'\bhabitab(le|ility)\b',
                r'\blife|living\b',
                r'\bearth.?like\b',
                r'\bgoldilocks\b'
            ],
            'priority': 4
        },
        'thanks': {
            'patterns': [
                r'\b(thanks|thank you|thx|ty|appreciate)\b'
            ],
            'priority': 1
        },
        'affirmative': {
            'patterns': [
                r'^(yes|yeah|yep|sure|ok|okay|go ahead|do it|proceed)$',
                r'^y$'
            ],
            'priority': 1
        },
        'negative': {
            'patterns': [
                r'^(no|nope|nah|cancel|stop|nevermind)$',
                r'^n$'
            ],
            'priority': 1
        }
    }

    def __init__(self):
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for intent, data in self.INTENTS.items():
            self.compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in data['patterns']
            ]

    def recognize(self, text: str) -> List[Tuple[str, float]]:
        """Recognize intents from text, return sorted by confidence."""
        matches = []
        text_lower = text.lower().strip()

        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    priority = self.INTENTS[intent]['priority']
                    confidence = 0.5 + (priority * 0.1)
                    matches.append((intent, confidence))
                    break

        # Sort by confidence descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def extract_target(self, text: str) -> Optional[str]:
        """Extract a target star/planet name from text."""
        # TIC ID
        match = re.search(r'\b(tic)\s*(\d+)\b', text, re.IGNORECASE)
        if match:
            return f"TIC {match.group(2)}"

        # TOI
        match = re.search(r'\b(toi)\s*([\d\.]+)\b', text, re.IGNORECASE)
        if match:
            return f"TOI {match.group(2)}"

        # Kepler
        match = re.search(r'\b(kepler|koi|kic)\s*([\d\-]+[a-z]?)\b', text, re.IGNORECASE)
        if match:
            prefix = match.group(1).upper()
            if prefix == 'KOI':
                prefix = 'KOI'
            elif prefix == 'KIC':
                prefix = 'KIC'
            else:
                prefix = 'Kepler'
            return f"{prefix}-{match.group(2)}"

        # EPIC (K2)
        match = re.search(r'\b(epic)\s*(\d+)\b', text, re.IGNORECASE)
        if match:
            return f"EPIC {match.group(2)}"

        # Generic star name
        match = re.search(r'\bstar\s+(named?\s+)?([A-Za-z0-9\-\+]+)\b', text, re.IGNORECASE)
        if match:
            return match.group(2)

        return None


# ============================================================================
# RESPONSE GENERATOR
# ============================================================================

class ResponseGenerator:
    """Generate natural language responses."""

    GREETINGS = [
        "Hello! I'm LARUN, your astronomical analysis assistant. How can I help you explore the cosmos today?",
        "Hi there! Ready to discover some exoplanets? What would you like to analyze?",
        "Greetings, fellow space explorer! I'm here to help with astronomical data analysis. What's on your mind?",
        "Hello! LARUN at your service. Would you like to fetch some stellar data or analyze light curves?",
    ]

    FAREWELLS = [
        "Goodbye! May your discoveries be plentiful. Clear skies!",
        "See you later! Keep looking up at the stars!",
        "Farewell, space explorer! Come back when you're ready to find more planets!",
        "Bye! Remember, somewhere out there, a planet is waiting to be discovered!",
    ]

    THANKS_RESPONSES = [
        "You're welcome! Let me know if you need anything else.",
        "Happy to help! Discovering the universe is what I'm here for.",
        "Anytime! There's always more cosmos to explore.",
        "My pleasure! Ready when you are for more analysis.",
    ]

    HELP_RESPONSE = """
I'm LARUN, an AI assistant specialized in astronomical data analysis. Here's what I can help you with:

**Data Retrieval**
- "Fetch light curve data for TIC 307210830"
- "Get Kepler data for star KIC 8191672"
- "Download TESS observations for TOI 700"

**Transit Detection**
- "Search for planet transits in this data"
- "Are there any exoplanet signals?"
- "Detect transits with high SNR"

**Anomaly Detection**
- "Find any unusual patterns in the light curve"
- "Check for anomalies"
- "Is there anything strange in this data?"

**Analysis**
- "Analyze this light curve"
- "What does this data show?"
- "Calculate the orbital period"

**Reports**
- "Generate a report for this candidate"
- "Create a NASA-compatible summary"

**Learning**
- "What is a transit?"
- "Explain the habitable zone"
- "How does light curve analysis work?"

Just ask naturally - I'll understand what you need!
"""

    EXPLANATIONS = {
        'transit': """
A **transit** occurs when a planet passes in front of its host star from our perspective on Earth.
During a transit, the star's brightness dips slightly as the planet blocks some of its light.

Key transit properties:
- **Depth**: How much the brightness drops (tells us planet size)
- **Duration**: How long the transit lasts (related to orbital distance)
- **Period**: Time between consecutive transits (the planet's "year")

For example, Earth causes a ~0.01% dip in the Sun's brightness lasting about 13 hours, every 365 days.
""",
        'light curve': """
A **light curve** is a graph showing how a star's brightness changes over time.

Light curves can reveal:
- **Transiting planets**: Regular, periodic dips
- **Eclipsing binaries**: Two stars orbiting each other
- **Stellar variability**: Pulsations, spots, flares
- **Instrument artifacts**: Systematic noise

TESS and Kepler missions have collected millions of light curves, making planet discovery possible!
""",
        'habitable zone': """
The **habitable zone** (or "Goldilocks zone") is the region around a star where liquid water could exist on a planet's surface.

Factors that affect it:
- **Star's luminosity**: Brighter stars have wider, farther habitable zones
- **Star's temperature**: Cooler stars have closer habitable zones
- **Planet's atmosphere**: Can shift the effective zone

Our Sun's habitable zone extends from about 0.95 to 1.37 AU. Earth sits comfortably at 1 AU!
""",
        'tinyml': """
**TinyML** refers to machine learning models small enough to run on microcontrollers and edge devices.

LARUN's TinyML model:
- **Size**: Less than 100KB
- **Inference**: Under 10ms on embedded devices
- **Application**: Real-time transit detection on spacecraft or remote observatories

This enables on-board processing without needing to send all data back to Earth!
""",
        'snr': """
**SNR (Signal-to-Noise Ratio)** measures how strong a signal is compared to background noise.

In transit detection:
- SNR > 7: Generally considered a reliable detection
- SNR > 10: Strong detection, high confidence
- SNR < 5: Weak signal, needs more data or confirmation

Higher SNR means we're more confident the transit is real and not just noise.
"""
    }

    def __init__(self):
        pass

    def greeting(self) -> str:
        return random.choice(self.GREETINGS)

    def farewell(self) -> str:
        return random.choice(self.FAREWELLS)

    def thanks(self) -> str:
        return random.choice(self.THANKS_RESPONSES)

    def help(self) -> str:
        return self.HELP_RESPONSE

    def explain(self, topic: str) -> str:
        """Generate explanation for a topic."""
        topic_lower = topic.lower()

        for key, explanation in self.EXPLANATIONS.items():
            if key in topic_lower:
                return explanation

        return f"I don't have a detailed explanation for '{topic}' yet, but I'm always learning! Try asking about transits, light curves, the habitable zone, TinyML, or SNR."

    def status(self, session: ChatSession) -> str:
        """Generate status response."""
        status_parts = ["Here's the current session status:\n"]

        if session.current_target:
            status_parts.append(f"- **Current target**: {session.current_target}")
        else:
            status_parts.append("- **Current target**: None set")

        if session.last_analysis:
            status_parts.append(f"- **Last analysis**: {session.last_analysis.get('type', 'N/A')}")

        status_parts.append(f"- **Messages in session**: {len(session.messages)}")

        if not session.current_target:
            status_parts.append("\nTry: \"Fetch data for TIC 307210830\" to get started!")

        return "\n".join(status_parts)

    def skills_list(self) -> str:
        """List available skills."""
        return """
**Available Skills (Tier 1 - Active)**

| ID | Name | Description |
|----|------|-------------|
| DATA-001 | NASA Data Ingestion | Fetch from MAST, TESS, Kepler |
| DATA-003 | Light Curve Processing | Normalize, detrend, clean |
| MODEL-001 | Spectral CNN | Train classification model |
| MODEL-002 | TFLite Export | Export for edge devices |
| DETECT-001 | Transit Detection | Find planet signals |
| DETECT-002 | Anomaly Detection | Flag unusual patterns |
| ANAL-005 | SNR Calculator | Signal quality analysis |
| REPORT-001 | Report Generator | NASA-compatible reports |

Say "run skill DATA-001" or just describe what you need naturally!
"""

    def fetch_response(self, target: str, success: bool, details: Dict = None) -> str:
        """Generate response for data fetch operation."""
        if success:
            details = details or {}
            n_points = details.get('n_points', 'unknown')
            source = details.get('source', 'NASA archives')

            return f"""
Successfully fetched light curve data for **{target}**!

- **Data points**: {n_points}
- **Source**: {source}
- **Time span**: {details.get('time_span', 'N/A')} days

What would you like to do next?
- "Analyze this data"
- "Search for transits"
- "Check for anomalies"
"""
        else:
            return f"""
I couldn't find data for **{target}**. This could mean:
- The target ID might be incorrect
- No observations exist for this target
- Network connectivity issues

Try:
- Double-check the target name (e.g., "TIC 307210830", "Kepler-186")
- Use a different target
- Ask me to show known exoplanet hosts
"""

    def transit_detection_response(self, results: Dict) -> str:
        """Generate response for transit detection."""
        n_candidates = results.get('n_candidates', 0)

        if n_candidates == 0:
            return """
No significant transit signals detected in this light curve.

This could mean:
- No transiting planets in this system (or they're too small to detect)
- The orbital plane isn't aligned with our line of sight
- More data might be needed for weak signals

Would you like me to:
- "Check for anomalies" (might find other interesting signals)
- "Try a different target"
- "Explain why transits might be missed"
"""
        else:
            response = f"""
**Transit Detection Results**

Found **{n_candidates}** potential transit signal(s)!

"""
            candidates = results.get('candidates', [])
            for i, c in enumerate(candidates[:5]):  # Show top 5
                response += f"""
**Candidate {i+1}**:
- Transit depth: {c.get('depth', 0)*100:.4f}%
- Duration: {c.get('duration_hours', 0):.2f} hours
- SNR: {c.get('snr', 0):.1f}
"""

            period = results.get('estimated_period')
            if period:
                response += f"\n**Estimated orbital period**: {period:.4f} days\n"

            response += """
What would you like to do?
- "Generate a report for this candidate"
- "Analyze the transit in detail"
- "Check if this is in the habitable zone"
"""
            return response

    def anomaly_response(self, results: Dict) -> str:
        """Generate response for anomaly detection."""
        n_anomalies = results.get('n_anomalies', 0)

        if n_anomalies == 0:
            return "No significant anomalies detected. The light curve appears to be relatively clean and stable."

        response = f"""
**Anomaly Detection Results**

Found **{n_anomalies}** unusual events in the light curve.

"""
        types = results.get('anomaly_types', {})
        if types:
            type_counts = {}
            for t in types.values():
                type_counts[t] = type_counts.get(t, 0) + 1

            response += "**Breakdown by type**:\n"
            for atype, count in type_counts.items():
                response += f"- {atype}: {count}\n"

        response += """
Would you like me to:
- "Explain what these anomalies mean"
- "Generate a detailed report"
- "Check if any could be transits"
"""
        return response

    def dont_understand(self, text: str) -> str:
        """Response when we don't understand."""
        suggestions = [
            "I'm not sure I understood that. Could you rephrase?",
            "Hmm, I didn't quite catch that. Try asking about data fetching, transit detection, or analysis.",
            "I'm specialized in astronomical data analysis. Try asking me to fetch data, detect transits, or explain concepts!",
        ]
        return random.choice(suggestions) + "\n\nType 'help' to see what I can do."


# ============================================================================
# CHAT ENGINE
# ============================================================================

class LarunChat:
    """Main chat engine for LARUN."""

    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.response_generator = ResponseGenerator()
        self.session = ChatSession(session_id=datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.model = None
        self.pending_action = None

    def load_model(self):
        """Load the TinyML model if available."""
        if MODEL_PATH.exists():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(MODEL_PATH))
                return True
            except Exception:
                pass
        return False

    def process_message(self, user_input: str) -> str:
        """Process user message and generate response."""
        # Add user message to history
        self.session.add_message('user', user_input)

        # Handle pending confirmations
        if self.pending_action:
            response = self._handle_pending(user_input)
            self.session.add_message('assistant', response)
            return response

        # Recognize intent
        intents = self.intent_recognizer.recognize(user_input)

        # Extract any target mentioned
        target = self.intent_recognizer.extract_target(user_input)
        if target:
            self.session.current_target = target

        # Generate response based on top intent
        if not intents:
            response = self.response_generator.dont_understand(user_input)
        else:
            top_intent = intents[0][0]
            response = self._handle_intent(top_intent, user_input)

        self.session.add_message('assistant', response)
        return response

    def _handle_intent(self, intent: str, text: str) -> str:
        """Handle a specific intent."""

        if intent == 'greeting':
            return self.response_generator.greeting()

        elif intent == 'farewell':
            return self.response_generator.farewell()

        elif intent == 'thanks':
            return self.response_generator.thanks()

        elif intent == 'help':
            return self.response_generator.help()

        elif intent == 'status':
            return self.response_generator.status(self.session)

        elif intent == 'list_skills':
            return self.response_generator.skills_list()

        elif intent == 'explain':
            return self.response_generator.explain(text)

        elif intent == 'fetch_data':
            return self._handle_fetch(text)

        elif intent == 'detect_transit':
            return self._handle_transit_detection(text)

        elif intent == 'detect_anomaly':
            return self._handle_anomaly_detection(text)

        elif intent == 'analyze':
            return self._handle_analysis(text)

        elif intent == 'generate_report':
            return self._handle_report(text)

        elif intent == 'train_model':
            return self._handle_training(text)

        elif intent == 'target_star':
            target = self.intent_recognizer.extract_target(text)
            if target:
                self.session.current_target = target
                return f"Target set to **{target}**. What would you like to do?\n- Fetch light curve data\n- Search for transits\n- Check for anomalies"
            return "I couldn't identify the target. Please specify like 'TIC 307210830' or 'Kepler-186'."

        elif intent == 'habitability':
            return self._handle_habitability(text)

        elif intent in ['affirmative', 'negative']:
            return "I'm ready for your next request! What would you like to analyze?"

        else:
            return self.response_generator.dont_understand(text)

    def _handle_pending(self, text: str) -> str:
        """Handle response to a pending action."""
        intents = self.intent_recognizer.recognize(text)

        if intents and intents[0][0] == 'affirmative':
            action = self.pending_action
            self.pending_action = None

            if action['type'] == 'fetch':
                return self._execute_fetch(action['target'])
            elif action['type'] == 'detect':
                return self._execute_detection()

        elif intents and intents[0][0] == 'negative':
            self.pending_action = None
            return "No problem! What would you like to do instead?"

        self.pending_action = None
        return self._handle_intent(intents[0][0] if intents else 'unknown', text)

    def _handle_fetch(self, text: str) -> str:
        """Handle data fetch request."""
        target = self.intent_recognizer.extract_target(text) or self.session.current_target

        if not target:
            return "Which star would you like to fetch data for? Please specify a target like:\n- TIC 307210830\n- Kepler-186\n- TOI 700"

        self.session.current_target = target
        return self._execute_fetch(target)

    def _execute_fetch(self, target: str) -> str:
        """Execute the data fetch."""
        print(f"\n{Colors.DIM}Fetching data for {target}...{Colors.END}")

        try:
            import lightkurve as lk
            import numpy as np

            # Search for light curves
            search_result = lk.search_lightcurve(target)

            if len(search_result) == 0:
                return self.response_generator.fetch_response(target, False)

            # Download first result
            lc = search_result[0].download()
            lc = lc.remove_nans().normalize()

            # Store in session
            self.session.context['lightcurve'] = {
                'time': lc.time.value,
                'flux': lc.flux.value,
                'target': target
            }

            details = {
                'n_points': len(lc.time),
                'source': search_result[0].mission,
                'time_span': f"{lc.time.value.max() - lc.time.value.min():.1f}"
            }

            self.session.last_analysis = {'type': 'fetch', 'target': target}

            return self.response_generator.fetch_response(target, True, details)

        except ImportError:
            # Simulate for demo if lightkurve not available
            import numpy as np

            # Create synthetic data
            time = np.linspace(0, 27.4, 1000)
            flux = 1.0 + 0.001 * np.random.randn(1000)

            # Add a synthetic transit
            transit_idx = np.where((time > 5) & (time < 5.2))[0]
            flux[transit_idx] -= 0.002

            self.session.context['lightcurve'] = {
                'time': time,
                'flux': flux,
                'target': target
            }

            details = {
                'n_points': len(time),
                'source': 'Synthetic (demo mode)',
                'time_span': '27.4'
            }

            self.session.last_analysis = {'type': 'fetch', 'target': target}

            return self.response_generator.fetch_response(target, True, details) + "\n*(Note: Using synthetic data - lightkurve not installed)*"

        except Exception as e:
            return self.response_generator.fetch_response(target, False)

    def _handle_transit_detection(self, text: str) -> str:
        """Handle transit detection request."""
        if 'lightcurve' not in self.session.context:
            if self.session.current_target:
                return f"I'll need to fetch the light curve first. Fetching data for {self.session.current_target}...\n\n" + self._execute_fetch(self.session.current_target) + "\n\nNow running transit detection..."
            return "I need light curve data first. Which star would you like me to analyze? (e.g., 'Fetch data for TIC 307210830')"

        return self._execute_detection()

    def _execute_detection(self) -> str:
        """Execute transit detection."""
        print(f"\n{Colors.DIM}Running transit detection...{Colors.END}")

        import numpy as np

        lc = self.session.context.get('lightcurve', {})
        time = np.array(lc.get('time', []))
        flux = np.array(lc.get('flux', []))

        if len(time) == 0:
            return "No light curve data available. Please fetch data first."

        # Simple transit detection
        candidates = []

        median_flux = np.median(flux)
        std_flux = np.std(flux)
        threshold = median_flux - 3 * std_flux

        # Find dips
        dips = flux < threshold
        in_dip = False
        dip_start = 0

        for i in range(len(dips)):
            if dips[i] and not in_dip:
                in_dip = True
                dip_start = i
            elif not dips[i] and in_dip:
                in_dip = False
                if i - dip_start > 3:
                    duration_hours = (time[i] - time[dip_start]) * 24
                    depth = 1 - np.min(flux[dip_start:i]) / median_flux
                    snr = depth / std_flux * np.sqrt(i - dip_start)

                    if snr > 5:
                        candidates.append({
                            'start_time': float(time[dip_start]),
                            'mid_time': float(np.mean(time[dip_start:i])),
                            'duration_hours': float(duration_hours),
                            'depth': float(depth),
                            'snr': float(snr)
                        })

        # Estimate period if multiple candidates
        period = None
        if len(candidates) >= 2:
            mid_times = sorted([c['mid_time'] for c in candidates])
            diffs = np.diff(mid_times)
            if len(diffs) > 0:
                period = float(np.median(diffs))

        results = {
            'n_candidates': len(candidates),
            'candidates': candidates,
            'estimated_period': period
        }

        self.session.last_analysis = {'type': 'transit_detection', 'results': results}

        return self.response_generator.transit_detection_response(results)

    def _handle_anomaly_detection(self, text: str) -> str:
        """Handle anomaly detection request."""
        if 'lightcurve' not in self.session.context:
            if self.session.current_target:
                self._execute_fetch(self.session.current_target)
            else:
                return "I need light curve data first. Which star would you like me to analyze?"

        print(f"\n{Colors.DIM}Running anomaly detection...{Colors.END}")

        import numpy as np

        lc = self.session.context.get('lightcurve', {})
        flux = np.array(lc.get('flux', []))

        if len(flux) == 0:
            return "No light curve data available."

        # Simple anomaly detection
        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        robust_std = 1.4826 * mad

        z_scores = np.abs(flux - median) / (robust_std + 1e-10)
        anomaly_mask = z_scores > 3.0

        anomaly_types = {}
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly:
                if flux[i] < median:
                    anomaly_types[i] = 'dip'
                else:
                    anomaly_types[i] = 'brightening'

        results = {
            'n_anomalies': int(np.sum(anomaly_mask)),
            'anomaly_types': anomaly_types
        }

        self.session.last_analysis = {'type': 'anomaly_detection', 'results': results}

        return self.response_generator.anomaly_response(results)

    def _handle_analysis(self, text: str) -> str:
        """Handle general analysis request."""
        if 'lightcurve' not in self.session.context:
            if self.session.current_target:
                return self._execute_fetch(self.session.current_target)
            return "What would you like me to analyze? Please specify a target star first."

        lc = self.session.context['lightcurve']
        import numpy as np

        time = np.array(lc['time'])
        flux = np.array(lc['flux'])

        response = f"""
**Light Curve Analysis: {lc.get('target', 'Unknown')}**

**Basic Statistics:**
- Data points: {len(time)}
- Time span: {time.max() - time.min():.2f} days
- Mean flux: {np.mean(flux):.6f}
- Flux std: {np.std(flux):.6f}
- Min flux: {np.min(flux):.6f}
- Max flux: {np.max(flux):.6f}

**Variability:**
- RMS: {np.sqrt(np.mean((flux - np.mean(flux))**2)):.6f}
- Range: {(np.max(flux) - np.min(flux)) * 100:.4f}%

Would you like me to:
- "Search for transits"
- "Check for anomalies"
- "Generate a report"
"""
        return response

    def _handle_report(self, text: str) -> str:
        """Handle report generation request."""
        if not self.session.last_analysis:
            return "I don't have any analysis results to report on yet. Try running a transit detection or anomaly analysis first."

        analysis = self.session.last_analysis
        target = self.session.current_target or "Unknown"

        report = f"""
**LARUN Analysis Report**
========================

**Target:** {target}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type:** {analysis.get('type', 'N/A')}

"""

        if analysis['type'] == 'transit_detection':
            results = analysis.get('results', {})
            report += f"""
**Transit Detection Summary:**
- Candidates found: {results.get('n_candidates', 0)}
- Estimated period: {results.get('estimated_period', 'N/A')} days

**Recommendation:** {'Submit for follow-up observation' if results.get('n_candidates', 0) > 0 else 'No significant signals - consider longer baseline'}
"""

        report += """
---
*Report generated by LARUN TinyML*
*Created by Padmanaban Veeraragavalu (Larun Engineering)*
"""

        return report

    def _handle_training(self, text: str) -> str:
        """Handle model training request."""
        return """
Model training is a computationally intensive task. To train the LARUN TinyML model:

**Option 1: Quick Training (Synthetic Data)**
```bash
python standalone_demo.py
```

**Option 2: Full Training (Real NASA Data)**
```bash
python train_real_data.py --planets 100 --epochs 100
```

**Option 3: Complete Pipeline**
```bash
python run_pipeline.py --num-stars 50 --epochs 100
```

Would you like me to explain the training process in more detail?
"""

    def _handle_habitability(self, text: str) -> str:
        """Handle habitability-related queries."""
        if self.session.last_analysis and self.session.last_analysis.get('type') == 'transit_detection':
            results = self.session.last_analysis.get('results', {})
            if results.get('estimated_period'):
                period = results['estimated_period']

                # Very rough habitability estimate (would need stellar parameters for accuracy)
                return f"""
**Habitability Assessment**

For the detected candidate with period **{period:.2f} days**:

Without knowing the host star's properties, I can't precisely determine habitability. However:

- A {period:.1f}-day orbit is relatively **{'short' if period < 10 else 'moderate' if period < 50 else 'long'}**
- For a Sun-like star, the habitable zone is around 200-400 days
- For cooler M-dwarf stars, {period:.1f} days could be in the habitable zone!

To properly assess habitability, I would need:
1. Stellar effective temperature (Teff)
2. Stellar luminosity
3. Planet radius estimate

Would you like me to explain more about habitable zones?
"""

        return self.response_generator.explain("habitable zone")


# ============================================================================
# TERMINAL INTERFACE
# ============================================================================

def print_chat_banner():
    print(f"""
{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     {Colors.CYAN}LARUN Chat{Colors.BOLD}                                            ║
║     {Colors.DIM}Conversational AI for Astronomical Analysis{Colors.BOLD}            ║
║                                                              ║
║     {Colors.GREEN}Type naturally - I understand astronomy!{Colors.BOLD}               ║
║     {Colors.DIM}Type 'help' for commands, 'quit' to exit{Colors.BOLD}                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.END}
""")


def format_response(text: str) -> str:
    """Format response with markdown-style rendering."""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', f'{Colors.BOLD}\\1{Colors.END}', text)
    # Cyan for code blocks
    text = re.sub(r'`(.+?)`', f'{Colors.CYAN}\\1{Colors.END}', text)
    return text


def run_chat():
    """Run the chat interface."""
    print_chat_banner()

    chat = LarunChat()

    # Check if model is available
    if chat.load_model():
        print(f"{Colors.GREEN}Model loaded successfully.{Colors.END}")
    else:
        print(f"{Colors.YELLOW}Running in demo mode (model not found).{Colors.END}")

    print(f"\n{Colors.CYAN}LARUN:{Colors.END} {chat.response_generator.greeting()}\n")

    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.GREEN}You:{Colors.END} ").strip()

            if not user_input:
                continue

            # Check for quit
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print(f"\n{Colors.CYAN}LARUN:{Colors.END} {chat.response_generator.farewell()}\n")
                break

            # Process message
            response = chat.process_message(user_input)

            # Format and print response
            formatted = format_response(response)
            print(f"\n{Colors.CYAN}LARUN:{Colors.END} {formatted}\n")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}LARUN:{Colors.END} {chat.response_generator.farewell()}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.END}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LARUN Chat - Conversational AI for Astronomy")
    parser.add_argument('--api', action='store_true', help='Start in API server mode')
    parser.add_argument('--port', type=int, default=8080, help='API server port')

    args = parser.parse_args()

    if args.api:
        print("API server mode not yet implemented. Running chat mode instead.")

    run_chat()


def main():
    """Entry point for CLI."""
    run_chat()

