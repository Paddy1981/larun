"""
Prediction Explainer for LARUN Educational Trainer
===================================================

Explains model predictions in educational terms for students.
Shows why a model classified something a certain way.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FeatureImportance:
    """Importance of a feature in a prediction."""
    name: str
    importance: float
    value: float
    description: str
    direction: str = "positive"  # positive or negative contribution


@dataclass
class ExampleComparison:
    """Comparison to a known example."""
    example_type: str
    similarity: float
    description: str
    key_differences: List[str] = field(default_factory=list)


@dataclass
class Explanation:
    """Complete explanation of a prediction."""
    prediction_class: str
    confidence: float
    summary: str
    features: List[FeatureImportance]
    comparisons: List[ExampleComparison]
    educational_notes: List[str]
    visualizations: Dict[str, Any] = field(default_factory=dict)


class PredictionExplainer:
    """
    Explains model predictions in educational terms.

    Provides:
    - Feature importance analysis
    - Comparison to known examples
    - Educational context
    - Visualization suggestions
    """

    # Class labels for each node type
    CLASS_LABELS = {
        'exoplanet': [
            'noise', 'stellar_signal', 'planetary_transit',
            'eclipsing_binary', 'instrument_artifact', 'unknown_anomaly'
        ],
        'variable_star': [
            'cepheid', 'rr_lyrae', 'delta_scuti', 'eclipsing_binary',
            'rotational', 'irregular', 'constant'
        ],
        'flare': [
            'no_flare', 'weak_flare', 'moderate_flare',
            'strong_flare', 'superflare'
        ],
        'transient': [
            'no_transient', 'sn_ia', 'sn_ii', 'sn_ibc',
            'kilonova', 'tde', 'other_transient'
        ],
    }

    # Educational descriptions for each class
    CLASS_DESCRIPTIONS = {
        # Exoplanet classes
        'planetary_transit': {
            'name': 'Planetary Transit',
            'description': 'A planet passing in front of its star, blocking some light.',
            'features': ['periodic dip', 'shallow depth (<1%)', 'flat bottom', 'symmetric shape'],
            'fun_fact': 'Most exoplanets are found using this method!',
        },
        'eclipsing_binary': {
            'name': 'Eclipsing Binary',
            'description': 'Two stars orbiting each other, taking turns blocking light.',
            'features': ['deep dips (>1%)', 'two different dip depths', 'V-shaped dips'],
            'fun_fact': 'About half of all stars are in binary systems!',
        },
        'stellar_signal': {
            'name': 'Stellar Variability',
            'description': 'Natural brightness changes from the star itself.',
            'features': ['gradual changes', 'irregular pattern', 'starspot modulation'],
            'fun_fact': 'Our Sun also varies in brightness due to sunspots!',
        },

        # Variable star classes
        'cepheid': {
            'name': 'Cepheid Variable',
            'description': 'A pulsating giant star with a regular period.',
            'features': ['asymmetric light curve', 'fast rise, slow decline', 'periods 1-100 days'],
            'fun_fact': 'Cepheids helped prove the universe is expanding!',
        },
        'rr_lyrae': {
            'name': 'RR Lyrae Variable',
            'description': 'An old, pulsating star found in globular clusters.',
            'features': ['short period (~0.5 days)', 'sawtooth light curve', 'large amplitude'],
            'fun_fact': 'Used as "standard candles" to measure distances!',
        },

        # Flare classes
        'superflare': {
            'name': 'Superflare',
            'description': 'An extremely powerful stellar flare.',
            'features': ['sudden brightness spike', 'energy >10^33 ergs', 'rare event'],
            'fun_fact': 'Can be 10,000 times more powerful than solar flares!',
        },

        # Transient classes
        'sn_ia': {
            'name': 'Type Ia Supernova',
            'description': 'A white dwarf exploding after gaining too much mass.',
            'features': ['consistent peak brightness', '~3 week rise time', 'no hydrogen lines'],
            'fun_fact': 'Used to discover the accelerating expansion of the universe!',
        },
    }

    def __init__(self):
        """Initialize the explainer."""
        pass

    def explain(
        self,
        prediction: Dict[str, Any],
        data: Optional[Any] = None,
        node_type: str = 'exoplanet',
    ) -> Dict[str, Any]:
        """
        Generate an explanation for a prediction.

        Args:
            prediction: Model prediction result
            data: Optional input data
            node_type: Type of node (exoplanet, variable_star, etc.)

        Returns:
            Explanation dictionary
        """
        # Extract prediction info
        pred_class = prediction.get('predicted_class', 0)
        confidence = prediction.get('confidence', 0.0)
        probabilities = prediction.get('probabilities', [])

        # Get class labels
        labels = self.CLASS_LABELS.get(node_type, ['unknown'])
        class_name = labels[pred_class] if pred_class < len(labels) else 'unknown'

        # Get class description
        class_info = self.CLASS_DESCRIPTIONS.get(class_name, {
            'name': class_name.replace('_', ' ').title(),
            'description': 'No description available.',
            'features': [],
            'fun_fact': '',
        })

        # Build explanation
        explanation = {
            'prediction': {
                'class': class_name,
                'class_name': class_info['name'],
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%",
            },
            'summary': self._generate_summary(class_info, confidence),
            'description': class_info['description'],
            'key_features': class_info.get('features', []),
            'fun_fact': class_info.get('fun_fact', ''),
            'confidence_interpretation': self._interpret_confidence(confidence),
            'alternative_classes': self._get_alternatives(probabilities, labels),
            'educational_notes': self._get_educational_notes(class_name, node_type),
            'what_to_look_for': self._get_visual_cues(class_name),
        }

        # Add data-specific analysis if provided
        if data is not None:
            explanation['data_analysis'] = self._analyze_data(data, class_name)

        return explanation

    def _generate_summary(self, class_info: Dict, confidence: float) -> str:
        """Generate a plain-language summary."""
        name = class_info.get('name', 'this signal')
        conf_word = 'confidently' if confidence > 0.8 else 'likely' if confidence > 0.5 else 'possibly'

        return f"The model {conf_word} identifies this as a **{name}**. {class_info.get('description', '')}"

    def _interpret_confidence(self, confidence: float) -> Dict[str, Any]:
        """Interpret the confidence level for students."""
        if confidence > 0.95:
            level = 'Very High'
            description = 'The model is very sure about this classification.'
            emoji = ''
        elif confidence > 0.8:
            level = 'High'
            description = 'The model is fairly confident, but not certain.'
            emoji = ''
        elif confidence > 0.6:
            level = 'Moderate'
            description = 'The model has some uncertainty. Other classes are possible.'
            emoji = ''
        elif confidence > 0.4:
            level = 'Low'
            description = 'The model is uncertain. This might be something else.'
            emoji = ''
        else:
            level = 'Very Low'
            description = 'The model is guessing. More data or analysis is needed.'
            emoji = ''

        return {
            'level': level,
            'description': description,
            'emoji': emoji,
            'percentage': f"{confidence * 100:.1f}%",
        }

    def _get_alternatives(
        self,
        probabilities: List[float],
        labels: List[str],
    ) -> List[Dict[str, Any]]:
        """Get alternative classifications."""
        if not probabilities or not labels:
            return []

        alternatives = []
        sorted_indices = np.argsort(probabilities)[::-1]

        for idx in sorted_indices[1:4]:  # Top 3 alternatives
            if idx < len(labels) and probabilities[idx] > 0.05:
                class_name = labels[idx]
                info = self.CLASS_DESCRIPTIONS.get(class_name, {})

                alternatives.append({
                    'class': class_name,
                    'name': info.get('name', class_name.replace('_', ' ').title()),
                    'probability': f"{probabilities[idx] * 100:.1f}%",
                    'description': info.get('description', ''),
                })

        return alternatives

    def _get_educational_notes(self, class_name: str, node_type: str) -> List[str]:
        """Get educational notes for the prediction."""
        notes = []

        # General notes based on node type
        if node_type == 'exoplanet':
            notes.append("Transit detection requires observing many orbits to confirm periodicity.")
            if class_name == 'planetary_transit':
                notes.append("Real planet signals must be verified with follow-up observations.")
                notes.append("The transit depth tells us the planet's size relative to the star.")

        elif node_type == 'variable_star':
            notes.append("Variable stars are classified by their light curve shapes and periods.")
            if class_name in ['cepheid', 'rr_lyrae']:
                notes.append("These stars are used as 'standard candles' to measure cosmic distances.")

        elif node_type == 'flare':
            notes.append("Stellar flares release enormous energy in short time spans.")
            notes.append("Flare activity affects the habitability of orbiting planets.")

        return notes

    def _get_visual_cues(self, class_name: str) -> List[str]:
        """Get visual cues to look for in the data."""
        cues = {
            'planetary_transit': [
                'Look for a flat-bottomed dip in brightness',
                'The dip should be symmetric (same ingress and egress)',
                'Depth is typically less than 1%',
                'Pattern repeats at regular intervals',
            ],
            'eclipsing_binary': [
                'Look for two dips of different depths per period',
                'Dips are often V-shaped rather than flat-bottomed',
                'Depth can be several percent',
                'May see ellipsoidal variations between eclipses',
            ],
            'cepheid': [
                'Look for an asymmetric waveform',
                'Fast rise, slow decline pattern',
                'Period typically 1-100 days',
                'Amplitude can be large (up to 1 magnitude)',
            ],
            'superflare': [
                'Look for a sudden, sharp spike in brightness',
                'Very fast rise (seconds to minutes)',
                'Slower exponential decay',
                'Can increase brightness by factors of 10-1000',
            ],
        }

        return cues.get(class_name, ['Examine the overall shape and periodicity of the signal'])

    def _analyze_data(self, data: Any, class_name: str) -> Dict[str, Any]:
        """Analyze input data for educational insights."""
        try:
            data = np.array(data).flatten()

            analysis = {
                'length': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'range': float(np.max(data) - np.min(data)),
            }

            # Add class-specific analysis
            if class_name == 'planetary_transit':
                # Look for dips
                median = np.median(data)
                below_median = data < (median - np.std(data))
                analysis['potential_transit_points'] = int(np.sum(below_median))
                analysis['dip_fraction'] = float(np.sum(below_median) / len(data))

            return analysis

        except Exception:
            return {'error': 'Could not analyze data'}

    def explain_simple(self, class_name: str) -> str:
        """
        Get a simple one-paragraph explanation.

        Useful for quick display in CLI.
        """
        info = self.CLASS_DESCRIPTIONS.get(class_name, {})

        if not info:
            return f"This appears to be a {class_name.replace('_', ' ')}."

        explanation = f"**{info.get('name', class_name)}**: {info.get('description', '')} "

        features = info.get('features', [])
        if features:
            explanation += f"Key features include: {', '.join(features[:3])}. "

        fun_fact = info.get('fun_fact', '')
        if fun_fact:
            explanation += f" {fun_fact}"

        return explanation
