"""
Multi-Node Result Aggregator for LARUN Federated TinyML System

Combines results from multiple specialized nodes into a unified analysis.
Handles consensus building, conflict resolution, and result prioritization.

Usage:
    aggregator = NodeAggregator()

    # Run multiple nodes
    results = [node.run(light_curve) for node in enabled_nodes]

    # Aggregate results
    combined = aggregator.aggregate(results, target_id="TIC_123456")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import numpy as np

from .base import NodeResult, NodeCategory


class AggregationStrategy(Enum):
    """Strategy for combining multiple node results."""
    UNANIMOUS = "unanimous"       # All nodes must agree
    MAJORITY = "majority"         # >50% must agree
    ANY = "any"                   # Any detection triggers
    WEIGHTED = "weighted"         # Weight by confidence
    PRIORITY = "priority"         # Use highest-priority node


@dataclass
class AggregatedResult:
    """Combined result from multiple nodes."""
    target_id: str                        # What was analyzed
    node_results: List[NodeResult]        # Individual node results
    primary_classification: str           # Combined classification
    overall_confidence: float             # Aggregated confidence
    detections_by_category: Dict[str, List[Dict]]  # Detections grouped
    consensus: bool                       # Whether nodes agree
    conflicting_nodes: List[str]          # Nodes that disagree
    summary: Dict[str, Any]               # Summary statistics
    recommendations: List[str]            # Follow-up recommendations
    total_inference_time_ms: float        # Total processing time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'target_id': self.target_id,
            'node_results': [r.to_dict() for r in self.node_results],
            'primary_classification': self.primary_classification,
            'overall_confidence': self.overall_confidence,
            'detections_by_category': self.detections_by_category,
            'consensus': self.consensus,
            'conflicting_nodes': self.conflicting_nodes,
            'summary': self.summary,
            'recommendations': self.recommendations,
            'total_inference_time_ms': self.total_inference_time_ms,
        }

    def get_detections(self, category: Optional[str] = None) -> List[Dict]:
        """Get all detections, optionally filtered by category."""
        if category:
            return self.detections_by_category.get(category, [])
        return [d for dets in self.detections_by_category.values() for d in dets]

    def has_detection(self, classification: str) -> bool:
        """Check if any node detected a specific classification."""
        return any(
            r.classification == classification
            for r in self.node_results
            if r.success
        )


@dataclass
class AggregatorConfig:
    """Configuration for result aggregation."""
    strategy: AggregationStrategy = AggregationStrategy.WEIGHTED
    min_confidence: float = 0.5          # Minimum confidence to include
    consensus_threshold: float = 0.7     # Agreement threshold for consensus
    priority_order: List[str] = field(default_factory=lambda: [
        'EXOPLANET-001',  # Highest priority
        'SUPERNOVA-001',
        'MICROLENS-001',
        'VSTAR-001',
        'FLARE-001',
        'ASTERO-001',
        'SPECTYPE-001',
        'GALAXY-001',
    ])


class NodeAggregator:
    """
    Aggregates results from multiple LARUN analysis nodes.

    Provides:
    - Result combination with multiple strategies
    - Consensus detection
    - Conflict resolution
    - Priority-based result selection
    - Summary statistics
    """

    # Classification priority (for conflict resolution)
    CLASSIFICATION_PRIORITY = {
        'planetary_transit': 10,
        'supernova': 9,
        'microlensing': 8,
        'eclipsing_binary': 7,
        'variable_star': 6,
        'flare': 5,
        'stellar_signal': 4,
        'stellar_oscillation': 3,
        'galaxy': 2,
        'noise': 1,
        'unknown': 0,
        'error': -1,
    }

    def __init__(self, config: Optional[AggregatorConfig] = None):
        """
        Initialize the aggregator.

        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregatorConfig()

    def aggregate(self, results: List[NodeResult],
                  target_id: str = "unknown") -> AggregatedResult:
        """
        Aggregate results from multiple nodes.

        Args:
            results: List of NodeResult from different nodes
            target_id: Identifier for the analyzed target

        Returns:
            AggregatedResult combining all node outputs
        """
        # Filter out failed results for classification
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if not successful:
            return self._create_error_result(results, target_id, failed)

        # Calculate primary classification based on strategy
        if self.config.strategy == AggregationStrategy.WEIGHTED:
            primary, confidence = self._weighted_classification(successful)
        elif self.config.strategy == AggregationStrategy.MAJORITY:
            primary, confidence = self._majority_classification(successful)
        elif self.config.strategy == AggregationStrategy.ANY:
            primary, confidence = self._any_classification(successful)
        elif self.config.strategy == AggregationStrategy.PRIORITY:
            primary, confidence = self._priority_classification(successful)
        else:  # UNANIMOUS
            primary, confidence = self._unanimous_classification(successful)

        # Check for consensus
        consensus, conflicting = self._check_consensus(successful)

        # Group detections by category
        detections_by_cat = self._group_detections(results)

        # Generate summary
        summary = self._generate_summary(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            successful, primary, confidence, consensus
        )

        # Calculate total inference time
        total_time = sum(r.inference_time_ms for r in results)

        return AggregatedResult(
            target_id=target_id,
            node_results=results,
            primary_classification=primary,
            overall_confidence=confidence,
            detections_by_category=detections_by_cat,
            consensus=consensus,
            conflicting_nodes=conflicting,
            summary=summary,
            recommendations=recommendations,
            total_inference_time_ms=total_time,
        )

    def _weighted_classification(self, results: List[NodeResult]
                                  ) -> tuple[str, float]:
        """
        Determine classification using confidence-weighted voting.

        Each node's vote is weighted by its confidence score.
        """
        votes: Dict[str, float] = {}

        for r in results:
            if r.confidence >= self.config.min_confidence:
                cls = r.classification
                votes[cls] = votes.get(cls, 0) + r.confidence

        if not votes:
            # Fall back to highest confidence result
            best = max(results, key=lambda r: r.confidence)
            return best.classification, best.confidence

        # Return classification with highest weighted vote
        primary = max(votes.keys(), key=lambda c: votes[c])

        # Calculate confidence as average of votes for primary
        primary_results = [r for r in results if r.classification == primary]
        confidence = np.mean([r.confidence for r in primary_results])

        return primary, float(confidence)

    def _majority_classification(self, results: List[NodeResult]
                                  ) -> tuple[str, float]:
        """Determine classification using simple majority voting."""
        votes: Dict[str, int] = {}

        for r in results:
            if r.confidence >= self.config.min_confidence:
                votes[r.classification] = votes.get(r.classification, 0) + 1

        if not votes:
            best = max(results, key=lambda r: r.confidence)
            return best.classification, best.confidence

        primary = max(votes.keys(), key=lambda c: votes[c])
        vote_fraction = votes[primary] / len(results)

        # Confidence is fraction of agreeing nodes
        return primary, vote_fraction

    def _any_classification(self, results: List[NodeResult]
                            ) -> tuple[str, float]:
        """
        Return the highest-priority classification from any node.

        Use this when you want to detect any interesting signal.
        """
        # Sort by classification priority, then confidence
        sorted_results = sorted(
            results,
            key=lambda r: (
                self.CLASSIFICATION_PRIORITY.get(r.classification, 0),
                r.confidence
            ),
            reverse=True
        )

        best = sorted_results[0]
        return best.classification, best.confidence

    def _priority_classification(self, results: List[NodeResult]
                                  ) -> tuple[str, float]:
        """
        Use classification from highest-priority node.

        Node priority is defined in config.priority_order.
        """
        # Sort by node priority
        node_priority = {
            node_id: i
            for i, node_id in enumerate(self.config.priority_order)
        }

        sorted_results = sorted(
            results,
            key=lambda r: node_priority.get(r.node_id, 999)
        )

        best = sorted_results[0]
        return best.classification, best.confidence

    def _unanimous_classification(self, results: List[NodeResult]
                                   ) -> tuple[str, float]:
        """
        Require all nodes to agree.

        Returns 'inconclusive' if nodes disagree.
        """
        classifications = set(r.classification for r in results)

        if len(classifications) == 1:
            cls = classifications.pop()
            confidence = np.mean([r.confidence for r in results])
            return cls, float(confidence)
        else:
            return 'inconclusive', 0.0

    def _check_consensus(self, results: List[NodeResult]
                         ) -> tuple[bool, List[str]]:
        """
        Check if nodes agree and identify conflicts.

        Returns:
            (consensus_reached, list_of_conflicting_node_ids)
        """
        if len(results) < 2:
            return True, []

        # Get all classifications
        classifications: Dict[str, List[str]] = {}
        for r in results:
            cls = r.classification
            if cls not in classifications:
                classifications[cls] = []
            classifications[cls].append(r.node_id)

        # Check if any classification has enough support
        total = len(results)
        for cls, node_ids in classifications.items():
            if len(node_ids) / total >= self.config.consensus_threshold:
                # Find conflicting nodes
                conflicting = [
                    r.node_id for r in results
                    if r.classification != cls
                ]
                return True, conflicting

        # No consensus
        # Find the minority nodes
        majority_cls = max(classifications.keys(),
                          key=lambda c: len(classifications[c]))
        conflicting = [
            node_id for cls, node_ids in classifications.items()
            for node_id in node_ids
            if cls != majority_cls
        ]

        return False, conflicting

    def _group_detections(self, results: List[NodeResult]
                          ) -> Dict[str, List[Dict]]:
        """Group all detections by category."""
        grouped: Dict[str, List[Dict]] = {}

        for r in results:
            if not r.success:
                continue

            # Determine category from node ID
            node_id = r.node_id
            if 'EXOPLANET' in node_id:
                category = 'exoplanet'
            elif 'VSTAR' in node_id:
                category = 'variable_star'
            elif 'FLARE' in node_id:
                category = 'flare'
            elif 'ASTERO' in node_id:
                category = 'asteroseismology'
            elif 'SUPERNOVA' in node_id:
                category = 'transient'
            elif 'GALAXY' in node_id:
                category = 'galaxy'
            elif 'SPECTYPE' in node_id:
                category = 'spectral'
            elif 'MICROLENS' in node_id:
                category = 'microlensing'
            else:
                category = 'other'

            if category not in grouped:
                grouped[category] = []

            for detection in r.detections:
                detection['source_node'] = node_id
                detection['confidence'] = r.confidence
                grouped[category].append(detection)

        return grouped

    def _generate_summary(self, results: List[NodeResult]) -> Dict[str, Any]:
        """Generate summary statistics."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        return {
            'nodes_run': len(results),
            'nodes_successful': len(successful),
            'nodes_failed': len(failed),
            'failed_nodes': [r.node_id for r in failed],
            'classifications': {
                r.node_id: r.classification
                for r in successful
            },
            'confidences': {
                r.node_id: r.confidence
                for r in successful
            },
            'avg_confidence': float(np.mean([r.confidence for r in successful])) if successful else 0,
            'max_confidence': max([r.confidence for r in successful]) if successful else 0,
            'total_detections': sum(len(r.detections) for r in successful),
        }

    def _generate_recommendations(self, results: List[NodeResult],
                                   primary: str, confidence: float,
                                   consensus: bool) -> List[str]:
        """Generate follow-up recommendations."""
        recommendations = []

        # Low confidence
        if confidence < 0.6:
            recommendations.append(
                "Low confidence detection. Consider collecting more data "
                "or running additional analysis."
            )

        # No consensus
        if not consensus:
            recommendations.append(
                "Nodes disagree on classification. Manual review recommended."
            )

        # Classification-specific recommendations
        if primary == 'planetary_transit':
            recommendations.append(
                "Potential exoplanet transit detected. Run full vetting suite."
            )
            recommendations.append(
                "Cross-match with known exoplanet catalogs."
            )

        elif primary == 'eclipsing_binary':
            recommendations.append(
                "Eclipsing binary detected. Check for odd-even transit depth variations."
            )

        elif primary == 'variable_star':
            recommendations.append(
                "Variable star detected. Run period analysis and classify variability type."
            )

        elif primary == 'flare':
            recommendations.append(
                "Stellar flare detected. Check for periodicity and characterize flare energy."
            )

        elif primary == 'microlensing':
            recommendations.append(
                "Potential microlensing event. Monitor for chromaticity and parallax effects."
            )

        elif primary == 'supernova':
            recommendations.append(
                "Transient detected. Prioritize spectroscopic follow-up."
            )

        # Check for multiple interesting detections
        detection_types = set()
        for r in results:
            if r.confidence >= 0.5:
                detection_types.add(r.classification)

        if len(detection_types) > 2:
            recommendations.append(
                f"Multiple phenomena detected ({', '.join(detection_types)}). "
                "This target may warrant detailed study."
            )

        return recommendations

    def _create_error_result(self, results: List[NodeResult],
                             target_id: str,
                             failed: List[NodeResult]) -> AggregatedResult:
        """Create result when all nodes failed."""
        error_messages = [r.error_message for r in failed if r.error_message]

        return AggregatedResult(
            target_id=target_id,
            node_results=results,
            primary_classification='error',
            overall_confidence=0.0,
            detections_by_category={},
            consensus=False,
            conflicting_nodes=[],
            summary={
                'nodes_run': len(results),
                'nodes_successful': 0,
                'nodes_failed': len(failed),
                'error_messages': error_messages,
            },
            recommendations=["All nodes failed. Check data quality and node configuration."],
            total_inference_time_ms=sum(r.inference_time_ms for r in results),
        )

    def format_result(self, result: AggregatedResult,
                      verbose: bool = False) -> str:
        """Format aggregated result for display."""
        lines = []

        lines.append(f"\n Multi-Node Analysis: {result.target_id}")
        lines.append("=" * 60)

        # Primary result
        confidence_bar = "" * int(result.overall_confidence * 10)
        confidence_bar += "" * (10 - int(result.overall_confidence * 10))

        lines.append(f"\nPrimary Classification: {result.primary_classification.upper()}")
        lines.append(f"Overall Confidence: {result.overall_confidence:.1%} [{confidence_bar}]")
        lines.append(f"Consensus: {'Yes' if result.consensus else 'No'}")

        if result.conflicting_nodes:
            lines.append(f"Conflicting: {', '.join(result.conflicting_nodes)}")

        # Node results
        lines.append("\n Node Results:")
        lines.append("-" * 60)

        for r in result.node_results:
            status = "" if r.success else ""
            conf = f"{r.confidence:.1%}" if r.success else "N/A"
            time_str = f"{r.inference_time_ms:.1f}ms"

            lines.append(
                f"  {status} {r.node_id:<14} {r.classification:<20} "
                f"{conf:>6} {time_str:>8}"
            )

            if not r.success and verbose:
                lines.append(f"      Error: {r.error_message}")

        # Detections
        total_detections = sum(len(d) for d in result.detections_by_category.values())
        if total_detections > 0:
            lines.append(f"\n Detections ({total_detections} total):")
            lines.append("-" * 60)

            for category, detections in result.detections_by_category.items():
                lines.append(f"  [{category.upper()}]")
                for det in detections[:5]:  # Limit to 5 per category
                    det_str = ", ".join(f"{k}={v}" for k, v in det.items()
                                       if k not in ('source_node', 'confidence'))
                    lines.append(f"    - {det_str}")
                if len(detections) > 5:
                    lines.append(f"    ... and {len(detections) - 5} more")

        # Recommendations
        if result.recommendations:
            lines.append("\n Recommendations:")
            lines.append("-" * 60)
            for rec in result.recommendations:
                lines.append(f"  - {rec}")

        # Summary
        lines.append("\n Summary:")
        lines.append(f"  Nodes: {result.summary['nodes_successful']}/{result.summary['nodes_run']} successful")
        lines.append(f"  Total inference time: {result.total_inference_time_ms:.1f}ms")

        return "\n".join(lines)
