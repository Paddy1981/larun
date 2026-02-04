"""
TinyML Pipeline Framework
=========================
A structured framework for combining TinyML models in series and parallel
configurations with human interface checkpoints.

Pipeline Types:
- Series: Output of one model feeds into the next
- Parallel: Multiple models run independently on same/different inputs
- Ensemble: Multiple models vote on same input
- Conditional: Model selection based on previous results
- Human-in-the-Loop: Requires human approval before proceeding

Based on LARUN.SPACE Technical Requirements Document (TRD-LARUN-2026-001)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import time
from datetime import datetime


class PipelineStatus(Enum):
    """Status of a pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """Type of pipeline node."""
    MODEL = "model"
    PARALLEL = "parallel"
    SERIES = "series"
    ENSEMBLE = "ensemble"
    CONDITIONAL = "conditional"
    HUMAN_CHECKPOINT = "human_checkpoint"
    DATA_TRANSFORM = "data_transform"
    AGGREGATOR = "aggregator"


@dataclass
class PipelineResult:
    """Result from a pipeline execution."""
    node_id: str
    status: PipelineStatus
    predictions: Optional[np.ndarray] = None
    confidences: Optional[np.ndarray] = None
    labels: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    human_review: Optional[Dict[str, Any]] = None
    children_results: List['PipelineResult'] = field(default_factory=list)


@dataclass
class HumanReviewRequest:
    """Request for human review at a checkpoint."""
    checkpoint_id: str
    node_id: str
    question: str
    options: List[str]
    context: Dict[str, Any]
    results_so_far: List[PipelineResult]
    timeout_seconds: Optional[int] = None
    default_action: Optional[str] = None


@dataclass
class HumanReviewResponse:
    """Response from human review."""
    checkpoint_id: str
    decision: str
    notes: Optional[str] = None
    reviewer_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class PipelineNode(ABC):
    """Abstract base class for pipeline nodes."""

    def __init__(self, node_id: str, node_type: NodeType):
        self.node_id = node_id
        self.node_type = node_type
        self.status = PipelineStatus.PENDING

    @abstractmethod
    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        """Execute the node with given inputs."""
        pass


class ModelNode(PipelineNode):
    """Node that wraps a TinyML model."""

    def __init__(self, node_id: str, model: Any, input_key: str = "data",
                 confidence_threshold: float = 0.5):
        super().__init__(node_id, NodeType.MODEL)
        self.model = model
        self.input_key = input_key
        self.confidence_threshold = confidence_threshold

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        start_time = time.time()
        self.status = PipelineStatus.RUNNING

        try:
            data = inputs.get(self.input_key)
            if data is None:
                raise ValueError(f"Missing input key: {self.input_key}")

            # Run inference
            predictions, confidences = self.model.predict(data)

            # Get labels if available
            labels = None
            if hasattr(self.model, 'class_labels'):
                labels = [self.model.class_labels[p] for p in predictions]

            self.status = PipelineStatus.COMPLETED
            return PipelineResult(
                node_id=self.node_id,
                status=self.status,
                predictions=predictions,
                confidences=confidences,
                labels=labels,
                metadata={
                    "model_type": type(self.model).__name__,
                    "confidence_threshold": self.confidence_threshold,
                    "above_threshold": bool(np.mean(confidences) >= self.confidence_threshold)
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            self.status = PipelineStatus.FAILED
            return PipelineResult(
                node_id=self.node_id,
                status=self.status,
                metadata={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000
            )


class ParallelNode(PipelineNode):
    """Node that executes multiple children in parallel."""

    def __init__(self, node_id: str, children: List[PipelineNode],
                 aggregation: str = "all"):
        super().__init__(node_id, NodeType.PARALLEL)
        self.children = children
        self.aggregation = aggregation  # "all", "any", "majority"

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        start_time = time.time()
        self.status = PipelineStatus.RUNNING

        children_results = []
        for child in self.children:
            result = child.execute(inputs, context)
            children_results.append(result)

        # Aggregate results
        all_completed = all(r.status == PipelineStatus.COMPLETED for r in children_results)
        any_failed = any(r.status == PipelineStatus.FAILED for r in children_results)

        if any_failed:
            self.status = PipelineStatus.FAILED
        elif all_completed:
            self.status = PipelineStatus.COMPLETED
        else:
            self.status = PipelineStatus.AWAITING_HUMAN

        return PipelineResult(
            node_id=self.node_id,
            status=self.status,
            metadata={
                "aggregation": self.aggregation,
                "children_count": len(self.children),
                "completed_count": sum(1 for r in children_results if r.status == PipelineStatus.COMPLETED)
            },
            children_results=children_results,
            execution_time_ms=(time.time() - start_time) * 1000
        )


class SeriesNode(PipelineNode):
    """Node that executes children in series, passing outputs to inputs."""

    def __init__(self, node_id: str, children: List[PipelineNode],
                 output_mapping: Optional[Dict[str, str]] = None):
        super().__init__(node_id, NodeType.SERIES)
        self.children = children
        self.output_mapping = output_mapping or {}

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        start_time = time.time()
        self.status = PipelineStatus.RUNNING

        children_results = []
        current_inputs = inputs.copy()

        for i, child in enumerate(self.children):
            result = child.execute(current_inputs, context)
            children_results.append(result)

            if result.status == PipelineStatus.FAILED:
                self.status = PipelineStatus.FAILED
                break

            if result.status == PipelineStatus.AWAITING_HUMAN:
                self.status = PipelineStatus.AWAITING_HUMAN
                break

            # Pass outputs to next stage
            if result.predictions is not None:
                current_inputs["previous_predictions"] = result.predictions
                current_inputs["previous_confidences"] = result.confidences
                current_inputs["previous_labels"] = result.labels

            # Custom output mapping
            for output_key, input_key in self.output_mapping.items():
                if output_key in result.metadata:
                    current_inputs[input_key] = result.metadata[output_key]

        if self.status == PipelineStatus.RUNNING:
            self.status = PipelineStatus.COMPLETED

        return PipelineResult(
            node_id=self.node_id,
            status=self.status,
            predictions=children_results[-1].predictions if children_results else None,
            confidences=children_results[-1].confidences if children_results else None,
            labels=children_results[-1].labels if children_results else None,
            metadata={"stages_completed": len(children_results)},
            children_results=children_results,
            execution_time_ms=(time.time() - start_time) * 1000
        )


class EnsembleNode(PipelineNode):
    """Node that combines predictions from multiple models."""

    def __init__(self, node_id: str, children: List[PipelineNode],
                 voting_method: str = "majority",
                 weights: Optional[List[float]] = None):
        super().__init__(node_id, NodeType.ENSEMBLE)
        self.children = children
        self.voting_method = voting_method  # "majority", "weighted", "confidence"
        self.weights = weights or [1.0] * len(children)

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        start_time = time.time()
        self.status = PipelineStatus.RUNNING

        children_results = []
        for child in self.children:
            result = child.execute(inputs, context)
            children_results.append(result)

        # Combine predictions
        all_predictions = [r.predictions for r in children_results if r.predictions is not None]
        all_confidences = [r.confidences for r in children_results if r.confidences is not None]

        if not all_predictions:
            self.status = PipelineStatus.FAILED
            return PipelineResult(
                node_id=self.node_id,
                status=self.status,
                metadata={"error": "No valid predictions from ensemble members"},
                children_results=children_results,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        # Stack and vote
        predictions_stack = np.stack(all_predictions)  # (n_models, n_samples)

        if self.voting_method == "majority":
            # Simple majority voting
            final_predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 0, predictions_stack
            )
            # Confidence = agreement ratio
            final_confidences = np.apply_along_axis(
                lambda x: np.bincount(x).max() / len(x), 0, predictions_stack
            )

        elif self.voting_method == "weighted":
            # Weighted voting
            n_classes = int(predictions_stack.max()) + 1
            weighted_votes = np.zeros((predictions_stack.shape[1], n_classes))
            for i, preds in enumerate(predictions_stack):
                for j, pred in enumerate(preds):
                    weighted_votes[j, int(pred)] += self.weights[i]
            final_predictions = np.argmax(weighted_votes, axis=1)
            final_confidences = np.max(weighted_votes, axis=1) / np.sum(self.weights)

        else:  # confidence-weighted
            confidences_stack = np.stack(all_confidences)
            n_classes = int(predictions_stack.max()) + 1
            confidence_votes = np.zeros((predictions_stack.shape[1], n_classes))
            for i, (preds, confs) in enumerate(zip(predictions_stack, confidences_stack)):
                for j, (pred, conf) in enumerate(zip(preds, confs)):
                    confidence_votes[j, int(pred)] += conf
            final_predictions = np.argmax(confidence_votes, axis=1)
            final_confidences = np.max(confidence_votes, axis=1) / len(all_predictions)

        self.status = PipelineStatus.COMPLETED

        # Get labels from first model that has them
        labels = None
        for result in children_results:
            if result.labels:
                model = self.children[children_results.index(result)]
                if hasattr(model, 'model') and hasattr(model.model, 'class_labels'):
                    labels = [model.model.class_labels[p] for p in final_predictions]
                    break

        return PipelineResult(
            node_id=self.node_id,
            status=self.status,
            predictions=final_predictions,
            confidences=final_confidences,
            labels=labels,
            metadata={
                "voting_method": self.voting_method,
                "ensemble_size": len(self.children),
                "agreement_rate": float(np.mean(final_confidences))
            },
            children_results=children_results,
            execution_time_ms=(time.time() - start_time) * 1000
        )


class ConditionalNode(PipelineNode):
    """Node that selects execution path based on conditions."""

    def __init__(self, node_id: str, condition_node: PipelineNode,
                 branches: Dict[str, PipelineNode],
                 default_branch: Optional[str] = None):
        super().__init__(node_id, NodeType.CONDITIONAL)
        self.condition_node = condition_node
        self.branches = branches
        self.default_branch = default_branch

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        start_time = time.time()
        self.status = PipelineStatus.RUNNING

        # Execute condition
        condition_result = self.condition_node.execute(inputs, context)

        if condition_result.status != PipelineStatus.COMPLETED:
            self.status = condition_result.status
            return PipelineResult(
                node_id=self.node_id,
                status=self.status,
                metadata={"error": "Condition evaluation failed"},
                children_results=[condition_result],
                execution_time_ms=(time.time() - start_time) * 1000
            )

        # Determine branch
        if condition_result.labels:
            branch_key = condition_result.labels[0]
        elif condition_result.predictions is not None:
            branch_key = str(int(condition_result.predictions[0]))
        else:
            branch_key = self.default_branch

        # Execute selected branch
        if branch_key in self.branches:
            branch_result = self.branches[branch_key].execute(inputs, context)
            self.status = branch_result.status
        elif self.default_branch and self.default_branch in self.branches:
            branch_result = self.branches[self.default_branch].execute(inputs, context)
            self.status = branch_result.status
        else:
            self.status = PipelineStatus.FAILED
            branch_result = None

        return PipelineResult(
            node_id=self.node_id,
            status=self.status,
            predictions=branch_result.predictions if branch_result else None,
            confidences=branch_result.confidences if branch_result else None,
            labels=branch_result.labels if branch_result else None,
            metadata={
                "selected_branch": branch_key,
                "condition_prediction": str(condition_result.predictions[0]) if condition_result.predictions is not None else None
            },
            children_results=[condition_result, branch_result] if branch_result else [condition_result],
            execution_time_ms=(time.time() - start_time) * 1000
        )


class HumanCheckpointNode(PipelineNode):
    """Node that requires human review before proceeding."""

    def __init__(self, node_id: str, question: str, options: List[str],
                 continue_on: List[str], timeout_seconds: Optional[int] = None,
                 default_action: Optional[str] = None,
                 review_handler: Optional[Callable[[HumanReviewRequest], HumanReviewResponse]] = None):
        super().__init__(node_id, NodeType.HUMAN_CHECKPOINT)
        self.question = question
        self.options = options
        self.continue_on = continue_on
        self.timeout_seconds = timeout_seconds
        self.default_action = default_action
        self.review_handler = review_handler
        self.pending_review: Optional[HumanReviewRequest] = None
        self.review_response: Optional[HumanReviewResponse] = None

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        start_time = time.time()
        self.status = PipelineStatus.AWAITING_HUMAN

        # Create review request
        self.pending_review = HumanReviewRequest(
            checkpoint_id=f"{self.node_id}_{int(time.time())}",
            node_id=self.node_id,
            question=self.question,
            options=self.options,
            context={
                "inputs": {k: str(type(v)) for k, v in inputs.items()},
                "pipeline_context": context
            },
            results_so_far=context.get("results_so_far", []),
            timeout_seconds=self.timeout_seconds,
            default_action=self.default_action
        )

        # If we have a handler, call it
        if self.review_handler:
            self.review_response = self.review_handler(self.pending_review)

            if self.review_response.decision in self.continue_on:
                self.status = PipelineStatus.COMPLETED
            else:
                self.status = PipelineStatus.CANCELLED

            return PipelineResult(
                node_id=self.node_id,
                status=self.status,
                metadata={
                    "question": self.question,
                    "decision": self.review_response.decision,
                    "notes": self.review_response.notes
                },
                human_review={
                    "request": {
                        "question": self.question,
                        "options": self.options
                    },
                    "response": {
                        "decision": self.review_response.decision,
                        "notes": self.review_response.notes,
                        "reviewer_id": self.review_response.reviewer_id
                    }
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )

        # No handler - return awaiting status
        return PipelineResult(
            node_id=self.node_id,
            status=self.status,
            metadata={
                "question": self.question,
                "options": self.options,
                "awaiting_input": True
            },
            human_review={
                "request": {
                    "checkpoint_id": self.pending_review.checkpoint_id,
                    "question": self.question,
                    "options": self.options
                }
            },
            execution_time_ms=(time.time() - start_time) * 1000
        )

    def provide_response(self, response: HumanReviewResponse) -> PipelineResult:
        """Provide the human review response and get result."""
        self.review_response = response

        if response.decision in self.continue_on:
            self.status = PipelineStatus.COMPLETED
        else:
            self.status = PipelineStatus.CANCELLED

        return PipelineResult(
            node_id=self.node_id,
            status=self.status,
            metadata={
                "question": self.question,
                "decision": response.decision,
                "notes": response.notes
            },
            human_review={
                "request": {
                    "question": self.question,
                    "options": self.options
                },
                "response": {
                    "decision": response.decision,
                    "notes": response.notes,
                    "reviewer_id": response.reviewer_id
                }
            },
            execution_time_ms=0.0
        )


class DataTransformNode(PipelineNode):
    """Node that transforms data between pipeline stages."""

    def __init__(self, node_id: str, transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        super().__init__(node_id, NodeType.DATA_TRANSFORM)
        self.transform_fn = transform_fn

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        start_time = time.time()
        self.status = PipelineStatus.RUNNING

        try:
            transformed = self.transform_fn(inputs)
            self.status = PipelineStatus.COMPLETED
            return PipelineResult(
                node_id=self.node_id,
                status=self.status,
                metadata={"transformed_keys": list(transformed.keys())},
                execution_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            self.status = PipelineStatus.FAILED
            return PipelineResult(
                node_id=self.node_id,
                status=self.status,
                metadata={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000
            )


class Pipeline:
    """Main pipeline orchestrator."""

    def __init__(self, pipeline_id: str, root_node: PipelineNode,
                 description: str = ""):
        self.pipeline_id = pipeline_id
        self.root_node = root_node
        self.description = description
        self.status = PipelineStatus.PENDING
        self.result: Optional[PipelineResult] = None
        self.context: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []

    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Execute the pipeline."""
        self.status = PipelineStatus.RUNNING
        self.context = context or {}
        self.context["results_so_far"] = []

        start_time = time.time()

        self.result = self.root_node.execute(inputs, self.context)
        self.status = self.result.status

        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "status": self.status.value,
            "execution_time_ms": (time.time() - start_time) * 1000
        })

        return self.result

    def get_pending_checkpoints(self) -> List[HumanReviewRequest]:
        """Get all pending human review checkpoints."""
        pending = []
        self._collect_pending_checkpoints(self.root_node, pending)
        return pending

    def _collect_pending_checkpoints(self, node: PipelineNode, pending: List):
        """Recursively collect pending checkpoints."""
        if isinstance(node, HumanCheckpointNode) and node.pending_review and not node.review_response:
            pending.append(node.pending_review)

        if hasattr(node, 'children'):
            for child in node.children:
                self._collect_pending_checkpoints(child, pending)
        if hasattr(node, 'condition_node'):
            self._collect_pending_checkpoints(node.condition_node, pending)
        if hasattr(node, 'branches'):
            for branch in node.branches.values():
                self._collect_pending_checkpoints(branch, pending)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "description": self.description,
            "status": self.status.value,
            "execution_history": self.execution_history,
            "result": self._result_to_dict(self.result) if self.result else None
        }

    def _result_to_dict(self, result: PipelineResult) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "node_id": result.node_id,
            "status": result.status.value,
            "predictions": result.predictions.tolist() if result.predictions is not None else None,
            "confidences": result.confidences.tolist() if result.confidences is not None else None,
            "labels": result.labels,
            "metadata": result.metadata,
            "execution_time_ms": result.execution_time_ms,
            "human_review": result.human_review,
            "children_results": [self._result_to_dict(c) for c in result.children_results]
        }


# ============================================================================
# Pre-built Pipeline Templates for LARUN.SPACE
# ============================================================================

def create_exoplanet_detection_pipeline(models: Dict[str, Any],
                                        human_review_handler: Optional[Callable] = None) -> Pipeline:
    """
    Create the main exoplanet detection pipeline per TRD-LARUN-2026-001.

    Pipeline Flow:
    1. [PARALLEL] Initial Analysis:
       - EXOPLANET-001: Transit detection
       - VSTAR-001: Variable star check
       - FLARE-001: Flare detection

    2. [HUMAN CHECKPOINT] Review initial results

    3. [CONDITIONAL] Based on detection:
       - If transit detected -> Continue to validation
       - If variable star -> Route to VSTAR analysis
       - If flare -> Flag and stop

    4. [SERIES] Validation Pipeline:
       - Vetting tests (odd-even, v-shape, secondary eclipse)
       - FPP calculation
       - Ensemble confirmation

    5. [HUMAN CHECKPOINT] Final validation review

    6. [OUTPUT] Detection result
    """

    # Stage 1: Parallel initial analysis
    initial_parallel = ParallelNode(
        "initial_analysis",
        children=[
            ModelNode("exoplanet_detection", models.get("EXOPLANET-001"), confidence_threshold=0.7),
            ModelNode("variable_star_check", models.get("VSTAR-001"), confidence_threshold=0.6),
            ModelNode("flare_check", models.get("FLARE-001"), confidence_threshold=0.8)
        ]
    )

    # Stage 2: Human checkpoint
    initial_review = HumanCheckpointNode(
        "initial_review",
        question="Review initial detection results. Do you want to proceed with validation?",
        options=["proceed", "modify_parameters", "reject", "flag_for_expert"],
        continue_on=["proceed", "modify_parameters"],
        timeout_seconds=3600,
        default_action="proceed",
        review_handler=human_review_handler
    )

    # Stage 3: Conditional routing (simplified for now)
    # In full implementation, this would route based on detection type

    # Stage 4: Validation ensemble
    validation_ensemble = EnsembleNode(
        "validation_ensemble",
        children=[
            ModelNode("validation_model_1", models.get("EXOPLANET-001"), confidence_threshold=0.8),
            # Add more validation models as needed
        ],
        voting_method="confidence"
    )

    # Stage 5: Final human review
    final_review = HumanCheckpointNode(
        "final_review",
        question="Review validation results. Confirm detection status:",
        options=["validated_planet", "likely_false_positive", "needs_followup", "inconclusive"],
        continue_on=["validated_planet", "needs_followup"],
        review_handler=human_review_handler
    )

    # Combine into series pipeline
    full_pipeline = SeriesNode(
        "full_detection_pipeline",
        children=[
            initial_parallel,
            initial_review,
            validation_ensemble,
            final_review
        ]
    )

    return Pipeline(
        "exoplanet_detection_v1",
        full_pipeline,
        description="Complete exoplanet detection and validation pipeline per TRD-LARUN-2026-001"
    )


def create_stellar_classification_pipeline(models: Dict[str, Any]) -> Pipeline:
    """
    Create pipeline for stellar classification.

    Pipeline Flow:
    1. [PARALLEL] Multi-aspect analysis:
       - SPECTYPE-001: Spectral type classification
       - VSTAR-001: Variability classification
       - ASTERO-001: Asteroseismology analysis

    2. [ENSEMBLE] Combine stellar characterization results

    3. [OUTPUT] Complete stellar profile
    """

    classification_parallel = ParallelNode(
        "stellar_classification",
        children=[
            ModelNode("spectral_type", models.get("SPECTYPE-001"), confidence_threshold=0.7),
            ModelNode("variability", models.get("VSTAR-001"), confidence_threshold=0.6),
            ModelNode("asteroseismology", models.get("ASTERO-001"), confidence_threshold=0.6)
        ]
    )

    return Pipeline(
        "stellar_classification_v1",
        classification_parallel,
        description="Multi-aspect stellar characterization pipeline"
    )


def create_transient_detection_pipeline(models: Dict[str, Any],
                                        human_review_handler: Optional[Callable] = None) -> Pipeline:
    """
    Create pipeline for transient event detection.

    Pipeline Flow:
    1. [PARALLEL] Multi-type detection:
       - SUPERNOVA-001: Supernova detection
       - FLARE-001: Flare detection
       - MICROLENS-001: Microlensing detection

    2. [CONDITIONAL] Route based on highest confidence detection

    3. [HUMAN CHECKPOINT] Expert review for significant detections

    4. [OUTPUT] Transient classification and alert
    """

    # Parallel detection
    transient_parallel = ParallelNode(
        "transient_detection",
        children=[
            ModelNode("supernova", models.get("SUPERNOVA-001"), confidence_threshold=0.75),
            ModelNode("flare", models.get("FLARE-001"), confidence_threshold=0.8),
            ModelNode("microlensing", models.get("MICROLENS-001"), confidence_threshold=0.7)
        ]
    )

    # Human review for alerts
    alert_review = HumanCheckpointNode(
        "alert_review",
        question="Transient detected. Review and confirm alert status:",
        options=["confirm_alert", "false_alarm", "need_more_data", "escalate_to_expert"],
        continue_on=["confirm_alert", "need_more_data"],
        review_handler=human_review_handler
    )

    pipeline_root = SeriesNode(
        "transient_pipeline",
        children=[transient_parallel, alert_review]
    )

    return Pipeline(
        "transient_detection_v1",
        pipeline_root,
        description="Multi-type transient event detection with alert system"
    )


# ============================================================================
# Pipeline Execution Utilities
# ============================================================================

class PipelineRunner:
    """Utility class for running and managing pipelines."""

    def __init__(self):
        self.pipelines: Dict[str, Pipeline] = {}
        self.execution_log: List[Dict[str, Any]] = []

    def register_pipeline(self, pipeline: Pipeline):
        """Register a pipeline for execution."""
        self.pipelines[pipeline.pipeline_id] = pipeline

    def execute(self, pipeline_id: str, inputs: Dict[str, Any]) -> PipelineResult:
        """Execute a registered pipeline."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline_id}")

        pipeline = self.pipelines[pipeline_id]
        result = pipeline.execute(inputs)

        self.execution_log.append({
            "pipeline_id": pipeline_id,
            "timestamp": datetime.now().isoformat(),
            "status": result.status.value,
            "execution_time_ms": result.execution_time_ms
        })

        return result

    def get_pending_reviews(self) -> Dict[str, List[HumanReviewRequest]]:
        """Get all pending human reviews across pipelines."""
        pending = {}
        for pipeline_id, pipeline in self.pipelines.items():
            checkpoints = pipeline.get_pending_checkpoints()
            if checkpoints:
                pending[pipeline_id] = checkpoints
        return pending


if __name__ == "__main__":
    # Demo: Create and run a simple pipeline
    print("Pipeline Framework Demo")
    print("=" * 50)

    # Import models (would be actual models in production)
    from specialized_models import get_model

    # Load models
    models = {
        "EXOPLANET-001": get_model("EXOPLANET-001"),
        "VSTAR-001": get_model("VSTAR-001"),
        "FLARE-001": get_model("FLARE-001"),
    }

    # Create simple human review handler (auto-approve for demo)
    def auto_approve_handler(request: HumanReviewRequest) -> HumanReviewResponse:
        print(f"\n[HUMAN CHECKPOINT] {request.question}")
        print(f"  Options: {request.options}")
        print(f"  Auto-selecting: {request.options[0]}")
        return HumanReviewResponse(
            checkpoint_id=request.checkpoint_id,
            decision=request.options[0],
            notes="Auto-approved for demo"
        )

    # Create pipeline
    pipeline = create_exoplanet_detection_pipeline(models, auto_approve_handler)

    # Generate test data
    test_data = np.random.randn(1, 1024, 1).astype(np.float32)

    # Execute
    print("\nExecuting pipeline...")
    result = pipeline.execute({"data": test_data})

    print(f"\nPipeline Status: {result.status.value}")
    print(f"Execution Time: {result.execution_time_ms:.2f}ms")
    print(f"Children Results: {len(result.children_results)}")

    # Show result structure
    print("\nResult Structure:")
    print(json.dumps(pipeline.to_dict(), indent=2, default=str))
