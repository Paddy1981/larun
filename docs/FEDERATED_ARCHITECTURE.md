# Federated Larun TinyML Architecture

## Vision
A network of specialized TinyML nodes coordinated by a central "Grand Larun" that:
- Receives detections from edge nodes
- Aggregates and validates signals
- Shares learned patterns back to nodes
- Enables distributed exoplanet hunting

## Architecture

```
                    ┌─────────────────────┐
                    │   GRAND LARUN       │
                    │   (Central Hub)     │
                    │   - Aggregation     │
                    │   - Validation      │
                    │   - Model Updates   │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
    │  TinyML-1   │     │  TinyML-2   │     │  TinyML-N   │
    │  Transit    │     │  Anomaly    │     │  Variable   │
    │  Detection  │     │  Detection  │     │  Star Class │
    └─────────────┘     └─────────────┘     └─────────────┘
```

## Node Types

| Node | TFLite Model | Purpose | Output |
|------|--------------|---------|--------|
| **Transit** | `transit_detector.tflite` | Detect dips | Period, depth, confidence |
| **Anomaly** | `anomaly_detector.tflite` | Flag unusual patterns | Anomaly score |
| **Variable** | `variable_classifier.tflite` | Classify star type | Star class |
| **Quality** | `quality_scorer.tflite` | Data quality check | Quality score |

## Communication Protocol

```python
# Message format between nodes
{
    "node_id": "transit-001",
    "timestamp": "2026-01-31T01:00:00Z",
    "target": "TIC 307210830",
    "detection": {
        "type": "transit_candidate",
        "confidence": 0.92,
        "period": 3.5,
        "depth_ppm": 1200
    }
}
```

## Implementation Phases

### Phase 1: Specialized TFLite Models
- [ ] Train separate models for each node type
- [ ] Optimize for <100KB each

### Phase 2: Node Communication
- [ ] MQTT or REST protocol
- [ ] Message queue for offline nodes

### Phase 3: Grand Larun Coordinator
- [ ] Aggregation service
- [ ] Cross-validation between nodes
- [ ] Federated learning updates

## Status
**Approved**: 2026-01-31
**Priority**: Future Sprint
