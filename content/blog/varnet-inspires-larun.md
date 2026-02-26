# What a High Schooler's 1.5M Object Discovery Means for Space Democratization

*February 2026 — larun.space*

---

## A Teenager Did What Caltech Couldn't

In late 2024, Matteo Paz — a high school student — published a paper in *The Astronomical Journal*
that stopped the astronomy community in its tracks. Working from a laptop, without a telescope or
institutional resources, he ran a small AI model called **VARnet** across NASA's entire NEOWISE
archive and found **1.5 million previously unknown space objects**.

Not 1,500. Not 150,000. One point five million.

His model used Fourier transforms and wavelet analysis — mathematical tools invented decades ago —
combined with a lightweight convolutional neural network. Total model size: comparable to a
high-quality JPEG photo. GPU required: none (inference under 1 millisecond per source).

This is the most important result in observational astronomy in years, and it happened because
the right small model was applied to the right large dataset.

---

## The Key Insight: Specialized Small Models Beat Brute Force

Here's what VARnet did NOT do:
- It did not use a large language model
- It did not require GPU-heavy inference
- It did not need fine-tuned foundation models

What it DID do:
- Extract a small set of physically meaningful features (Fourier peak frequencies, wavelet energy distributions)
- Train a focused 4-class classifier: non-variable, transient, pulsator, eclipsing
- Apply this 10,000× across 450 million NEOWISE sources

The result: 1.9 million candidates, 1.5 million confirmed new variables.

This is exactly what larun.space was built to prove: **a federation of tiny, specialized AI models
is more powerful than one large general model** for scientific discovery tasks.

---

## How larun.space Embodies the Same Philosophy

We launched larun.space before Paz's paper was published, but he validated our thesis exactly.

Our platform now has 12 specialized TinyML models:

| Model | Purpose | Size |
|-------|---------|------|
| EXOPLANET-001 | Transit detection | 43 KB |
| VSTAR-001 | Variable star classification | 27 KB |
| FLARE-001 | Stellar flare detection | 5 KB |
| VARDET-001 | VARnet-inspired variability detector | ~50 KB |
| ANOMALY-001 | Catch objects no other model recognizes | ~20 KB |
| ... | 8 more specialized models | ... |

None of these models require a GPU. All run in under 500ms. All together, they provide deeper
characterization than any single large model.

---

## The Citizen Discovery Engine

Starting today, larun.space users can do what Paz did — systematically search NASA archives
for unknown objects.

The **Citizen Discovery Engine** (available at larun.space/discover):

1. Select a sky region using an interactive star map
2. Choose your data source: TESS, Kepler, or NEOWISE
3. Run all 12 TinyML models automatically
4. Cross-match results against 6+ catalogs
5. Submit candidates for community verification
6. Get permanently credited for confirmed discoveries

The difference from VARnet: while Paz classified objects into 4 categories, larun.space provides
12 specialized models — adding asteroseismology, microlensing, flare detection, period finding,
and anomaly detection that VARnet didn't cover.

---

## "Federation of TinyML" — Why 12 Small Models > 1 Large Model

| Factor | 12 TinyML Models | 1 Large Language Model (7B+) |
|--------|-----------------|-------------------------------|
| Inference cost | ~$0/month (CPU) | $500-2000/month (GPU) |
| Latency | <500ms | 2-10 seconds |
| Astronomy accuracy | 95-100% per domain | 70-85% general |
| Scalable | Linear with users | GPU-bound |
| Offline capable | ✓ | ✗ |

Claude API is used on larun.space for exactly three things: parsing natural language queries,
generating readable reports from model outputs, and explaining anomalies in astrophysical context.
Not for inference. Not for classification.

---

## Call to Action

If a high schooler with a laptop can find 1.5 million unknown objects in NASA data, imagine
what a community of curious students, amateur astronomers, and researchers worldwide could find.

**Start discovering at [larun.space/discover](https://larun.space/discover).**

Free tier: 5 discovery runs per month. No PhD required. No GPU required. Just curiosity.

---

*Larun Engineering LLP — "Federation of TinyML for Space Science"*
*larun.space | sattrack.larun.space*
