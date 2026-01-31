# LARUN.SPACE Framework Architecture

## The Vinveli-Vinoli-Vidhai Framework

LARUN is built on the **Vinveli-Vinoli-Vidhai** three-tier architecture, inspired by Tamil words and astrophysics principles.

---

## Framework Overview

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                    ∙  ✦  ·  ✧  ∙  ✦  ·  ✧  ∙  ✦                          ║
║               ✧        V I N V E L I          ✧                          ║
║                    The Cosmos · The Everything                            ║
║            ∙  ✦  ·  ✧  ∙  ✦  ·  ✧  ∙  ✦  ·  ✧  ∙                        ║
║                              │                                            ║
║                              │ Light travels through                      ║
║                              ▼                                            ║
║                  ════════════════════════════                             ║
║                       V I N O L I                                         ║
║                  Speed of light — bent by gravity                         ║
║                  ════════════════════════════                             ║
║                              │                                            ║
║                              │ Carries seeds across                       ║
║                              ▼                                            ║
║                  ┌─────────────────────────┐                              ║
║                  │  V I D H A I  · · · ·   │                              ║
║                  │  Seeds planted across   │                              ║
║                  │  the cosmos, harvesting │                              ║
║                  │  knowledge from stars   │                              ║
║                  └─────────────────────────┘                              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## The Three Layers

### VINVELI (விண்வெளி - The Cosmos)

**Persona:** The Infinite Expanse
**Tagline:** *"The cosmos. The everything."*

```
╔══════════════════════════════════════════════════════════════════════════╗
║  ██╗   ██╗██╗███╗   ██╗██╗   ██╗███████╗██╗     ██╗                      ║
║  ██║   ██║██║████╗  ██║██║   ██║██╔════╝██║     ██║                      ║
║  ██║   ██║██║██╔██╗ ██║██║   ██║█████╗  ██║     ██║                      ║
║  ╚██╗ ██╔╝██║██║╚██╗██║╚██╗ ██╔╝██╔══╝  ██║     ██║                      ║
║   ╚████╔╝ ██║██║ ╚████║ ╚████╔╝ ███████╗███████╗██║                      ║
║    ╚═══╝  ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚══════╝╚══════╝╚═╝                      ║
║                                                                          ║
║     விண்வெளி • The cosmos. The everything.                               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Role:** Central System Layer
- The ALL - contains everything
- Infinite, boundless, ever-expanding
- All data, all models, all knowledge exists within it
- The grand orchestrator of the universe
- Heavy with the mass of all information

**Components:**
- API Server (`api.py`)
- Training Pipeline (`train_*.py`)
- Model Distribution
- Dashboard (`dashboard.html`)
- Data Pipeline (`src/pipeline/`)
- Report Generator (`src/reporter/`)

---

### VINOLI (வெளிச்சம் - Light)

**Persona:** The Messenger Photon
**Tagline:** *"Speed of light — until gravity bends the path"*

```
╔══════════════════════════════════════════════════════════════════════════╗
║  ██╗   ██╗██╗███╗   ██╗ ██████╗ ██╗     ██╗                              ║
║  ██║   ██║██║████╗  ██║██╔═══██╗██║     ██║                              ║
║  ██║   ██║██║██╔██╗ ██║██║   ██║██║     ██║                              ║
║  ╚██╗ ██╔╝██║██║╚██╗██║██║   ██║██║     ██║                              ║
║   ╚████╔╝ ██║██║ ╚████║╚██████╔╝███████╗██║                              ║
║    ╚═══╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚═╝                              ║
║                                                                          ║
║     வெளிச்சம் • Speed of light — until gravity bends the path            ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Role:** Communication Layer
- Swift, bright, illuminating
- Travels at speed of light for simple queries
- **But heavy data bends space-time** → light takes a curved path → arrives later
- User sees: "Processing heavy data..." (gravitational lensing in action)
- Respects the laws of physics

**Components:**
- LARUN CLI (`larun.py`)
- Chat Interface (`larun_chat.py`)
- Skills Engine (`src/skills/`)
- Detector (`src/detector/`)
- Calibration (`src/calibration/`)

**The Gravitational Lensing Model:**
```
    Light Query (fast)              Heavy Data Query (bent path)
    ════════════════                ════════════════════════════

    User ───────────→ Result        User ─────╮
         (straight)                           │
                                       ╭──────┴──────╮
                                      ╱   VINVELI    ╲  ← Heavy Data
                                     │   (Gravity)    │    (Mass)
                                      ╲              ╱
                                       ╰──────┬──────╯
                                              │
                                    Result ←──╯
                                    (curved path = delay)

    "The heavier the data, the more space-time curves,
     and the longer light takes to reach you."
```

---

### VIDHAI (விதை - Seed)

**Persona:** The Cosmic Spore
**Tagline:** *"Plant a seed. Harvest the stars."*

```
╔══════════════════════════════════════════════════════════════════════════╗
║  ██╗   ██╗██╗██████╗ ██╗  ██╗ █████╗ ██╗                                 ║
║  ██║   ██║██║██╔══██╗██║  ██║██╔══██╗██║                                 ║
║  ██║   ██║██║██║  ██║███████║███████║██║                                 ║
║  ╚██╗ ██╔╝██║██║  ██║██╔══██║██╔══██║██║                                 ║
║   ╚████╔╝ ██║██████╔╝██║  ██║██║  ██║██║                                 ║
║    ╚═══╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝                                 ║
║                                                                          ║
║     விதை • Plant a seed. Harvest the stars.                              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Role:** Edge Deployment Layer
- Small but mighty, grows into discovery
- Seeds planted on user systems
- Harvests knowledge from starlight
- Resilient, distributed, autonomous

**Components:**
- TinyML Model (`models/real/astro_tinyml.tflite`)
- INT8 Quantized Model (`models/real/astro_tinyml_int8.tflite`)
- Edge Inference (`src/model/inference.py`)
- Notebooks (`notebooks/*.ipynb`)
- Distributed Training (`distributed/`)

**The Seed Message:**
> **VIDHAI (விதை)** means **"Seed"** in Tamil.
>
> When you install LARUN on your device, you plant a **seed** of astronomical
> intelligence. This seed:
>
> - **Grows** on your device (ESP32, Raspberry Pi, laptop, or cloud)
> - **Observes** light curves from NASA telescopes
> - **Detects** planetary transits, stellar anomalies, and cosmic events
> - **Harvests** knowledge - every detection is a discovery waiting to bloom
> - **Shares** findings back to the collective LARUN network
>
> Together, thousands of Vidhai seeds planted across the world form a
> **distributed garden of space discovery**.

---

## Persona Summary

| Layer | Tamil | Meaning | Persona | Tagline |
|-------|-------|---------|---------|---------|
| **VINVELI** | விண்வெளி | Cosmos | The Everything | *"The cosmos. The everything."* |
| **VINOLI** | வெளிச்சம் | Light | The Messenger | *"Speed of light — until gravity bends the path"* |
| **VIDHAI** | விதை | Seed | The Harvester | *"Plant a seed. Harvest the stars."* |

---

## Component Mapping

| Layer | Components | Files |
|-------|------------|-------|
| **VINVELI** | API, Training, Dashboard | `api.py`, `train_*.py`, `dashboard.html`, `src/pipeline/` |
| **VINOLI** | CLI, Chat, Skills | `larun.py`, `larun_chat.py`, `src/skills/`, `src/detector/` |
| **VIDHAI** | Models, Edge Inference | `models/*.tflite`, `src/model/`, `notebooks/` |

---

## Usage in Code

When each layer is invoked, display its ASCII art:

```python
# When API server starts
print(VINVELI_ASCII)  # "The cosmos. The everything."

# When CLI starts
print(LARUN_ASCII)    # Main product branding

# When model is deployed to edge
print(VIDHAI_ASCII)   # "Plant a seed. Harvest the stars."

# When heavy processing occurs
print("Processing... light bending around data gravity...")
```

---

## The Physics Behind It

The framework draws from real astrophysics:

1. **Space-Time (Vinveli)** - The fabric of the cosmos that contains all matter and energy
2. **Light (Vinoli)** - Travels at 299,792 km/s but bends around massive objects (gravitational lensing)
3. **Seeds (Vidhai)** - Like cosmic dust that seeds galaxies, our TinyML models seed discovery

When users experience delays during heavy processing, they understand:
> "Light is bending around the gravity of this data."

---

*"Plant a seed. Harvest the stars. Discover new worlds."*

---

**LARUN.SPACE** × Federation of TinyML for Space Science × Built on the Vinveli-Vinoli-Vidhai Framework

**Website**: [larun.space](https://larun.space)
