# LARUN Quick Start Guide

Get from zero to exoplanet detection in 5 minutes.

---

## 1. Install (1 minute)

```bash
# Clone repository
git clone https://github.com/Paddy1981/larun.git
cd larun

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Start LARUN (30 seconds)

```bash
# Interactive CLI
python larun.py
```

You'll see:

```
╔══════════════════════════════════════════════════════════════════════════╗
║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗                          ║
║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║                          ║
║     ...                                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝

LARUN> _
```

---

## 3. Fetch Your First Target (1 minute)

```
LARUN> /fetch TIC 307210830
```

This downloads TESS light curve data for TOI-700, a known exoplanet host.

Output:
```
Fetching data for TIC 307210830...
Found 4 sectors of TESS data
Downloaded 45,000 data points
Light curve saved to: data/TIC_307210830.fits
```

---

## 4. Detect Transits (1 minute)

```
LARUN> /detect
```

Output:
```
Running transit detection...

Results:
  Candidates found: 3

  #1: Period = 9.977 days, Depth = 824 ppm, SNR = 12.3
  #2: Period = 37.42 days, Depth = 612 ppm, SNR = 8.7
  #3: Period = 16.05 days, Depth = 445 ppm, SNR = 7.2

Detection complete!
```

---

## 5. Analyze a Candidate (1 minute)

### Run BLS Periodogram

```
LARUN> /bls
```

### Fit Transit Model

```
LARUN> /fit
```

### Check for False Positives

```
LARUN> /vet
```

---

## 6. Generate Report (30 seconds)

```
LARUN> /report
```

Creates a PDF report at `output/TIC_307210830_report.pdf`

---

## Alternative: Chat Mode

For natural language interaction:

```bash
python larun_chat.py
```

Then ask questions like:
- "Search for transits in Kepler-11"
- "What's the period of the detected planet?"
- "Is this a false positive?"

---

## Common Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/skills` | List available analysis skills |
| `/fetch <target>` | Download NASA data |
| `/detect` | Run transit detection |
| `/bls` | BLS periodogram analysis |
| `/phase` | Phase fold light curve |
| `/fit` | Fit transit model |
| `/vet` | Vetting for false positives |
| `/stellar <teff>` | Classify star |
| `/planet` | Calculate planet properties |
| `/report` | Generate PDF report |
| `/quit` | Exit LARUN |

---

## Example Targets

Try these known exoplanet systems:

| Target | System | What You'll Find |
|--------|--------|------------------|
| `TIC 307210830` | TOI-700 | 3 Earth-sized planets |
| `TIC 261136679` | TOI-1338 | Circumbinary planet |
| `KIC 10666592` | Kepler-11 | 6 transiting planets |
| `TIC 441462736` | TOI-1233 | 4 planet system |

---

## Next Steps

- [Training Guide](TRAINING_GUIDE.md) - Train your own model
- [API Documentation](../api.py) - Use the REST API
- [Skills Reference](../SKILLS_ROADMAP.md) - All 24+ analysis skills

---

## Troubleshooting

### "No data found"

```
LARUN> /fetch TIC 123456789
Error: No TESS data available for this target
```

**Solution**: Try a different target or check the TIC ID.

### "Model not found"

```
Error: Model file not found at models/real/astro_tinyml.tflite
```

**Solution**: Train the model first:
```bash
python train_real_data.py --planets 100 --non-planets 100 --epochs 50
```

### Slow Data Download

NASA archives can be slow. Use cached data:
```
LARUN> /fetch --use-cache
```

---

## Get Help

- `/help` in LARUN CLI
- [GitHub Issues](https://github.com/Paddy1981/larun/issues)
- [Documentation](../README.md)

---

**You're ready to discover exoplanets!**
