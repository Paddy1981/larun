# LARUN TinyML - Installation Guide

Complete installation instructions for LARUN, the TinyML-powered astronomical data analysis system.

## Prerequisites

| Requirement | Minimum Version | Check Command |
|-------------|-----------------|---------------|
| Python | 3.8 | `python --version` |
| pip | 20.0 | `pip --version` |
| git | (optional) | `git --version` |

### Installing Python

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **IMPORTANT:** Check "Add Python to PATH" during installation

**macOS:**
```bash
brew install python
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

---

## Quick Install

### One-Command Installation

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/Paddy1981/larun/main/install.ps1 | iex
```

**Linux/macOS (Bash):**
```bash
curl -fsSL https://raw.githubusercontent.com/Paddy1981/larun/main/install.sh | bash
```

### Using pip

```bash
pip install larun
```

### Using pipx (Recommended for CLI tools)

```bash
pipx install larun
```

---

## Install from Source

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/Paddy1981/larun.git
cd larun

# Install
pip install .
```

### Development Installation

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/Paddy1981/larun.git
cd larun

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Verify Installation

After installation, verify everything works:

```bash
# Check CLI is available
larun --help

# Start interactive CLI
larun

# Start chat interface
larun-chat
```

Inside the LARUN CLI, type:
```
/status
/skills
/help
```

---

## Platform-Specific Notes

### Windows

- Use PowerShell (not Command Prompt) for best experience
- If `larun` command is not found, restart your terminal
- Python Scripts folder should be in PATH: `C:\Users\<USER>\AppData\Local\Programs\Python\Python3X\Scripts`

### macOS

- If using Apple Silicon (M1/M2), TensorFlow should work natively
- Use `python3` instead of `python` if needed

### Linux

- Some astronomical packages may need additional system libraries:
  ```bash
  sudo apt install libffi-dev libssl-dev
  ```

---

## GPU Support (Optional)

For faster model training with GPU:

**NVIDIA GPU:**
```bash
pip install larun[gpu]
# OR
pip install tensorflow[and-cuda]
```

**Apple Silicon (M1/M2):**
TensorFlow Metal support is included automatically.

---

## Troubleshooting

### "larun" command not found

1. Ensure Python Scripts directory is in your PATH
2. Restart your terminal/shell
3. Try running with: `python -m larun`

### TensorFlow installation issues

```bash
# Install TensorFlow separately first
pip install tensorflow

# Then install larun
pip install larun
```

### Permission errors

```bash
# Linux/macOS: Use user install
pip install --user larun

# Windows: Run PowerShell as Administrator
```

### Network/proxy issues

```bash
pip install larun --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

---

## Uninstall

```bash
pip uninstall larun
```

To remove all data and models:
```bash
# Linux/macOS
rm -rf ~/.larun

# Windows
rmdir /s %USERPROFILE%\.larun
```

---

## Next Steps

After installation:

1. **Start LARUN:** `larun`
2. **View available skills:** `/skills`
3. **Train the model:** `/train`
4. **Fetch NASA data:** `/fetch Kepler-186`
5. **Detect transits:** `/detect`

For more information, see the [README](README.md) and [Documentation](docs/).
