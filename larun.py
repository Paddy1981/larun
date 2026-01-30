#!/usr/bin/env python3
"""
LARUN TinyML - Interactive CLI with Skills System
==================================================
Interactive terminal interface for AstroTinyML with dynamic skill loading.

TinyML-powered astronomical data analysis for exoplanet discovery.

Usage:
    python larun.py              # Start interactive mode
    python larun.py --help       # Show help

Skills Commands:
    /skills         - List all available skills
    /skill <ID>     - Show skill details
    /run <ID>       - Execute a skill

Developer Commands:
    /addon          - List/load developer addons
    /generate       - Generate Python scripts and ML models

Created by: Padmanaban Veeraragavalu (Larun Engineering)
With AI assistance from: Claude (Anthropic)

Project: LARUN TinyML × Astrodata
License: MIT
"""

import sys
import json
import time
import readline  # For command history
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "2.0.0"
MODEL_PATH = Path("models/real/astro_tinyml.h5")
DATA_PATH = Path("data/real/training_data.npz")
SKILLS_PATH = Path("skills/skills.yaml")

CONFIG = {
    "num_planets": 50,
    "num_epochs": 100,
    "input_size": 1024,
    "num_classes": 6,
    "batch_size": 16,
    "min_snr": 7.0,
}

CLASS_NAMES = ["noise", "stellar", "transit", "binary", "artifact", "unknown"]

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
# SKILL LOADER
# ============================================================================

class Skill:
    """Represents a single skill from skills.yaml"""
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get('id', 'UNKNOWN')
        self.name = data.get('name', 'Unknown Skill')
        self.category = data.get('category', 'misc')
        self.tier = data.get('tier', 1)
        self.status = data.get('status', 'planned')
        self.description = data.get('description', '')
        self.command = data.get('command', '')
        self.dependencies = data.get('dependencies', [])
        self.inputs = data.get('inputs', [])
        self.outputs = data.get('outputs', [])
        self.examples = data.get('examples', [])
        self.metrics = data.get('metrics', [])

    @property
    def is_active(self) -> bool:
        return self.status == 'active'

    @property
    def status_icon(self) -> str:
        icons = {
            'active': f'{Colors.GREEN}●{Colors.END}',
            'partial': f'{Colors.YELLOW}◐{Colors.END}',
            'planned': f'{Colors.DIM}○{Colors.END}',
            'deprecated': f'{Colors.RED}✗{Colors.END}'
        }
        return icons.get(self.status, '?')

    @property
    def tier_color(self) -> str:
        colors = {
            1: Colors.RED,
            2: Colors.YELLOW,
            3: Colors.GREEN,
            4: Colors.BLUE,
            5: Colors.MAGENTA,
            6: Colors.CYAN,
            7: Colors.HEADER
        }
        return colors.get(self.tier, Colors.DIM)


class SkillLoader:
    """Loads and manages skills from skills.yaml"""

    def __init__(self, skills_path: Path = SKILLS_PATH):
        self.skills_path = skills_path
        self.skills: Dict[str, Skill] = {}
        self.metadata: Dict[str, Any] = {}
        self.loaded = False
        self._load()

    def _load(self):
        """Load skills from YAML file"""
        if not YAML_AVAILABLE:
            return

        if not self.skills_path.exists():
            return

        try:
            with open(self.skills_path) as f:
                data = yaml.safe_load(f)

            self.metadata = data.get('metadata', {})

            for skill_data in data.get('skills', []):
                skill = Skill(skill_data)
                self.skills[skill.id] = skill

            self.loaded = True
        except Exception as e:
            print(f"{Colors.RED}Error loading skills: {e}{Colors.END}")

    def get(self, skill_id: str) -> Optional[Skill]:
        """Get skill by ID"""
        return self.skills.get(skill_id.upper())

    def list_by_category(self, category: str = None) -> List[Skill]:
        """List skills, optionally filtered by category"""
        skills = list(self.skills.values())
        if category:
            skills = [s for s in skills if s.category == category]
        return sorted(skills, key=lambda s: (s.tier, s.id))

    def list_by_tier(self, tier: int) -> List[Skill]:
        """List skills by tier"""
        return [s for s in self.skills.values() if s.tier == tier]

    def list_active(self) -> List[Skill]:
        """List only active skills"""
        return [s for s in self.skills.values() if s.is_active]

    def categories(self) -> List[str]:
        """Get unique categories"""
        return sorted(set(s.category for s in self.skills.values()))

    def check_dependencies(self, skill: Skill) -> List[str]:
        """Check which dependencies are missing"""
        missing = []
        for dep in skill.dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        return missing


# Global skill loader
skill_loader = SkillLoader()

# ============================================================================
# ADDON SYSTEM
# ============================================================================

ADDONS_PATH = Path("addons")
ADDONS_SKILLS_PATH = Path("skills/addons")

class AddonLoader:
    """Loads optional addon modules for developers."""

    def __init__(self):
        self.loaded_addons: Dict[str, Any] = {}
        self.addon_skills: Dict[str, Skill] = {}

    def list_available(self) -> List[str]:
        """List available addons."""
        available = []

        # Check Python modules in addons/
        if ADDONS_PATH.exists():
            for f in ADDONS_PATH.glob("*.py"):
                if not f.name.startswith("_"):
                    available.append(f.stem)

        return available

    def load(self, addon_name: str) -> bool:
        """Load an addon by name."""
        if addon_name in self.loaded_addons:
            return True  # Already loaded

        addon_path = ADDONS_PATH / f"{addon_name}.py"
        if not addon_path.exists():
            return False

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(addon_name, addon_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self.loaded_addons[addon_name] = module

            # Load addon skills if YAML exists
            addon_skills_path = ADDONS_SKILLS_PATH / f"{addon_name}.yaml"
            if addon_skills_path.exists() and YAML_AVAILABLE:
                self._load_addon_skills(addon_skills_path)

            return True
        except Exception as e:
            print(f"{Colors.RED}Error loading addon {addon_name}: {e}{Colors.END}")
            return False

    def _load_addon_skills(self, path: Path):
        """Load skills from addon YAML."""
        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            for skill_data in data.get('skills', []):
                skill = Skill(skill_data)
                self.addon_skills[skill.id] = skill
        except Exception:
            pass

    def get_addon(self, name: str):
        """Get loaded addon module."""
        return self.loaded_addons.get(name)

    def is_loaded(self, name: str) -> bool:
        """Check if addon is loaded."""
        return name in self.loaded_addons


# Global addon loader
addon_loader = AddonLoader()

# ============================================================================
# UTILITIES
# ============================================================================

def print_banner():
    print(f"""
{Colors.BOLD}╔══════════════════════════════════════════════════════════════════════╗
║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗                      ║
║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║                      ║
║     ██║     ███████║██████╔╝██║   ██║██╔██╗ ██║                      ║
║     ██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║                      ║
║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║                      ║
║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                      ║
║                                                                      ║
║     {Colors.CYAN}TinyML for Astronomical Data Analysis{Colors.BOLD}                         ║
║     {Colors.DIM}v{VERSION} • {len(skill_loader.skills)} skills loaded{Colors.BOLD}                                    ║
╚══════════════════════════════════════════════════════════════════════╝{Colors.END}
""")

def print_prompt():
    return f"{Colors.GREEN}larun{Colors.END} {Colors.DIM}>{Colors.END} "

def print_success(message):
    print(f"{Colors.GREEN}✓{Colors.END} {message}")

def print_error(message):
    print(f"{Colors.RED}✗{Colors.END} {message}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ{Colors.END} {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠{Colors.END} {message}")

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class LarunState:
    def __init__(self):
        self.model = None
        self.data = None
        self.history = []
        self.last_results = None

    def load_model(self):
        if MODEL_PATH.exists():
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(MODEL_PATH))
            return True
        return False

    def load_data(self):
        if DATA_PATH.exists():
            import numpy as np
            self.data = np.load(str(DATA_PATH))
            return True
        return False

state = LarunState()

# ============================================================================
# SKILL COMMANDS
# ============================================================================

def cmd_skills(args):
    """List all available skills"""
    if not skill_loader.loaded:
        print_error("Skills not loaded. Make sure skills/skills.yaml exists.")
        print_info("Install PyYAML: pip install pyyaml")
        return

    # Parse arguments
    category_filter = None
    tier_filter = None
    show_all = '--all' in args

    for i, arg in enumerate(args):
        if arg == '--category' and i + 1 < len(args):
            category_filter = args[i + 1]
        if arg == '--tier' and i + 1 < len(args):
            try:
                tier_filter = int(args[i + 1])
            except:
                pass

    print(f"\n{Colors.BOLD}LARUN Skills{Colors.END}")
    print(f"{'─' * 60}")

    # Group by category
    categories = skill_loader.categories()

    for category in categories:
        if category_filter and category != category_filter:
            continue

        skills = skill_loader.list_by_category(category)
        if tier_filter:
            skills = [s for s in skills if s.tier == tier_filter]
        if not show_all:
            skills = [s for s in skills if s.is_active or s.status == 'partial']

        if not skills:
            continue

        print(f"\n{Colors.BOLD}{category.upper()}{Colors.END}")

        for skill in skills:
            tier_badge = f"{skill.tier_color}T{skill.tier}{Colors.END}"
            print(f"  {skill.status_icon} {tier_badge} {Colors.CYAN}{skill.id:12}{Colors.END} {skill.name}")

    # Legend
    print(f"\n{Colors.DIM}Legend: ● active  ◐ partial  ○ planned{Colors.END}")
    print(f"{Colors.DIM}Use /skill <ID> for details, /run <ID> to execute{Colors.END}")
    print(f"{Colors.DIM}Use /skills --all to show planned skills{Colors.END}\n")


def cmd_skill(args):
    """Show details for a specific skill"""
    if not args:
        print_error("Usage: /skill <SKILL_ID>")
        print_info("Example: /skill DATA-001")
        return

    skill_id = args[0].upper()
    skill = skill_loader.get(skill_id)

    if not skill:
        print_error(f"Skill not found: {skill_id}")
        print_info("Use /skills to list available skills")
        return

    # Header
    print(f"\n{Colors.BOLD}{'═' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{skill.name}{Colors.END}")
    print(f"{Colors.DIM}{skill.id} • Tier {skill.tier} • {skill.status}{Colors.END}")
    print(f"{'═' * 60}")

    # Description
    print(f"\n{skill.description}")

    # Command
    if skill.command:
        print(f"\n{Colors.BOLD}Command:{Colors.END}")
        print(f"  {Colors.CYAN}{skill.command}{Colors.END}")

    # Inputs
    if skill.inputs:
        print(f"\n{Colors.BOLD}Inputs:{Colors.END}")
        for inp in skill.inputs:
            name = inp.get('name', '?')
            inp_type = inp.get('type', 'any')
            default = inp.get('default', '')
            required = inp.get('required', False)
            desc = inp.get('description', '')

            req_badge = f"{Colors.RED}*{Colors.END}" if required else " "
            default_str = f" = {default}" if default else ""

            print(f"  {req_badge} {Colors.GREEN}{name}{Colors.END}: {inp_type}{default_str}")
            if desc:
                print(f"      {Colors.DIM}{desc}{Colors.END}")

    # Outputs
    if skill.outputs:
        print(f"\n{Colors.BOLD}Outputs:{Colors.END}")
        for out in skill.outputs:
            out_type = out.get('type', 'any')
            formats = out.get('format', [])
            print(f"  → {out_type} [{', '.join(formats)}]")

    # Dependencies
    if skill.dependencies:
        print(f"\n{Colors.BOLD}Dependencies:{Colors.END}")
        missing = skill_loader.check_dependencies(skill)
        for dep in skill.dependencies:
            if dep in missing:
                print(f"  {Colors.RED}✗{Colors.END} {dep} (not installed)")
            else:
                print(f"  {Colors.GREEN}✓{Colors.END} {dep}")

    # Examples
    if skill.examples:
        print(f"\n{Colors.BOLD}Examples:{Colors.END}")
        for ex in skill.examples:
            print(f"  {Colors.DIM}${Colors.END} {ex}")

    print()


def cmd_run(args):
    """Execute a skill"""
    if not args:
        print_error("Usage: /run <SKILL_ID> [--arg value ...]")
        print_info("Example: /run DATA-001 --source tess --target 'Kepler-186'")
        return

    skill_id = args[0].upper()
    skill = skill_loader.get(skill_id)

    if not skill:
        print_error(f"Skill not found: {skill_id}")
        return

    if not skill.is_active:
        print_warning(f"Skill {skill_id} is not active (status: {skill.status})")
        print_info("This skill is planned for future implementation")
        return

    # Check dependencies
    missing = skill_loader.check_dependencies(skill)
    if missing:
        print_error(f"Missing dependencies: {', '.join(missing)}")
        print_info(f"Install with: pip install {' '.join(missing)}")
        return

    # Parse skill arguments
    skill_args = args[1:]

    print(f"\n{Colors.BOLD}Executing: {skill.name}{Colors.END}")
    print(f"{'─' * 40}")

    # Route to appropriate handler
    handlers = {
        'DATA-001': execute_data_001,
        'DATA-002': execute_data_002,
        'DATA-003': execute_data_003,
        'MODEL-001': execute_model_001,
        'MODEL-002': execute_model_002,
        'DETECT-001': execute_detect_001,
        'DETECT-002': execute_detect_002,
        'REPORT-001': execute_report_001,
    }

    handler = handlers.get(skill_id)
    if handler:
        try:
            handler(skill_args, skill)
        except Exception as e:
            print_error(f"Execution failed: {e}")
    else:
        print_warning(f"No handler implemented for {skill_id}")
        print_info("This skill exists but execution is not yet implemented")

    print()


# ============================================================================
# SKILL HANDLERS
# ============================================================================

def parse_skill_args(args: List[str], skill: Skill) -> Dict[str, Any]:
    """Parse command line arguments into skill inputs"""
    parsed = {}

    # Set defaults
    for inp in skill.inputs:
        if 'default' in inp:
            parsed[inp['name']] = inp['default']

    # Parse provided arguments
    i = 0
    while i < len(args):
        if args[i].startswith('--'):
            key = args[i][2:]
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                parsed[key] = args[i + 1]
                i += 2
            else:
                parsed[key] = True
                i += 1
        else:
            i += 1

    return parsed


def execute_data_001(args: List[str], skill: Skill):
    """NASA Data Ingestion"""
    params = parse_skill_args(args, skill)

    source = params.get('source', 'tess')
    target = params.get('target', '')
    limit = int(params.get('limit', 100))

    if not target:
        print_error("Target required. Use --target <name>")
        return

    print_info(f"Source: {source.upper()}")
    print_info(f"Target: {target}")

    import lightkurve as lk

    print_info("Searching...")
    search = lk.search_lightcurve(target, mission=source.upper() if source != 'exoplanet_archive' else None)

    if len(search) == 0:
        print_error(f"No data found for {target}")
        return

    print_success(f"Found {len(search)} observations")

    for i, result in enumerate(search[:5]):
        print(f"  [{i+1}] {result.mission} - {result.exptime}")

    if len(search) > 5:
        print(f"  ... and {len(search) - 5} more")

    # Download first
    print_info("Downloading first observation...")
    lc = search[0].download()
    lc = lc.remove_nans().normalize()

    print_success(f"Downloaded {len(lc.flux)} data points")
    print_info(f"Time span: {lc.time.value[-1] - lc.time.value[0]:.1f} days")


def execute_data_002(args: List[str], skill: Skill):
    """FITS File Parser"""
    params = parse_skill_args(args, skill)

    filepath = params.get('file', '')
    if not filepath:
        print_error("File path required. Use --file <path>")
        return

    from astropy.io import fits

    print_info(f"Reading: {filepath}")

    with fits.open(filepath) as hdul:
        print_success(f"Opened FITS file with {len(hdul)} extensions")
        for i, hdu in enumerate(hdul):
            print(f"  [{i}] {hdu.name}: {type(hdu).__name__}")


def execute_data_003(args: List[str], skill: Skill):
    """Light Curve Processing"""
    params = parse_skill_args(args, skill)
    print_info("Processing light curve...")
    print_warning("Use /run DATA-001 first to fetch data")


def execute_model_001(args: List[str], skill: Skill):
    """Spectral CNN Training"""
    params = parse_skill_args(args, skill)
    epochs = int(params.get('epochs', 100))

    print_info(f"Training for {epochs} epochs")
    cmd_train(['--epochs', str(epochs)])


def execute_model_002(args: List[str], skill: Skill):
    """TFLite Export"""
    params = parse_skill_args(args, skill)

    if not MODEL_PATH.exists():
        print_error("No model found. Run /run MODEL-001 first")
        return

    import tensorflow as tf

    print_info("Loading model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print_info("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if params.get('quantize', True):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print_info("Applying INT8 quantization")

    tflite_model = converter.convert()

    output_path = Path("models/real/astro_tinyml.tflite")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print_success(f"Exported to {output_path} ({size_kb:.1f} KB)")


def execute_detect_001(args: List[str], skill: Skill):
    """Transit Detection"""
    params = parse_skill_args(args, skill)
    cmd_detect(args)


def execute_detect_002(args: List[str], skill: Skill):
    """Anomaly Detection"""
    params = parse_skill_args(args, skill)
    print_info("Running anomaly detection...")
    cmd_detect(['--mode', 'anomaly'])


def execute_report_001(args: List[str], skill: Skill):
    """NASA Report Generator"""
    params = parse_skill_args(args, skill)
    fmt = params.get('format', 'html')
    print_info(f"Generating {fmt.upper()} report...")
    print_warning("Report generation requires detection results")


# ============================================================================
# CORE COMMANDS
# ============================================================================

def cmd_help(args):
    """Show available commands"""
    print(f"""
{Colors.BOLD}LARUN Commands{Colors.END}

{Colors.CYAN}Skills System:{Colors.END}
  /skills              List all available skills
  /skill <ID>          Show skill details
  /run <ID> [args]     Execute a skill

{Colors.CYAN}Analysis:{Colors.END}
  /bls <target>        Run BLS periodogram for transit detection
  /phase <target> <P>  Phase fold light curve at period P
  /detect              Run detection on loaded data

{Colors.CYAN}Data:{Colors.END}
  /train [opts]        Train model on NASA data
  /fetch <target>      Fetch star data

{Colors.CYAN}Developer:{Colors.END}
  /addon               List/load developer addons
  /generate            Generate Python scripts and ML models

{Colors.CYAN}System:{Colors.END}
  /status              Show system status
  /config [key] [val]  View/modify settings
  /clear               Clear screen
  /exit                Exit LARUN

{Colors.DIM}Type naturally: "run bls on TOI-700", "train the model", etc.{Colors.END}
""")


def cmd_status(args):
    """Show current status"""
    print(f"\n{Colors.BOLD}LARUN Status{Colors.END}")
    print(f"{'─' * 40}")

    # Skills status
    if skill_loader.loaded:
        active = len(skill_loader.list_active())
        total = len(skill_loader.skills)
        print_success(f"Skills loaded: {active} active / {total} total")
    else:
        print_warning("Skills not loaded")

    # Model status
    if state.model:
        print_success(f"Model loaded: {MODEL_PATH}")
    elif MODEL_PATH.exists():
        print_warning("Model available but not loaded")
    else:
        print_error("No trained model found")

    # Data status
    if DATA_PATH.exists():
        import numpy as np
        data = np.load(str(DATA_PATH))
        print_success(f"Training data: {len(data['X'])} samples")
    else:
        print_warning("No training data cached")

    print()


def cmd_train(args):
    """Train the model"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--planets', type=int, default=CONFIG['num_planets'])
    parser.add_argument('--epochs', type=int, default=CONFIG['num_epochs'])
    parser.add_argument('--skip-fetch', action='store_true')
    opts, _ = parser.parse_known_args(args)

    print(f"\n{Colors.BOLD}Training Pipeline{Colors.END}")
    print(f"{'─' * 40}")
    print_info(f"Planets: {opts.planets}, Epochs: {opts.epochs}")

    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    # Fetch data
    X, y = fetch_data(opts.planets, opts.skip_fetch)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(-1, CONFIG["input_size"], 1)
    X_val = X_val.reshape(-1, CONFIG["input_size"], 1)

    print_info(f"Train: {len(X_train)}, Validation: {len(X_val)}")

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(16, 7, activation='relu', input_shape=(CONFIG["input_size"], 1)),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Conv1D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(CONFIG["num_classes"], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=opts.epochs, batch_size=16, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
    )

    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print()
    print_success(f"Training complete! Accuracy: {val_acc*100:.1f}%")

    Path("models/real").mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print_success(f"Model saved to {MODEL_PATH}")

    state.model = model
    print()


def fetch_data(num_planets, skip=False):
    """Fetch NASA data"""
    import numpy as np

    Path("data/real").mkdir(parents=True, exist_ok=True)

    if skip and DATA_PATH.exists():
        data = np.load(str(DATA_PATH))
        print_success(f"Loaded {len(data['X'])} cached samples")
        return data['X'], data['y']

    print(f"\n{Colors.BOLD}Fetching NASA Data{Colors.END}")

    import lightkurve as lk
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
    import pandas as pd

    X, y = [], []
    size = CONFIG["input_size"]

    print_info("Querying NASA Exoplanet Archive...")
    try:
        planets = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="pl_name,hostname,disc_facility",
            where="disc_facility LIKE '%TESS%' OR disc_facility LIKE '%Kepler%'"
        ).to_pandas()
        print_success(f"Found {len(planets)} exoplanets")
    except Exception as e:
        print_error(f"Archive query failed: {e}")
        planets = pd.DataFrame()

    if len(planets) > 0:
        print_info("Downloading light curves...")
        count = 0
        for _, row in planets.iterrows():
            if count >= num_planets:
                break
            host = row.get('hostname', '')
            if not host:
                continue
            try:
                for mission in ["TESS", "Kepler"]:
                    search = lk.search_lightcurve(host, mission=mission)
                    if len(search) > 0:
                        lc = search[0].download()
                        if lc:
                            lc = lc.remove_nans().normalize()
                            flux = np.interp(np.linspace(0,1,size),
                                           np.linspace(0,1,len(lc.flux.value)),
                                           lc.flux.value)
                            X.append(flux.astype(np.float32))
                            y.append(2)
                            count += 1
                            print(f"   {Colors.GREEN}✓{Colors.END} [{count}/{num_planets}] {host}")
                            break
            except:
                continue

    # Synthetic samples
    print_info("Adding synthetic samples...")
    for _ in range(30):
        X.append(np.random.normal(1, 0.01, size).astype(np.float32))
        y.append(0)
    for _ in range(30):
        t = np.linspace(0, 10, size)
        flux = 1 + 0.02*np.sin(2*np.pi*t/np.random.uniform(0.5,2))
        X.append(flux.astype(np.float32))
        y.append(1)
    for _ in range(30):
        t = np.linspace(0, 10, size)
        p = np.random.uniform(0.5, 3)
        flux = np.ones(size)
        phase = (t % p) / p
        flux[np.abs(phase) < 0.05] -= np.random.uniform(0.1, 0.4)
        X.append(flux.astype(np.float32))
        y.append(3)
    for _ in range(30):
        flux = np.ones(size)
        for _ in range(np.random.randint(1,5)):
            flux[np.random.randint(size):] += np.random.uniform(-0.1, 0.1)
        X.append(flux.astype(np.float32))
        y.append(4)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)

    np.savez(str(DATA_PATH), X=X, y=y)
    print_success(f"Saved {len(X)} samples")

    return X, y


def cmd_detect(args):
    """Run detection"""
    if not state.model:
        if MODEL_PATH.exists():
            print_info("Loading model...")
            state.load_model()
        else:
            print_error("No model found. Run /train first.")
            return

    import numpy as np

    if not DATA_PATH.exists():
        print_error("No data found. Run /train first.")
        return

    data = np.load(str(DATA_PATH))
    X, y = data['X'], data['y']
    X = X.reshape(-1, CONFIG["input_size"], 1)

    print(f"\n{Colors.BOLD}Running Detection{Colors.END}")
    print(f"{'─' * 40}")

    preds = state.model.predict(X, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    correct = np.sum(pred_classes == y)
    acc = correct / len(y)

    print_success(f"Processed {len(X)} samples")
    print_success(f"Accuracy: {acc*100:.1f}%")

    print(f"\n{Colors.BOLD}Results by Class:{Colors.END}")
    for i, name in enumerate(CLASS_NAMES):
        mask = y == i
        if np.sum(mask) > 0:
            class_acc = np.sum(pred_classes[mask] == i) / np.sum(mask)
            print(f"   {name:15} {np.sum(mask):3} samples, {class_acc*100:.0f}% correct")

    transit_mask = (pred_classes == 2) & (confidences > 0.7)
    print(f"\n{Colors.CYAN}Transit candidates:{Colors.END} {np.sum(transit_mask)}")
    print()


def cmd_fetch(args):
    """Fetch light curve for a specific target"""
    if not args:
        print_error("Usage: /fetch <target_name>")
        return

    target = ' '.join(args)

    # Use DATA-001 skill
    execute_data_001(['--target', target], skill_loader.get('DATA-001') or Skill({}))


def cmd_config(args):
    """View or modify configuration"""
    if not args:
        print(f"\n{Colors.BOLD}Configuration:{Colors.END}")
        for k, v in CONFIG.items():
            print(f"   {k}: {v}")
        print()
        return

    if len(args) >= 2:
        key, value = args[0], args[1]
        if key in CONFIG:
            try:
                CONFIG[key] = type(CONFIG[key])(value)
                print_success(f"Set {key} = {value}")
            except:
                print_error(f"Invalid value for {key}")


def cmd_clear(args):
    """Clear screen"""
    print('\033[2J\033[H', end='')


def cmd_addon(args):
    """Load or list addons"""
    if not args:
        print(f"\n{Colors.BOLD}Available Addons:{Colors.END}")
        available = addon_loader.list_available()

        if not available:
            print(f"   {Colors.DIM}No addons found in {ADDONS_PATH}{Colors.END}")
        else:
            for addon in available:
                loaded = addon_loader.is_loaded(addon)
                status = f"{Colors.GREEN}●{Colors.END} loaded" if loaded else f"{Colors.DIM}○{Colors.END} available"
                print(f"   {addon:20} [{status}]")

        print(f"\n{Colors.DIM}Usage: /addon <name> to load an addon{Colors.END}\n")
        return

    addon_name = args[0].lower()

    if addon_loader.is_loaded(addon_name):
        print_info(f"Addon '{addon_name}' already loaded")
        return

    print_info(f"Loading addon: {addon_name}...")
    if addon_loader.load(addon_name):
        print_success(f"Addon '{addon_name}' loaded successfully")
        skills_count = len(addon_loader.addon_skills)
        if skills_count:
            print_info(f"Added {skills_count} new skills")
    else:
        print_error(f"Failed to load addon '{addon_name}'")


def cmd_generate(args):
    """Generate code using codegen addon"""
    # Auto-load codegen addon if not loaded
    if not addon_loader.is_loaded('codegen'):
        if 'codegen' in addon_loader.list_available():
            print_info("Loading codegen addon...")
            if not addon_loader.load('codegen'):
                print_error("Failed to load codegen addon")
                return
        else:
            print_error("Codegen addon not found. Please install it first.")
            return

    codegen = addon_loader.get_addon('codegen')
    if not codegen:
        print_error("Codegen module not available")
        return

    if not args:
        print(f"""
{Colors.BOLD}Code Generation{Colors.END}
{Colors.DIM}Generate Python scripts and ML models{Colors.END}

{Colors.BOLD}Usage:{Colors.END}
  /generate script <type>      Generate a Python script
  /generate model <arch>       Generate an ML model
  /generate training <type>    Generate a training script
  /generate pipeline <source>  Generate a data pipeline

{Colors.BOLD}Script Types:{Colors.END}
  data_fetch, lightcurve_analysis, transit_search,
  anomaly_detection, report_generation

{Colors.BOLD}Model Architectures:{Colors.END}
  cnn_1d, lstm, transformer, autoencoder, hybrid

{Colors.BOLD}Examples:{Colors.END}
  /generate script data_fetch
  /generate model cnn_1d
  /generate training cnn
  /generate pipeline tess
""")
        return

    cmd = args[0].lower()
    cmd_args = args[1:] if len(args) > 1 else []

    generator = codegen.create_generator("./generated")

    try:
        if cmd == "script":
            task = cmd_args[0] if cmd_args else "data_fetch"
            target = cmd_args[1] if len(cmd_args) > 1 else "TIC 307210830"
            path = generator.generate_script(task, target)
            print_success(f"Generated script: {path}")

        elif cmd == "model":
            arch = cmd_args[0] if cmd_args else "cnn_1d"
            input_shape = (200, 1)  # Default shape
            path = generator.generate_ml_model(arch, input_shape, num_classes=6)
            print_success(f"Generated model: {path}")

        elif cmd == "training":
            model_type = cmd_args[0] if cmd_args else "cnn"
            path = generator.generate_training_script(model_type)
            print_success(f"Generated training script: {path}")

        elif cmd == "pipeline":
            source = cmd_args[0] if cmd_args else "tess"
            steps = ["fetch", "clean", "normalize", "detect", "report"]
            path = generator.generate_pipeline(source, steps)
            print_success(f"Generated pipeline: {path}")

        else:
            print_error(f"Unknown generate command: {cmd}")

    except Exception as e:
        print_error(f"Generation failed: {e}")
    print_banner()


def cmd_bls(args):
    """Run BLS periodogram for transit detection"""
    print(f"\n{Colors.BOLD}BLS Periodogram - Transit Detection{Colors.END}")
    print(f"{Colors.DIM}Box Least Squares algorithm for periodic transit signals{Colors.END}\n")

    # Parse arguments
    target = None
    min_period = 0.5
    max_period = 20.0

    i = 0
    while i < len(args):
        if args[i] == '--target' and i + 1 < len(args):
            target = args[i + 1]
            i += 2
        elif args[i] == '--min-period' and i + 1 < len(args):
            min_period = float(args[i + 1])
            i += 2
        elif args[i] == '--max-period' and i + 1 < len(args):
            max_period = float(args[i + 1])
            i += 2
        elif not args[i].startswith('--'):
            target = args[i]
            i += 1
        else:
            i += 1

    if not target:
        print(f"""
{Colors.BOLD}Usage:{Colors.END}
  /bls <target>                    Run BLS on a target star
  /bls --target "TIC 307210830"    Specify target by TIC ID
  /bls <target> --min-period 1 --max-period 30

{Colors.BOLD}Options:{Colors.END}
  --target       Target name or TIC ID
  --min-period   Minimum period to search (days, default: 0.5)
  --max-period   Maximum period to search (days, default: 20)

{Colors.BOLD}Examples:{Colors.END}
  /bls TIC 307210830
  /bls "Kepler-10" --min-period 0.5 --max-period 50
  /bls TOI-700 --max-period 30
""")
        return

    try:
        # Import required modules
        import numpy as np
        import lightkurve as lk
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        from skills.periodogram import BLSPeriodogram, phase_fold, bin_phase_curve

        print(f"{Colors.CYAN}Fetching light curve for {target}...{Colors.END}")

        # Search for light curve
        search = lk.search_lightcurve(target, mission=['TESS', 'Kepler'])

        if len(search) == 0:
            print_error(f"No light curves found for {target}")
            return

        print(f"{Colors.DIM}Found {len(search)} observations{Colors.END}")

        # Download first observation
        lc = search[0].download()
        lc = lc.remove_nans().normalize().remove_outliers(sigma=3)

        time = lc.time.value
        flux = lc.flux.value

        print(f"{Colors.DIM}Light curve: {len(time)} points, baseline: {time.max()-time.min():.1f} days{Colors.END}")

        # Run BLS
        print(f"\n{Colors.CYAN}Running BLS periodogram...{Colors.END}")
        print(f"{Colors.DIM}Period range: {min_period} - {max_period} days{Colors.END}")

        bls = BLSPeriodogram(
            min_period=min_period,
            max_period=max_period,
            n_periods=5000
        )

        result = bls.compute(time, flux, min_snr=7.0)

        # Display results
        print(f"\n{Colors.GREEN}═══════════════════════════════════════════════════════════{Colors.END}")
        print(f"{Colors.BOLD}BLS Results for {target}{Colors.END}")
        print(f"{Colors.GREEN}═══════════════════════════════════════════════════════════{Colors.END}")

        print(f"\n{Colors.BOLD}Best Period:{Colors.END} {result.best_period:.6f} days")
        print(f"{Colors.BOLD}BLS Power:{Colors.END}   {result.best_power:.4f}")
        print(f"{Colors.BOLD}FAP:{Colors.END}         {result.fap:.2e}")

        if result.candidates:
            print(f"\n{Colors.BOLD}Transit Candidates:{Colors.END}")
            for i, c in enumerate(result.candidates, 1):
                print(f"\n  {Colors.CYAN}Candidate {i}:{Colors.END}")
                print(f"    Period:   {c.period:.4f} days ({c.period*24:.2f} hours)")
                print(f"    Depth:    {c.depth*1e6:.0f} ppm ({c.depth*100:.3f}%)")
                print(f"    Duration: {c.duration*24:.2f} hours")
                print(f"    SNR:      {c.snr:.1f}")
                print(f"    T0:       {c.t0:.4f}")

                # Estimate planet radius (assuming Sun-like star)
                rp_rs = np.sqrt(c.depth)
                rp_earth = rp_rs * 109.2  # Sun radius in Earth radii
                print(f"    Est. Rp:  {rp_earth:.1f} R⊕ (Sun-like star)")
        else:
            print(f"\n{Colors.YELLOW}No significant transit candidates found (SNR < 7){Colors.END}")

        # Save results
        output_file = f"output/bls_{target.replace(' ', '_')}.json"
        Path("output").mkdir(exist_ok=True)

        import json
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"\n{Colors.DIM}Results saved to: {output_file}{Colors.END}")
        print()

    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_info("Install with: pip install lightkurve astropy")
    except Exception as e:
        print_error(f"BLS analysis failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_phase(args):
    """Phase fold a light curve at a given period"""
    print(f"\n{Colors.BOLD}Phase Folding{Colors.END}")
    print(f"{Colors.DIM}Fold light curve at a given period{Colors.END}\n")

    if len(args) < 2:
        print(f"""
{Colors.BOLD}Usage:{Colors.END}
  /phase <target> <period>         Phase fold at given period
  /phase <target> <period> --t0 <epoch>

{Colors.BOLD}Examples:{Colors.END}
  /phase "TIC 307210830" 3.5
  /phase "Kepler-10" 0.837 --t0 2454964.5
""")
        return

    target = args[0]
    period = float(args[1])
    t0 = 0.0

    # Parse optional t0
    if '--t0' in args:
        idx = args.index('--t0')
        if idx + 1 < len(args):
            t0 = float(args[idx + 1])

    try:
        import numpy as np
        import lightkurve as lk
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        from skills.periodogram import phase_fold, bin_phase_curve

        print(f"{Colors.CYAN}Fetching light curve for {target}...{Colors.END}")

        search = lk.search_lightcurve(target, mission=['TESS', 'Kepler'])
        if len(search) == 0:
            print_error(f"No light curves found for {target}")
            return

        lc = search[0].download()
        lc = lc.remove_nans().normalize().remove_outliers(sigma=3)

        time = lc.time.value
        flux = lc.flux.value

        # Phase fold
        phase, flux_folded = phase_fold(time, flux, period, t0)

        # Bin the data
        bin_phase, bin_flux, bin_err = bin_phase_curve(phase, flux_folded, n_bins=100)

        # Calculate transit depth from binned data
        min_idx = np.nanargmin(bin_flux)
        depth = 1.0 - bin_flux[min_idx]

        print(f"\n{Colors.GREEN}═══════════════════════════════════════════════════════════{Colors.END}")
        print(f"{Colors.BOLD}Phase Folded: {target} at P={period:.4f}d{Colors.END}")
        print(f"{Colors.GREEN}═══════════════════════════════════════════════════════════{Colors.END}")

        print(f"\n{Colors.BOLD}Transit Depth:{Colors.END} {depth*1e6:.0f} ppm ({depth*100:.3f}%)")
        print(f"{Colors.BOLD}Phase at min:{Colors.END}  {bin_phase[min_idx]:.3f}")

        # ASCII phase curve visualization
        print(f"\n{Colors.BOLD}Phase Curve:{Colors.END}")
        print_ascii_phase_curve(bin_phase, bin_flux)

        # Save results
        output_file = f"output/phase_{target.replace(' ', '_')}.json"
        Path("output").mkdir(exist_ok=True)

        import json
        with open(output_file, 'w') as f:
            json.dump({
                'target': target,
                'period': period,
                't0': t0,
                'depth_ppm': float(depth * 1e6),
                'phase': bin_phase.tolist(),
                'flux': bin_flux.tolist(),
                'flux_err': bin_err.tolist()
            }, f, indent=2)

        print(f"\n{Colors.DIM}Results saved to: {output_file}{Colors.END}")

    except Exception as e:
        print_error(f"Phase folding failed: {e}")
        import traceback
        traceback.print_exc()


def print_ascii_phase_curve(phase, flux, width=60, height=10):
    """Print ASCII representation of phase curve"""
    import numpy as np

    # Normalize flux for display
    flux_norm = flux - np.nanmin(flux)
    flux_norm = flux_norm / np.nanmax(flux_norm) if np.nanmax(flux_norm) > 0 else flux_norm

    # Create ASCII plot
    for row in range(height, -1, -1):
        line = ""
        threshold = row / height

        for i, (p, f) in enumerate(zip(phase, flux_norm)):
            if np.isnan(f):
                line += " "
            elif f >= threshold:
                if row == height:
                    line += "─"
                else:
                    line += "█"
            else:
                line += " "

        # Add axis label
        if row == height:
            print(f"  1.00 │{line}│")
        elif row == height // 2:
            y_val = np.nanmin(flux) + (np.nanmax(flux) - np.nanmin(flux)) * 0.5
            print(f"  {y_val:.3f}│{line}│")
        elif row == 0:
            print(f"  {np.nanmin(flux):.3f}│{line}│")
        else:
            print(f"       │{line}│")

    # X-axis
    print(f"       └{'─' * len(phase)}┘")
    print(f"       -0.5{' ' * (len(phase)//2 - 4)}0{' ' * (len(phase)//2 - 3)}0.5")


def process_natural_language(query):
    """Process natural language queries"""
    query_lower = query.lower()

    if any(w in query_lower for w in ['train', 'learn', 'fit']):
        cmd_train([])
    elif any(w in query_lower for w in ['detect', 'classify', 'predict']):
        cmd_detect([])
    elif any(w in query_lower for w in ['bls', 'periodogram', 'period']):
        cmd_bls([])
    elif any(w in query_lower for w in ['phase', 'fold']):
        cmd_phase([])
    elif any(w in query_lower for w in ['skill', 'abilities', 'capabilities']):
        cmd_skills([])
    elif any(w in query_lower for w in ['status', 'info']):
        cmd_status([])
    elif any(w in query_lower for w in ['help', 'command']):
        cmd_help([])
    else:
        print(f"\n{Colors.DIM}Try /help for commands or /skills to see capabilities{Colors.END}\n")


# ============================================================================
# MAIN LOOP
# ============================================================================

COMMANDS = {
    '/help': cmd_help,
    '/status': cmd_status,
    '/skills': cmd_skills,
    '/skill': cmd_skill,
    '/run': cmd_run,
    '/train': cmd_train,
    '/detect': cmd_detect,
    '/fetch': cmd_fetch,
    '/config': cmd_config,
    '/clear': cmd_clear,
    '/addon': cmd_addon,
    '/generate': cmd_generate,
    '/bls': cmd_bls,
    '/phase': cmd_phase,
}

def main():
    # Install yaml if needed
    if not YAML_AVAILABLE:
        print_warning("PyYAML not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyyaml', '-q'])
        print_success("Installed PyYAML. Please restart larun.")
        return

    print_banner()

    if skill_loader.loaded:
        active = len(skill_loader.list_active())
        print_info(f"Loaded {active} active skills. Type /skills to see all.\n")

    if MODEL_PATH.exists():
        print_info(f"Found trained model. Type /status for details.\n")

    while True:
        try:
            user_input = input(print_prompt()).strip()

            if not user_input:
                continue

            if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                print(f"\n{Colors.DIM}Goodbye!{Colors.END}\n")
                break

            if user_input.startswith('/'):
                parts = user_input.split()
                cmd = parts[0].lower()
                args = parts[1:]

                if cmd in COMMANDS:
                    COMMANDS[cmd](args)
                else:
                    print_error(f"Unknown command: {cmd}")
                    print_info("Type /help for available commands")
            else:
                process_natural_language(user_input)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.DIM}Use /exit to quit{Colors.END}\n")
        except EOFError:
            break
        except Exception as e:
            print_error(f"Error: {e}")

if __name__ == "__main__":
    main()
