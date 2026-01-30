#!/usr/bin/env python3
"""
LARUN TinyML - Interactive CLI
==============================
Interactive terminal interface for AstroTinyML, similar to Claude Code.

Usage:
    python larun.py              # Start interactive mode
    python larun.py --help       # Show help

Commands:
    /train      - Train model on NASA data
    /detect     - Run detection on data
    /report     - Generate reports
    /status     - Show current status
    /config     - View/edit configuration
    /help       - Show available commands
    /exit       - Exit the CLI

Larun. x Astrodata
"""

import sys
import json
import time
import readline  # For command history
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "1.0.0"
MODEL_PATH = Path("models/real/astro_tinyml.h5")
DATA_PATH = Path("data/real/training_data.npz")

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
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

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
║     {Colors.DIM}v{VERSION}{Colors.BOLD}                                                           ║
╚══════════════════════════════════════════════════════════════════════╝{Colors.END}
""")

def print_prompt():
    return f"{Colors.GREEN}larun{Colors.END} {Colors.DIM}>{Colors.END} "

def print_thinking(message):
    print(f"{Colors.DIM}⠋ {message}...{Colors.END}", end='\r')

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
        self.loaded = False

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
# COMMANDS
# ============================================================================

def cmd_help(args):
    """Show available commands"""
    print(f"""
{Colors.BOLD}Available Commands:{Colors.END}

  {Colors.CYAN}/train{Colors.END} [--planets N] [--epochs N]
      Train model on real NASA data from TESS/Kepler

  {Colors.CYAN}/detect{Colors.END} [--file PATH] [--target NAME]
      Run detection on data or specific target

  {Colors.CYAN}/fetch{Colors.END} [TARGET_NAME]
      Fetch light curve for a specific star

  {Colors.CYAN}/status{Colors.END}
      Show current model and data status

  {Colors.CYAN}/config{Colors.END} [key] [value]
      View or modify configuration

  {Colors.CYAN}/clear{Colors.END}
      Clear the screen

  {Colors.CYAN}/exit{Colors.END} or {Colors.CYAN}/quit{Colors.END}
      Exit LARUN

{Colors.DIM}Type any question or command to interact with the system.{Colors.END}
""")

def cmd_status(args):
    """Show current status"""
    print(f"\n{Colors.BOLD}LARUN Status{Colors.END}")
    print(f"{'─' * 40}")

    # Model status
    if state.model:
        print_success(f"Model loaded: {MODEL_PATH}")
        print(f"   Parameters: {state.model.count_params():,}")
    elif MODEL_PATH.exists():
        print_warning(f"Model available but not loaded")
    else:
        print_error("No trained model found")

    # Data status
    if DATA_PATH.exists():
        import numpy as np
        data = np.load(str(DATA_PATH))
        print_success(f"Training data: {len(data['X'])} samples")
    else:
        print_warning("No training data cached")

    # Config
    print(f"\n{Colors.BOLD}Configuration:{Colors.END}")
    for k, v in CONFIG.items():
        print(f"   {k}: {v}")
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
    print()

    # Import dependencies
    print_thinking("Loading dependencies")
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    print_success("Dependencies loaded")

    # Fetch data
    X, y = fetch_data(opts.planets, opts.skip_fetch)

    # Train
    print(f"\n{Colors.BOLD}Training Model{Colors.END}")
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

    # Save
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

    print_thinking("Querying NASA Exoplanet Archive")
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
                            y.append(2)  # planetary_transit
                            count += 1
                            print(f"   {Colors.GREEN}✓{Colors.END} [{count}/{num_planets}] {host}")
                            break
            except:
                continue

    # Add synthetic samples
    print_info("Adding synthetic samples for class balance...")

    for _ in range(30):  # Noise
        X.append(np.random.normal(1, 0.01, size).astype(np.float32))
        y.append(0)

    for _ in range(30):  # Stellar
        t = np.linspace(0, 10, size)
        flux = 1 + 0.02*np.sin(2*np.pi*t/np.random.uniform(0.5,2))
        X.append(flux.astype(np.float32))
        y.append(1)

    for _ in range(30):  # Binary
        t = np.linspace(0, 10, size)
        p = np.random.uniform(0.5, 3)
        flux = np.ones(size)
        phase = (t % p) / p
        flux[np.abs(phase) < 0.05] -= np.random.uniform(0.1, 0.4)
        X.append(flux.astype(np.float32))
        y.append(3)

    for _ in range(30):  # Artifact
        flux = np.ones(size)
        for _ in range(np.random.randint(1,5)):
            flux[np.random.randint(size):] += np.random.uniform(-0.1, 0.1)
        X.append(flux.astype(np.float32))
        y.append(4)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)

    np.savez(str(DATA_PATH), X=X, y=y)
    print_success(f"Saved {len(X)} samples to {DATA_PATH}")

    return X, y

def cmd_detect(args):
    """Run detection"""
    if not state.model:
        if MODEL_PATH.exists():
            print_thinking("Loading model")
            state.load_model()
            print_success("Model loaded")
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

    # Show breakdown
    print(f"\n{Colors.BOLD}Results by Class:{Colors.END}")
    for i, name in enumerate(CLASS_NAMES):
        mask = y == i
        if np.sum(mask) > 0:
            class_acc = np.sum(pred_classes[mask] == i) / np.sum(mask)
            print(f"   {name:15} {np.sum(mask):3} samples, {class_acc*100:.0f}% correct")

    # Transit candidates
    transit_mask = (pred_classes == 2) & (confidences > 0.7)
    print(f"\n{Colors.CYAN}Transit candidates:{Colors.END} {np.sum(transit_mask)}")
    print()

def cmd_fetch(args):
    """Fetch light curve for a specific target"""
    if not args:
        print_error("Usage: /fetch <target_name>")
        print_info("Example: /fetch Kepler-186")
        return

    target = ' '.join(args)
    print(f"\n{Colors.BOLD}Fetching: {target}{Colors.END}")

    import lightkurve as lk
    import numpy as np

    print_thinking(f"Searching for {target}")

    try:
        search = lk.search_lightcurve(target)
        if len(search) == 0:
            print_error(f"No light curves found for {target}")
            return

        print_success(f"Found {len(search)} observations")
        print()

        # Show available observations
        for i, result in enumerate(search[:5]):
            print(f"   [{i+1}] {result.mission} - {result.exptime}")

        if len(search) > 5:
            print(f"   ... and {len(search)-5} more")

        # Download first one
        print()
        print_thinking("Downloading light curve")
        lc = search[0].download()
        lc = lc.remove_nans().normalize()

        print_success(f"Downloaded {len(lc.flux)} data points")
        print_info(f"Time span: {lc.time.value[-1] - lc.time.value[0]:.1f} days")
        print_info(f"Mean flux: {np.mean(lc.flux.value):.4f}")

        # Quick classification if model loaded
        if state.model:
            flux = np.interp(np.linspace(0,1,CONFIG["input_size"]),
                           np.linspace(0,1,len(lc.flux.value)),
                           lc.flux.value).astype(np.float32)
            flux = (flux - flux.mean()) / (flux.std() + 1e-8)
            flux = flux.reshape(1, -1, 1)

            pred = state.model.predict(flux, verbose=0)[0]
            pred_class = np.argmax(pred)
            conf = pred[pred_class]

            print()
            print(f"{Colors.BOLD}Classification:{Colors.END} {CLASS_NAMES[pred_class]} ({conf*100:.1f}%)")

        print()

    except Exception as e:
        print_error(f"Failed: {e}")

def cmd_config(args):
    """View or modify configuration"""
    if not args:
        print(f"\n{Colors.BOLD}Configuration:{Colors.END}")
        for k, v in CONFIG.items():
            print(f"   {k}: {v}")
        print()
        return

    if len(args) == 1:
        key = args[0]
        if key in CONFIG:
            print(f"   {key}: {CONFIG[key]}")
        else:
            print_error(f"Unknown config key: {key}")
    elif len(args) >= 2:
        key, value = args[0], args[1]
        if key in CONFIG:
            try:
                CONFIG[key] = type(CONFIG[key])(value)
                print_success(f"Set {key} = {value}")
            except:
                print_error(f"Invalid value for {key}")
        else:
            print_error(f"Unknown config key: {key}")

def cmd_clear(args):
    """Clear screen"""
    print('\033[2J\033[H', end='')  # ANSI escape codes to clear screen
    print_banner()

def process_natural_language(query):
    """Process natural language queries"""
    query_lower = query.lower()

    if any(w in query_lower for w in ['train', 'learn', 'fit']):
        print_info("I'll start training. Use /train for more options.")
        cmd_train([])
    elif any(w in query_lower for w in ['detect', 'classify', 'predict', 'analyze']):
        print_info("Running detection...")
        cmd_detect([])
    elif any(w in query_lower for w in ['status', 'info', 'loaded']):
        cmd_status([])
    elif any(w in query_lower for w in ['fetch', 'download', 'get']) and any(w in query_lower for w in ['kepler', 'tess', 'star', 'planet']):
        words = query.split()
        for i, w in enumerate(words):
            if w.lower() in ['kepler', 'tess', 'hd', 'tic', 'kic']:
                target = ' '.join(words[i:i+2]) if i+1 < len(words) else w
                cmd_fetch([target])
                return
        print_info("Which target? Try: /fetch Kepler-186")
    elif any(w in query_lower for w in ['help', 'command', 'how']):
        cmd_help([])
    elif 'exoplanet' in query_lower or 'planet' in query_lower:
        print_info("LARUN detects exoplanets using TinyML on spectral data from NASA's TESS and Kepler missions.")
        print_info("Try /train to train on real NASA data, then /detect to classify light curves.")
    else:
        print(f"\n{Colors.DIM}I'm not sure how to help with that.{Colors.END}")
        print(f"{Colors.DIM}Try /help to see available commands, or ask about training, detection, or fetching data.{Colors.END}\n")

# ============================================================================
# MAIN LOOP
# ============================================================================

COMMANDS = {
    '/help': cmd_help,
    '/status': cmd_status,
    '/train': cmd_train,
    '/detect': cmd_detect,
    '/fetch': cmd_fetch,
    '/config': cmd_config,
    '/clear': cmd_clear,
}

def main():
    print_banner()

    # Try to load existing model
    if MODEL_PATH.exists():
        print_info(f"Found trained model at {MODEL_PATH}")
        print_info("Type /status to see details or /help for commands\n")
    else:
        print_info("No trained model found. Type /train to get started.\n")

    while True:
        try:
            user_input = input(print_prompt()).strip()

            if not user_input:
                continue

            # Handle exit
            if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                print(f"\n{Colors.DIM}Goodbye!{Colors.END}\n")
                break

            # Handle commands
            if user_input.startswith('/'):
                parts = user_input.split()
                cmd = parts[0].lower()
                args = parts[1:]

                if cmd in COMMANDS:
                    COMMANDS[cmd](args)
                else:
                    print_error(f"Unknown command: {cmd}")
                    print_info("Type /help to see available commands")
            else:
                # Natural language processing
                process_natural_language(user_input)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.DIM}Use /exit to quit{Colors.END}\n")
        except EOFError:
            print(f"\n{Colors.DIM}Goodbye!{Colors.END}\n")
            break
        except Exception as e:
            print_error(f"Error: {e}")

if __name__ == "__main__":
    main()
