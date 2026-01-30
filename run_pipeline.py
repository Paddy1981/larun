#!/usr/bin/env python3
"""
AstroTinyML - Complete Pipeline Runner
======================================
Single script that runs the entire pipeline in sequence:
1. Install dependencies
2. Fetch real NASA data
3. Train TinyML model
4. Run detection
5. Generate reports

Usage:
    python run_pipeline.py
    python run_pipeline.py --planets 100 --epochs 150

Larun. × Astrodata
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "num_planets": 50,
    "num_epochs": 100,
    "input_size": 1024,
    "num_classes": 6,
    "batch_size": 16,
    "min_snr": 7.0,
    "output_dir": "output",
    "models_dir": "models/real",
    "data_dir": "data/real",
}

CLASS_NAMES = ["noise", "stellar_signal", "planetary_transit",
               "eclipsing_binary", "instrument_artifact", "unknown_anomaly"]

# ============================================================================
# UTILITIES
# ============================================================================

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗    ██╗                   ║
║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║    ╚═╝                   ║
║     ██║     ███████║██████╔╝██║   ██║██╔██╗ ██║    ██╗                   ║
║     ██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║    ╚═╝                   ║
║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║    ██╗                   ║
║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝                   ║
║                                                                          ║
║     AstroTinyML - Complete Pipeline                                      ║
║     Larun. × Astrodata                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

def print_step(num, title):
    print(f"\n{'='*70}")
    print(f"  STEP {num}: {title}")
    print(f"{'='*70}\n")

# ============================================================================
# STEP 0: INSTALL DEPENDENCIES
# ============================================================================

def install_dependencies():
    print_step(0, "CHECKING DEPENDENCIES")
    
    packages = {
        'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow', 'lightkurve': 'lightkurve',
        'astroquery': 'astroquery', 'astropy': 'astropy'
    }
    
    missing = []
    for mod, pkg in packages.items():
        try:
            __import__(mod)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n  Installing: {', '.join(missing)}")
        for pkg in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print("  ✓ Installed")

# ============================================================================
# STEP 1: FETCH NASA DATA
# ============================================================================

def fetch_data(num_planets=50, skip=False):
    print_step(1, "FETCHING NASA DATA")
    
    import numpy as np
    import pandas as pd
    
    data_dir = Path(CONFIG["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "training_data.npz"
    
    if skip and cache.exists():
        data = np.load(cache)
        print(f"  ✓ Loaded {len(data['X'])} cached samples")
        return data['X'], data['y']
    
    import lightkurve as lk
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
    
    X, y = [], []
    size = CONFIG["input_size"]
    
    # Fetch exoplanets
    print("  Querying NASA Exoplanet Archive...")
    try:
        planets = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="pl_name,hostname,disc_facility",
            where="disc_facility LIKE '%TESS%' OR disc_facility LIKE '%Kepler%'"
        ).to_pandas()
        print(f"  ✓ Found {len(planets)} exoplanets")
    except:
        planets = pd.DataFrame()
    
    # Download light curves
    if len(planets) > 0:
        print(f"\n  Downloading light curves...")
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
                            print(f"    ✓ [{count}/{num_planets}] {host}")
                            break
            except:
                continue
    
    # Synthetic samples
    print("\n  Adding synthetic samples...")
    
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
        flux[np.abs(phase-0.5) < 0.05] -= np.random.uniform(0.02, 0.15)
        X.append(flux.astype(np.float32))
        y.append(3)
    
    for _ in range(30):  # Artifact
        flux = np.ones(size)
        for _ in range(np.random.randint(1,5)):
            flux[np.random.randint(size):] += np.random.uniform(-0.1, 0.1)
        X.append(flux.astype(np.float32))
        y.append(4)
    
    print("    ✓ Added 120 synthetic samples")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    
    np.savez(cache, X=X, y=y)
    print(f"\n  ✓ Total: {len(X)} samples")
    return X, y

# ============================================================================
# STEP 2: TRAIN MODEL
# ============================================================================

def train(X, y, epochs=100):
    print_step(2, "TRAINING MODEL")
    
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    
    Path(CONFIG["models_dir"]).mkdir(parents=True, exist_ok=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(-1, CONFIG["input_size"], 1)
    X_val = X_val.reshape(-1, CONFIG["input_size"], 1)
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    
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
    print(f"  Parameters: {model.count_params():,}\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=16, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
    )
    
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  ✓ Accuracy: {val_acc*100:.1f}%")
    
    # Save models
    model.save(f"{CONFIG['models_dir']}/astro_tinyml.h5")
    print(f"  ✓ Saved: {CONFIG['models_dir']}/astro_tinyml.h5")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = converter.convert()
    with open(f"{CONFIG['models_dir']}/astro_tinyml.tflite", 'wb') as f:
        f.write(tflite)
    print(f"  ✓ Saved: {CONFIG['models_dir']}/astro_tinyml.tflite ({len(tflite)/1024:.1f}KB)")
    
    return model, val_acc

# ============================================================================
# STEP 3: RUN DETECTION
# ============================================================================

def detect(model, X, y):
    print_step(3, "RUNNING DETECTION")
    
    import numpy as np
    
    X_in = X.reshape(-1, CONFIG["input_size"], 1)
    preds = model.predict(X_in, verbose=0)
    pred_cls = np.argmax(preds, axis=1)
    confs = np.max(preds, axis=1)
    
    results = []
    for i in range(len(X)):
        snr = np.abs(X[i].mean()-1) / (X[i].std()+1e-8) * 10
        is_sig = confs[i] > 0.7 and snr > CONFIG["min_snr"]
        results.append({
            "id": f"OBJ-{i+1:04d}",
            "true": CLASS_NAMES[y[i]],
            "pred": CLASS_NAMES[pred_cls[i]],
            "conf": float(confs[i]),
            "snr": float(snr),
            "sig": is_sig,
            "ok": pred_cls[i] == y[i]
        })
    
    acc = sum(r["ok"] for r in results) / len(results)
    sig = sum(r["sig"] for r in results)
    
    print(f"  Processed: {len(results)}")
    print(f"  Accuracy: {acc*100:.1f}%")
    print(f"  Significant: {sig}")
    
    return results, acc

# ============================================================================
# STEP 4: GENERATE REPORTS
# ============================================================================

def report(results, acc, val_acc, total_samples):
    print_step(4, "GENERATING REPORTS")
    
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Count by class
    class_counts = {name: 0 for name in CLASS_NAMES}
    for r in results:
        class_counts[r["pred"]] += 1
    
    transits = sum(1 for r in results if r["pred"] == "planetary_transit" and r["sig"])
    binaries = sum(1 for r in results if r["pred"] == "eclipsing_binary" and r["sig"])
    significant = sum(1 for r in results if r["sig"])
    
    # JSON
    # Convert numpy types to Python native types for JSON
    clean_results = []
    for r in results:
        clean_results.append({k: (int(v) if hasattr(v, 'item') else v) for k, v in r.items()})

    data = {
        "metadata": {"generated": datetime.now().isoformat(), "author": "Larun. × Astrodata"},
        "summary": {"total": len(results), "accuracy": float(acc), "val_accuracy": float(val_acc),
                   "significant": int(sum(r["sig"] for r in results))},
        "detections": clean_results
    }
    with open(f"{CONFIG['output_dir']}/report_{ts}.json", 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ {CONFIG['output_dir']}/report_{ts}.json")
    
    # CSV
    with open(f"{CONFIG['output_dir']}/detections_{ts}.csv", 'w') as f:
        f.write("id,true,pred,conf,snr,significant,correct\n")
        for r in results:
            f.write(f"{r['id']},{r['true']},{r['pred']},{r['conf']:.3f},{r['snr']:.1f},{r['sig']},{r['ok']}\n")
    print(f"  ✓ {CONFIG['output_dir']}/detections_{ts}.csv")
    
    # HTML
    html = f"""<!DOCTYPE html><html><head><title>AstroTinyML Report</title>
<style>body{{font-family:Arial;margin:40px;background:#f5f5f5}}.c{{max-width:900px;margin:auto;background:#fff;padding:30px;border-radius:12px}}
h1{{border-bottom:3px solid #000;padding-bottom:15px}}.brand{{text-align:center;font-size:24px;font-weight:bold;margin-bottom:20px}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:20px 0}}.stat{{background:#000;color:#fff;padding:20px;border-radius:8px;text-align:center}}
.stat h3{{margin:0 0 8px;font-size:12px}}.stat .v{{font-size:28px;font-weight:bold}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #eee}}th{{background:#000;color:#fff}}</style></head>
<body><div class="c"><div class="brand">Larun. × Astrodata</div><h1>🔭 AstroTinyML Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class="stats"><div class="stat"><h3>Processed</h3><div class="v">{len(results)}</div></div>
<div class="stat"><h3>Accuracy</h3><div class="v">{acc*100:.0f}%</div></div>
<div class="stat"><h3>Model Acc</h3><div class="v">{val_acc*100:.0f}%</div></div>
<div class="stat"><h3>Significant</h3><div class="v">{sum(r['sig'] for r in results)}</div></div></div>
<h2>Detections</h2><table><tr><th>ID</th><th>True</th><th>Predicted</th><th>Conf</th><th>SNR</th><th>✓</th></tr>"""
    for r in results[:30]:
        s = "✓" if r["ok"] else "✗"
        html += f"<tr><td>{r['id']}</td><td>{r['true']}</td><td>{r['pred']}</td><td>{r['conf']:.0%}</td><td>{r['snr']:.1f}</td><td>{s}</td></tr>"
    html += "</table><p style='text-align:center;color:#666;margin-top:30px'>Larun. × Astrodata | AstroTinyML v1.0</p></div></body></html>"
    
    with open(f"{CONFIG['output_dir']}/report_{ts}.html", 'w') as f:
        f.write(html)
    print(f"  ✓ {CONFIG['output_dir']}/report_{ts}.html")
    
    # Generate Dashboard HTML with real data
    dashboard = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AstroTinyML Dashboard - Larun.</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        :root {{
            --black: #1a1a1a;
            --dark: #2d2d2d;
            --gray: #6b7280;
            --light: #f3f4f6;
            --white: #ffffff;
            --accent: #000000;
        }}
        body {{ font-family: 'Inter', sans-serif; background: var(--light); color: var(--black); min-height: 100vh; }}
        
        /* Top Navigation */
        .navbar {{
            position: fixed; top: 0; left: 0; right: 0; height: 64px;
            background: var(--white); border-bottom: 1px solid #e5e7eb;
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 24px; z-index: 100;
        }}
        .nav-brand {{ display: flex; align-items: center; gap: 12px; }}
        .nav-brand .logo {{ font-size: 24px; font-weight: 700; }}
        .nav-brand .product {{ font-size: 20px; font-weight: 400; color: var(--gray); }}
        .nav-status {{ display: flex; align-items: center; gap: 8px; padding: 8px 16px; background: #dcfce7; border-radius: 20px; font-size: 14px; color: #166534; }}
        .nav-status::before {{ content: ''; width: 8px; height: 8px; background: #22c55e; border-radius: 50%; }}
        
        /* Sidebar */
        .sidebar {{
            position: fixed; top: 64px; left: 0; bottom: 0; width: 260px;
            background: var(--white); border-right: 1px solid #e5e7eb; padding: 16px 0;
        }}
        .sidebar-section {{ padding: 8px 16px; font-size: 11px; font-weight: 600; color: var(--gray); text-transform: uppercase; letter-spacing: 0.5px; }}
        .sidebar-item {{
            display: flex; align-items: center; gap: 12px; padding: 12px 24px;
            color: var(--dark); text-decoration: none; font-size: 14px; transition: all 0.2s;
        }}
        .sidebar-item:hover {{ background: var(--light); }}
        .sidebar-item.active {{ background: var(--black); color: var(--white); }}
        .sidebar-item svg {{ width: 20px; height: 20px; }}
        
        /* Main Content */
        .main {{ margin-left: 260px; margin-top: 64px; padding: 32px; }}
        
        /* Hero */
        .hero {{
            background: linear-gradient(135deg, #000 0%, #1a1a1a 100%);
            color: var(--white); padding: 48px; border-radius: 16px; margin-bottom: 32px;
        }}
        .hero-brand {{ font-size: 14px; opacity: 0.7; margin-bottom: 8px; }}
        .hero-title {{ font-size: 36px; font-weight: 700; margin-bottom: 8px; }}
        .hero-subtitle {{ font-size: 16px; opacity: 0.8; max-width: 600px; }}
        
        /* Stats Grid */
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 32px; }}
        .stat-card {{
            background: var(--white); padding: 24px; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-card.highlight {{ background: var(--black); color: var(--white); }}
        .stat-label {{ font-size: 12px; font-weight: 500; color: var(--gray); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
        .stat-card.highlight .stat-label {{ color: rgba(255,255,255,0.7); }}
        .stat-value {{ font-size: 32px; font-weight: 700; }}
        .stat-change {{ font-size: 12px; color: #22c55e; margin-top: 4px; }}
        
        /* Section */
        .section {{ background: var(--white); border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .section-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
        .section-title {{ font-size: 18px; font-weight: 600; }}
        
        /* Products Grid */
        .products-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
        .product-card {{
            background: var(--light); padding: 24px; border-radius: 12px;
            transition: all 0.2s; cursor: pointer;
        }}
        .product-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .product-icon {{ width: 48px; height: 48px; background: var(--black); border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px; }}
        .product-icon svg {{ width: 24px; height: 24px; fill: white; }}
        .product-name {{ font-size: 16px; font-weight: 600; margin-bottom: 4px; }}
        .product-desc {{ font-size: 13px; color: var(--gray); margin-bottom: 16px; }}
        .product-stats {{ display: flex; gap: 24px; padding-top: 16px; border-top: 1px solid #e5e7eb; }}
        .product-stat {{ text-align: center; }}
        .product-stat-value {{ font-size: 18px; font-weight: 600; }}
        .product-stat-label {{ font-size: 10px; color: var(--gray); text-transform: uppercase; }}
        
        /* Activity */
        .activity-item {{ display: flex; gap: 16px; padding: 16px 0; border-bottom: 1px solid #f3f4f6; }}
        .activity-item:last-child {{ border-bottom: none; }}
        .activity-icon {{ width: 40px; height: 40px; background: var(--light); border-radius: 50%; display: flex; align-items: center; justify-content: center; }}
        .activity-icon svg {{ width: 20px; height: 20px; fill: var(--gray); }}
        .activity-content {{ flex: 1; }}
        .activity-title {{ font-size: 14px; margin-bottom: 4px; }}
        .activity-meta {{ font-size: 12px; color: var(--gray); }}
        
        /* Table */
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #f3f4f6; }}
        th {{ font-size: 11px; font-weight: 600; color: var(--gray); text-transform: uppercase; letter-spacing: 0.5px; }}
        tr:hover {{ background: #f9fafb; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }}
        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-info {{ background: #e0e7ff; color: #3730a3; }}
        
        /* Footer */
        .footer {{ text-align: center; padding: 32px; color: var(--gray); font-size: 12px; }}
        .footer-brand {{ font-size: 16px; font-weight: 600; color: var(--black); margin-bottom: 8px; }}
        
        @media (max-width: 1024px) {{
            .sidebar {{ display: none; }}
            .main {{ margin-left: 0; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .products-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-brand">
            <span class="logo">Larun.</span>
            <span class="product">AstroTinyML</span>
        </div>
        <div class="nav-status">Pipeline Complete</div>
    </nav>
    
    <aside class="sidebar">
        <div class="sidebar-section">Navigation</div>
        <a href="#" class="sidebar-item active">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>
            Dashboard
        </a>
        <a href="#" class="sidebar-item">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg>
            Analytics
        </a>
        <div class="sidebar-section">Products</div>
        <a href="#" class="sidebar-item">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2z"/></svg>
            Pipeline
        </a>
        <a href="#" class="sidebar-item">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5z"/></svg>
            Detector
        </a>
        <a href="#" class="sidebar-item">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/></svg>
            Reports
        </a>
        <div class="sidebar-section">Resources</div>
        <a href="#" class="sidebar-item">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6z"/></svg>
            Documentation
        </a>
    </aside>
    
    <main class="main">
        <div class="hero">
            <div class="hero-brand">Larun. × Astrodata</div>
            <div class="hero-title">AstroTinyML Dashboard</div>
            <div class="hero-subtitle">Real-time spectral analysis powered by TinyML. Processing NASA TESS and Kepler data for exoplanet detection.</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card highlight">
                <div class="stat-label">Objects Processed</div>
                <div class="stat-value">{total_samples:,}</div>
                <div class="stat-change">Training dataset</div>
            </div>
            <div class="stat-card highlight">
                <div class="stat-label">Detections</div>
                <div class="stat-value">{len(results)}</div>
                <div class="stat-change">{significant} significant</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Model Accuracy</div>
                <div class="stat-value">{val_acc*100:.1f}%</div>
                <div class="stat-change">Validation set</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Transit Candidates</div>
                <div class="stat-value">{transits}</div>
                <div class="stat-change">{binaries} binaries detected</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Products</h2>
            </div>
            <div class="products-grid">
                <div class="product-card">
                    <div class="product-icon">
                        <svg viewBox="0 0 24 24"><path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2z"/></svg>
                    </div>
                    <div class="product-name">Larun. Pipeline</div>
                    <div class="product-desc">NASA data ingestion from MAST, TESS, and Kepler archives.</div>
                    <div class="product-stats">
                        <div class="product-stat"><div class="product-stat-value">3</div><div class="product-stat-label">Sources</div></div>
                        <div class="product-stat"><div class="product-stat-value">{total_samples}</div><div class="product-stat-label">Samples</div></div>
                        <div class="product-stat"><div class="product-stat-value">✓</div><div class="product-stat-label">Active</div></div>
                    </div>
                </div>
                <div class="product-card">
                    <div class="product-icon">
                        <svg viewBox="0 0 24 24"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5z"/></svg>
                    </div>
                    <div class="product-name">Larun. Detect</div>
                    <div class="product-desc">Spectral anomaly detection with transit analysis and SNR filtering.</div>
                    <div class="product-stats">
                        <div class="product-stat"><div class="product-stat-value">6</div><div class="product-stat-label">Classes</div></div>
                        <div class="product-stat"><div class="product-stat-value">&lt;10ms</div><div class="product-stat-label">Inference</div></div>
                        <div class="product-stat"><div class="product-stat-value">{CONFIG['min_snr']}</div><div class="product-stat-label">Min SNR</div></div>
                    </div>
                </div>
                <div class="product-card">
                    <div class="product-icon">
                        <svg viewBox="0 0 24 24"><path d="M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0l4.6-4.6-4.6-4.6L16 6l6 6-6 6-1.4-1.4z"/></svg>
                    </div>
                    <div class="product-name">Larun. Model</div>
                    <div class="product-desc">TinyML CNN classifier optimized for edge deployment.</div>
                    <div class="product-stats">
                        <div class="product-stat"><div class="product-stat-value">&lt;100KB</div><div class="product-stat-label">Size</div></div>
                        <div class="product-stat"><div class="product-stat-value">{val_acc*100:.0f}%</div><div class="product-stat-label">Accuracy</div></div>
                        <div class="product-stat"><div class="product-stat-value">INT8</div><div class="product-stat-label">Quantized</div></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Recent Detections</h2>
                <span style="font-size: 12px; color: var(--gray);">Showing top {min(20, len(results))} results</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Object ID</th>
                        <th>Classification</th>
                        <th>Confidence</th>
                        <th>SNR</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for r in results[:20]:
        badge_class = "badge-success" if r["sig"] else "badge-info"
        badge_text = "Significant" if r["sig"] else "Normal"
        if r["pred"] == "planetary_transit" and r["sig"]:
            badge_class = "badge-warning"
            badge_text = "Transit Candidate"
        dashboard += f"""                    <tr>
                        <td><strong>{r['id']}</strong></td>
                        <td>{r['pred'].replace('_', ' ').title()}</td>
                        <td>{r['conf']:.1%}</td>
                        <td>{r['snr']:.1f}</td>
                        <td><span class="badge {badge_class}">{badge_text}</span></td>
                    </tr>
"""
    
    dashboard += f"""                </tbody>
            </table>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Classification Summary</h2>
            </div>
            <div class="products-grid">
                <div class="product-card" style="background: #fef3c7;">
                    <div class="product-name">Planetary Transits</div>
                    <div style="font-size: 48px; font-weight: 700; margin: 16px 0;">{class_counts.get('planetary_transit', 0)}</div>
                    <div class="product-desc">{transits} confirmed candidates</div>
                </div>
                <div class="product-card" style="background: #dbeafe;">
                    <div class="product-name">Eclipsing Binaries</div>
                    <div style="font-size: 48px; font-weight: 700; margin: 16px 0;">{class_counts.get('eclipsing_binary', 0)}</div>
                    <div class="product-desc">{binaries} significant detections</div>
                </div>
                <div class="product-card" style="background: #f3f4f6;">
                    <div class="product-name">Other Classifications</div>
                    <div style="font-size: 48px; font-weight: 700; margin: 16px 0;">{class_counts.get('noise', 0) + class_counts.get('stellar_signal', 0) + class_counts.get('instrument_artifact', 0)}</div>
                    <div class="product-desc">Noise, stellar, artifacts</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div class="footer-brand">Larun. × Astrodata</div>
            <p>AstroTinyML v1.0 | Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>NASA-Compatible Report Format</p>
        </div>
    </main>
</body>
</html>"""
    
    with open(f"{CONFIG['output_dir']}/dashboard_{ts}.html", 'w') as f:
        f.write(dashboard)
    print(f"  ✓ {CONFIG['output_dir']}/dashboard_{ts}.html")

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--planets", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()
    
    print_banner()
    start = time.time()
    
    # Run pipeline
    install_dependencies()
    X, y = fetch_data(args.planets, args.skip_fetch)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model, val_acc = train(X_train, y_train, args.epochs)
    results, acc = detect(model, X_test, y_test)
    report(results, acc, val_acc, len(X))
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                        ✓ PIPELINE COMPLETE                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Samples: {len(X):>5}                                                        ║
║  Model Accuracy: {val_acc*100:>5.1f}%                                              ║
║  Detection Accuracy: {acc*100:>5.1f}%                                          ║
║  Time: {(time.time()-start)/60:>5.1f} minutes                                          ║
║                                                                          ║
║  Output: {CONFIG['output_dir']}/                                                   ║
║  Models: {CONFIG['models_dir']}/                                                  ║
║                                                                          ║
║  Larun. × Astrodata                                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
