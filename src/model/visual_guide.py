"""
Visual Pipeline Guide for LARUN.SPACE
======================================
Provides visual representations of the ML pipeline for students and hobbyists.
Includes ASCII diagrams, progress indicators, and educational explanations.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @staticmethod
    def disable():
        """Disable colors for non-terminal output."""
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.DIM = ''


# Model descriptions for educational purposes
MODEL_DESCRIPTIONS = {
    "EXOPLANET-001": {
        "name": "Exoplanet Transit Detector",
        "emoji": "🪐",
        "simple": "Finds planets by detecting tiny dips in starlight",
        "detailed": """When a planet passes in front of its star (a transit),
it blocks a small amount of light. This model learns to recognize these
characteristic dips in brightness, distinguishing them from noise and
other phenomena like starspots.""",
        "input": "1024-point light curve (brightness over time)",
        "output": "Transit probability and classification",
        "fun_fact": "The first exoplanet transit was detected in 1999!"
    },
    "VSTAR-001": {
        "name": "Variable Star Classifier",
        "emoji": "⭐",
        "simple": "Identifies different types of pulsating stars",
        "detailed": """Many stars naturally change brightness over time.
Cepheids pulse like cosmic hearts, RR Lyrae stars are ancient stellar
beacons, and eclipsing binaries are actually two stars dancing together.
This model learns each star's unique 'signature'.""",
        "input": "512-point light curve",
        "output": "Star variability type (Cepheid, RR Lyrae, etc.)",
        "fun_fact": "Cepheids helped prove the universe extends beyond our galaxy!"
    },
    "FLARE-001": {
        "name": "Stellar Flare Detector",
        "emoji": "💥",
        "simple": "Spots explosive eruptions on stars",
        "detailed": """Stars can have explosive magnetic outbursts called
flares, releasing huge amounts of energy. These appear as sudden spikes
in brightness. Understanding stellar flares helps us know which stars
might be too violent to host habitable planets.""",
        "input": "256-point light curve segment",
        "output": "Flare intensity classification",
        "fun_fact": "Some stellar superflares are 10,000x stronger than solar flares!"
    },
    "ASTERO-001": {
        "name": "Asteroseismology Analyzer",
        "emoji": "🔊",
        "simple": "Listens to stars vibrating like cosmic bells",
        "detailed": """Stars ring like bells with many frequencies at once!
By analyzing these oscillations, we can 'see' inside stars and measure
their mass, age, and internal structure. It's like stellar ultrasound.""",
        "input": "512-point power spectrum",
        "output": "Oscillation type classification",
        "fun_fact": "The Sun oscillates with a period of about 5 minutes!"
    },
    "SUPERNOVA-001": {
        "name": "Supernova Detector",
        "emoji": "💫",
        "simple": "Catches exploding stars in distant galaxies",
        "detailed": """When massive stars die, they explode as supernovae,
briefly outshining entire galaxies. Type Ia supernovae are 'standard
candles' that helped us discover dark energy and the accelerating universe.""",
        "input": "128-point light curve",
        "output": "Transient event classification (Type Ia, II, etc.)",
        "fun_fact": "A supernova in our galaxy would be visible in daylight!"
    },
    "GALAXY-001": {
        "name": "Galaxy Classifier",
        "emoji": "🌌",
        "simple": "Sorts galaxies by their shape",
        "detailed": """Galaxies come in beautiful varieties: spirals like
our Milky Way, ellipticals that are smooth and round, and irregulars
that are chaotic. This model learns to recognize these cosmic shapes.""",
        "input": "64x64 pixel galaxy image",
        "output": "Morphology type (spiral, elliptical, etc.)",
        "fun_fact": "There are more galaxies than grains of sand on Earth!"
    },
    "SPECTYPE-001": {
        "name": "Spectral Type Classifier",
        "emoji": "🌈",
        "simple": "Identifies star types by their colors",
        "detailed": """Stars are classified O-B-A-F-G-K-M-L from hottest
(blue) to coolest (red). Our Sun is a G-type star. Each type has unique
properties - hot O stars live fast and die young, while cool M dwarfs
can shine for trillions of years.""",
        "input": "8 photometric color measurements",
        "output": "Spectral type (O, B, A, F, G, K, M, L)",
        "fun_fact": "Astronomers remember the sequence: Oh Be A Fine Girl/Guy, Kiss Me Lovingly!"
    },
    "MICROLENS-001": {
        "name": "Microlensing Detector",
        "emoji": "🔍",
        "simple": "Finds hidden objects using gravity's magnifying glass",
        "detailed": """Einstein predicted that gravity bends light. When a
massive object passes in front of a distant star, it acts as a cosmic
magnifying glass, temporarily brightening the star. This can reveal
invisible objects like black holes and rogue planets.""",
        "input": "512-point light curve",
        "output": "Lensing event type",
        "fun_fact": "Microlensing found a planet 13,000 light-years away!"
    }
}


class PipelineVisualizer:
    """
    Creates visual representations of the ML pipeline.
    Designed for students and hobbyists to understand the analysis flow.
    """

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        if not use_colors:
            Colors.disable()

    def print_header(self):
        """Print the LARUN.SPACE header."""
        header = f"""
{Colors.CYAN}{Colors.BOLD}
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🔭  LARUN.SPACE - Democratizing Space Discovery  🌟        ║
    ║                                                               ║
    ║       TinyML-Powered Exoplanet Detection Platform             ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
        print(header)

    def print_model_card(self, model_id: str):
        """Print an educational card about a model."""
        if model_id not in MODEL_DESCRIPTIONS:
            print(f"Unknown model: {model_id}")
            return

        info = MODEL_DESCRIPTIONS[model_id]
        card = f"""
{Colors.BOLD}╭─────────────────────────────────────────────────────────────────╮
│  {info['emoji']}  {info['name']:<52}  │
│     Model ID: {model_id:<48}  │
├─────────────────────────────────────────────────────────────────┤{Colors.ENDC}
│  {Colors.CYAN}What it does:{Colors.ENDC}
│  {info['simple']:<62}
│
│  {Colors.CYAN}How it works:{Colors.ENDC}
{self._wrap_text(info['detailed'], 64)}
│
│  {Colors.GREEN}Input:{Colors.ENDC} {info['input']:<54}
│  {Colors.GREEN}Output:{Colors.ENDC} {info['output']:<53}
│
│  {Colors.YELLOW}Fun Fact:{Colors.ENDC} {info['fun_fact']:<53}
{Colors.BOLD}╰─────────────────────────────────────────────────────────────────╯{Colors.ENDC}
"""
        print(card)

    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to fit within a box."""
        words = text.split()
        lines = []
        current_line = "│  "

        for word in words:
            if len(current_line) + len(word) + 1 <= width + 3:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = "│  " + word + " "

        if current_line.strip():
            lines.append(current_line)

        return "\n".join(lines)

    def print_pipeline_diagram(self, pipeline_type: str = "exoplanet"):
        """Print ASCII diagram of the pipeline."""

        if pipeline_type == "exoplanet":
            diagram = f"""
{Colors.BOLD}═══════════════════════════════════════════════════════════════════
                    EXOPLANET DETECTION PIPELINE
═══════════════════════════════════════════════════════════════════{Colors.ENDC}

    {Colors.CYAN}📥 INPUT: Light Curve Data{Colors.ENDC}
         │
         ▼
    ╔════════════════════════════════════════════════════════════╗
    ║  {Colors.YELLOW}STAGE 1: PARALLEL INITIAL ANALYSIS{Colors.ENDC}                       ║
    ║                                                            ║
    ║     ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐ ║
    ║     │ 🪐 EXOPLANET-001│  │ ⭐ VSTAR-001    │  │💥 FLARE  │ ║
    ║     │ Transit Detect  │  │ Variable Star  │  │  Detect  │ ║
    ║     └────────┬────────┘  └────────┬───────┘  └────┬─────┘ ║
    ║              │                    │               │       ║
    ║              └────────────────────┴───────────────┘       ║
    ╚════════════════════════════════════════════════════════════╝
                              │
                              ▼
    ╔════════════════════════════════════════════════════════════╗
    ║  {Colors.GREEN}STAGE 2: HUMAN CHECKPOINT 👤{Colors.ENDC}                            ║
    ║                                                            ║
    ║     "Review initial results. Proceed with validation?"     ║
    ║     [Proceed] [Modify] [Reject] [Expert Review]           ║
    ╚════════════════════════════════════════════════════════════╝
                              │
                              ▼
    ╔════════════════════════════════════════════════════════════╗
    ║  {Colors.YELLOW}STAGE 3: VALIDATION ENSEMBLE{Colors.ENDC}                            ║
    ║                                                            ║
    ║     ┌───────────────┐  ┌───────────────┐  ┌─────────────┐ ║
    ║     │ Odd-Even Test │  │ V-Shape Test  │  │ Secondary   │ ║
    ║     │   ✓ / ✗       │  │    ✓ / ✗      │  │ Eclipse ✓/✗ │ ║
    ║     └───────────────┘  └───────────────┘  └─────────────┘ ║
    ║                                                            ║
    ║     Combined FPP: ████████░░ 82%  [Likely Planet]         ║
    ╚════════════════════════════════════════════════════════════╝
                              │
                              ▼
    ╔════════════════════════════════════════════════════════════╗
    ║  {Colors.GREEN}STAGE 4: FINAL HUMAN REVIEW 👤{Colors.ENDC}                          ║
    ║                                                            ║
    ║     "Confirm detection status:"                           ║
    ║     [Validated Planet] [False Positive] [Needs Followup]  ║
    ╚════════════════════════════════════════════════════════════╝
                              │
                              ▼
    {Colors.GREEN}📤 OUTPUT: Detection Result + Validation Report{Colors.ENDC}

═══════════════════════════════════════════════════════════════════
"""
        elif pipeline_type == "stellar":
            diagram = f"""
{Colors.BOLD}═══════════════════════════════════════════════════════════════════
                   STELLAR CLASSIFICATION PIPELINE
═══════════════════════════════════════════════════════════════════{Colors.ENDC}

    {Colors.CYAN}📥 INPUT: Star Observations{Colors.ENDC}
         │
         ▼
    ╔════════════════════════════════════════════════════════════╗
    ║  {Colors.YELLOW}PARALLEL MULTI-ASPECT ANALYSIS{Colors.ENDC}                          ║
    ║                                                            ║
    ║     ┌─────────────────┐  ┌─────────────────┐              ║
    ║     │🌈 SPECTYPE-001  │  │ ⭐ VSTAR-001    │              ║
    ║     │ Spectral Type   │  │ Variability     │              ║
    ║     │ (O,B,A,F,G,K,M) │  │ Classification  │              ║
    ║     └────────┬────────┘  └────────┬────────┘              ║
    ║              │                    │                        ║
    ║              │    ┌───────────────┘                        ║
    ║              │    │    ┌─────────────────┐                 ║
    ║              │    │    │🔊 ASTERO-001    │                 ║
    ║              │    │    │ Asteroseismology│                 ║
    ║              │    │    │ (stellar 'sound')│                ║
    ║              │    │    └────────┬────────┘                 ║
    ║              └────┴─────────────┘                          ║
    ╚════════════════════════════════════════════════════════════╝
                              │
                              ▼
    {Colors.GREEN}📤 OUTPUT: Complete Stellar Profile{Colors.ENDC}
         • Spectral Type: G2V (like our Sun)
         • Variability: Rotational (starspots)
         • Oscillations: Solar-like (5-min period)

═══════════════════════════════════════════════════════════════════
"""
        else:  # transient
            diagram = f"""
{Colors.BOLD}═══════════════════════════════════════════════════════════════════
                   TRANSIENT DETECTION PIPELINE
═══════════════════════════════════════════════════════════════════{Colors.ENDC}

    {Colors.CYAN}📥 INPUT: Time-Series Data{Colors.ENDC}
         │
         ▼
    ╔════════════════════════════════════════════════════════════╗
    ║  {Colors.YELLOW}PARALLEL TRANSIENT SEARCH{Colors.ENDC}                               ║
    ║                                                            ║
    ║     ┌─────────────────┐  ┌─────────────────┐              ║
    ║     │💫 SUPERNOVA-001 │  │💥 FLARE-001     │              ║
    ║     │ Exploding Stars │  │ Stellar Flares  │              ║
    ║     └────────┬────────┘  └────────┬────────┘              ║
    ║              │                    │                        ║
    ║              │    ┌───────────────┘                        ║
    ║              │    │    ┌─────────────────┐                 ║
    ║              │    │    │🔍 MICROLENS-001 │                 ║
    ║              │    │    │ Gravity Lensing │                 ║
    ║              │    │    └────────┬────────┘                 ║
    ║              └────┴─────────────┘                          ║
    ╚════════════════════════════════════════════════════════════╝
                              │
                              ▼
    ╔════════════════════════════════════════════════════════════╗
    ║  {Colors.GREEN}HUMAN ALERT REVIEW 👤{Colors.ENDC}                                   ║
    ║                                                            ║
    ║     ⚠️  TRANSIENT DETECTED!                                ║
    ║     [Confirm Alert] [False Alarm] [Need More Data]        ║
    ╚════════════════════════════════════════════════════════════╝
                              │
                              ▼
    {Colors.GREEN}📤 OUTPUT: Alert + Classification{Colors.ENDC}

═══════════════════════════════════════════════════════════════════
"""
        print(diagram)

    def print_progress_bar(self, current: int, total: int, label: str = "",
                          width: int = 40, status: str = "running"):
        """Print a visual progress bar."""
        filled = int(width * current / total) if total > 0 else 0
        empty = width - filled

        if status == "completed":
            bar_char = "█"
            color = Colors.GREEN
        elif status == "running":
            bar_char = "▓"
            color = Colors.YELLOW
        else:
            bar_char = "░"
            color = Colors.RED

        bar = color + bar_char * filled + Colors.DIM + "░" * empty + Colors.ENDC
        percent = (current / total * 100) if total > 0 else 0

        print(f"    {label:20} [{bar}] {percent:5.1f}%")

    def print_model_status(self, statuses: Dict[str, str]):
        """Print status of all models in the pipeline."""
        print(f"\n{Colors.BOLD}Model Status:{Colors.ENDC}\n")

        status_icons = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "awaiting_human": "👤"
        }

        for model_id, status in statuses.items():
            icon = status_icons.get(status, "❓")
            info = MODEL_DESCRIPTIONS.get(model_id, {})
            name = info.get("name", model_id)
            emoji = info.get("emoji", "")

            if status == "completed":
                color = Colors.GREEN
            elif status == "running":
                color = Colors.YELLOW
            elif status == "failed":
                color = Colors.RED
            else:
                color = Colors.DIM

            print(f"    {icon} {emoji} {color}{name:<30}{Colors.ENDC} [{status}]")

    def print_result_summary(self, result: Dict[str, Any]):
        """Print a user-friendly result summary."""
        print(f"""
{Colors.BOLD}╔════════════════════════════════════════════════════════════════╗
║                       ANALYSIS RESULTS                         ║
╚════════════════════════════════════════════════════════════════╝{Colors.ENDC}
""")

        if "detection" in result:
            det = result["detection"]
            if det:
                print(f"    {Colors.GREEN}🎉 DETECTION: Potential Planet Candidate!{Colors.ENDC}")
            else:
                print(f"    {Colors.YELLOW}📊 DETECTION: No significant signal found{Colors.ENDC}")

        if "confidence" in result:
            conf = result["confidence"] * 100
            self.print_progress_bar(int(conf), 100, "Confidence", status="completed")

        if "classification" in result:
            print(f"\n    Classification: {Colors.CYAN}{result['classification']}{Colors.ENDC}")

        if "validation" in result:
            val = result["validation"]
            print(f"\n    {Colors.BOLD}Validation Tests:{Colors.ENDC}")
            for test, passed in val.items():
                icon = "✅" if passed else "❌"
                print(f"      {icon} {test}")

        if "next_steps" in result:
            print(f"\n    {Colors.BOLD}Recommended Next Steps:{Colors.ENDC}")
            for i, step in enumerate(result["next_steps"], 1):
                print(f"      {i}. {step}")

    def print_human_checkpoint(self, question: str, options: List[str],
                               context: Optional[Dict] = None):
        """Print a human checkpoint interface."""
        print(f"""
{Colors.BOLD}{Colors.YELLOW}╔════════════════════════════════════════════════════════════════╗
║                    👤 HUMAN REVIEW NEEDED                      ║
╚════════════════════════════════════════════════════════════════╝{Colors.ENDC}

    {Colors.BOLD}Question:{Colors.ENDC}
    {question}

    {Colors.BOLD}Options:{Colors.ENDC}""")

        for i, option in enumerate(options, 1):
            print(f"      [{i}] {option}")

        if context:
            print(f"\n    {Colors.DIM}Context:{Colors.ENDC}")
            for key, value in context.items():
                print(f"      • {key}: {value}")

        print(f"""
    {Colors.CYAN}Enter your choice (1-{len(options)}):{Colors.ENDC} """, end="")

    def print_all_models(self):
        """Print cards for all available models."""
        print(f"""
{Colors.BOLD}═══════════════════════════════════════════════════════════════════
                    AVAILABLE TINYML MODELS
═══════════════════════════════════════════════════════════════════{Colors.ENDC}
""")
        for model_id in MODEL_DESCRIPTIONS:
            self.print_model_card(model_id)

    def print_quick_reference(self):
        """Print a quick reference guide."""
        print(f"""
{Colors.BOLD}═══════════════════════════════════════════════════════════════════
                    QUICK REFERENCE GUIDE
═══════════════════════════════════════════════════════════════════{Colors.ENDC}

    {Colors.BOLD}What is LARUN.SPACE?{Colors.ENDC}
    A platform that uses tiny AI models to help discover exoplanets!
    Our models are small enough to run on a Raspberry Pi, yet powerful
    enough to find planets around distant stars.

    {Colors.BOLD}How does it work?{Colors.ENDC}
    1. 📥 You provide light curve data (star brightness over time)
    2. 🤖 AI models analyze the data in parallel
    3. 👤 You review the results and make decisions
    4. 📤 Get a validated detection report

    {Colors.BOLD}Key Concepts:{Colors.ENDC}

    {Colors.CYAN}Transit{Colors.ENDC} - When a planet passes in front of its star,
            blocking some light. We detect this tiny dip!

    {Colors.CYAN}False Positive{Colors.ENDC} - Something that looks like a planet but isn't.
            Could be a binary star or instrument glitch.

    {Colors.CYAN}Validation{Colors.ENDC} - Tests to confirm a detection is real.
            We check: odd-even depths, V-shape, secondary eclipses.

    {Colors.CYAN}FPP{Colors.ENDC} - False Positive Probability. If FPP < 1.5%,
            we can call it a validated planet!

    {Colors.BOLD}Pipeline Types:{Colors.ENDC}

    🪐 {Colors.CYAN}Exoplanet Detection{Colors.ENDC}
       Find new worlds around other stars

    ⭐ {Colors.CYAN}Stellar Classification{Colors.ENDC}
       Understand the star hosting potential planets

    💫 {Colors.CYAN}Transient Detection{Colors.ENDC}
       Catch cosmic explosions and rare events

    {Colors.BOLD}Need Help?{Colors.ENDC}
    • Type 'help' for commands
    • Type 'models' to see all AI models
    • Type 'pipeline [name]' to see pipeline diagram

═══════════════════════════════════════════════════════════════════
""")


def generate_web_component() -> str:
    """Generate React component for web interface visualization."""
    return '''
// PipelineVisualizer.tsx - React component for LARUN.SPACE
// This component provides visual pipeline representation for the web interface

import React, { useState } from 'react';

interface ModelInfo {
  id: string;
  name: string;
  emoji: string;
  simple: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'awaiting_human';
  confidence?: number;
}

interface PipelineVisualizerProps {
  models: ModelInfo[];
  currentStage: number;
  onHumanResponse?: (decision: string) => void;
  humanCheckpoint?: {
    question: string;
    options: string[];
  };
}

const StatusIcon: React.FC<{ status: string }> = ({ status }) => {
  const icons: Record<string, string> = {
    pending: '⏳',
    running: '🔄',
    completed: '✅',
    failed: '❌',
    awaiting_human: '👤'
  };
  return <span className="text-2xl">{icons[status] || '❓'}</span>;
};

const ConfidenceBar: React.FC<{ value: number }> = ({ value }) => (
  <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
    <div
      className={`h-2.5 rounded-full ${
        value >= 0.8 ? 'bg-green-600' : value >= 0.5 ? 'bg-yellow-600' : 'bg-red-600'
      }`}
      style={{ width: `${value * 100}%` }}
    />
  </div>
);

const ModelCard: React.FC<{ model: ModelInfo }> = ({ model }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`border rounded-lg p-4 cursor-pointer transition-all ${
        model.status === 'running' ? 'border-yellow-500 shadow-lg' :
        model.status === 'completed' ? 'border-green-500' :
        model.status === 'failed' ? 'border-red-500' :
        'border-gray-300'
      }`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-3xl">{model.emoji}</span>
          <div>
            <h3 className="font-bold">{model.name}</h3>
            <p className="text-sm text-gray-500">{model.id}</p>
          </div>
        </div>
        <StatusIcon status={model.status} />
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t">
          <p className="text-sm mb-3">{model.simple}</p>
          {model.confidence !== undefined && (
            <div>
              <span className="text-sm text-gray-500">Confidence: {(model.confidence * 100).toFixed(1)}%</span>
              <ConfidenceBar value={model.confidence} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const HumanCheckpoint: React.FC<{
  question: string;
  options: string[];
  onResponse: (decision: string) => void;
}> = ({ question, options, onResponse }) => (
  <div className="bg-yellow-50 border-2 border-yellow-400 rounded-lg p-6 my-4">
    <div className="flex items-center gap-2 mb-4">
      <span className="text-3xl">👤</span>
      <h3 className="font-bold text-lg">Human Review Needed</h3>
    </div>
    <p className="mb-4">{question}</p>
    <div className="flex flex-wrap gap-2">
      {options.map((option) => (
        <button
          key={option}
          onClick={() => onResponse(option)}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          {option}
        </button>
      ))}
    </div>
  </div>
);

export const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({
  models,
  currentStage,
  onHumanResponse,
  humanCheckpoint
}) => {
  const stages = [
    { name: 'Initial Analysis', description: 'Parallel model execution' },
    { name: 'Human Review', description: 'Review initial results' },
    { name: 'Validation', description: 'Ensemble validation tests' },
    { name: 'Final Review', description: 'Confirm detection' }
  ];

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">🔭 LARUN.SPACE</h1>
        <p className="text-gray-600">Democratizing Space Discovery</p>
      </div>

      {/* Stage Progress */}
      <div className="flex justify-between mb-8">
        {stages.map((stage, index) => (
          <div
            key={stage.name}
            className={`flex-1 text-center ${
              index <= currentStage ? 'text-blue-600' : 'text-gray-400'
            }`}
          >
            <div className={`w-8 h-8 mx-auto rounded-full flex items-center justify-center ${
              index < currentStage ? 'bg-green-500 text-white' :
              index === currentStage ? 'bg-blue-500 text-white' :
              'bg-gray-200'
            }`}>
              {index < currentStage ? '✓' : index + 1}
            </div>
            <p className="text-sm mt-2 font-medium">{stage.name}</p>
            <p className="text-xs">{stage.description}</p>
          </div>
        ))}
      </div>

      {/* Human Checkpoint */}
      {humanCheckpoint && onHumanResponse && (
        <HumanCheckpoint
          question={humanCheckpoint.question}
          options={humanCheckpoint.options}
          onResponse={onHumanResponse}
        />
      )}

      {/* Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {models.map((model) => (
          <ModelCard key={model.id} model={model} />
        ))}
      </div>

      {/* Legend */}
      <div className="mt-8 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-bold mb-2">Status Legend</h4>
        <div className="flex flex-wrap gap-4 text-sm">
          <span>⏳ Pending</span>
          <span>🔄 Running</span>
          <span>✅ Completed</span>
          <span>❌ Failed</span>
          <span>👤 Needs Human Review</span>
        </div>
      </div>
    </div>
  );
};

export default PipelineVisualizer;
'''


if __name__ == "__main__":
    viz = PipelineVisualizer()

    viz.print_header()
    viz.print_quick_reference()

    print("\n\n")
    viz.print_pipeline_diagram("exoplanet")

    print("\n\n")
    viz.print_all_models()

    # Example status display
    print("\n\n")
    viz.print_model_status({
        "EXOPLANET-001": "completed",
        "VSTAR-001": "completed",
        "FLARE-001": "running",
        "ASTERO-001": "pending"
    })

    # Example result summary
    print("\n\n")
    viz.print_result_summary({
        "detection": True,
        "confidence": 0.87,
        "classification": "Planetary Transit",
        "validation": {
            "Odd-Even Test": True,
            "V-Shape Test": True,
            "Secondary Eclipse": True
        },
        "next_steps": [
            "Submit to ExoFOP for follow-up observations",
            "Generate CTOI submission package",
            "Request radial velocity measurements"
        ]
    })
