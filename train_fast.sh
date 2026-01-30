#!/bin/bash
# LARUN Fast Training Launcher
# Uses distributed training with parallel data fetching

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LARUN Fast Training                                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check for required packages
python3 -c "import lightkurve" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -q lightkurve astroquery tensorflow scikit-learn
}

# Default settings (can be overridden)
WORKERS=${1:-8}
PLANETS=${2:-50}
EPOCHS=${3:-50}

echo ""
echo "Settings:"
echo "  Workers: $WORKERS (parallel downloads)"
echo "  Planets: $PLANETS"
echo "  Epochs:  $EPOCHS"
echo ""

# Run distributed training
python3 distributed/train_distributed.py \
    --workers $WORKERS \
    --planets $PLANETS \
    --non-planets $PLANETS \
    --epochs $EPOCHS \
    --output ./models/distributed

echo ""
echo "Training complete! Models saved to ./models/distributed"
