#!/bin/bash
# =============================================================================
# Baseline Experiments Runner
# =============================================================================
# Usage:
#   ./run_baseline.sh           # Run all missing experiments
#   ./run_baseline.sh --check   # Check experiment status only
#   ./run_baseline.sh --analyze # Generate analysis report only
# =============================================================================

cd "$(dirname "$0")"

echo "========================================"
echo "Federated Learning Baseline Experiments"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Activate conda environment if exists
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate fedgpro 2>/dev/null || echo "[INFO] Using default Python environment"
fi

# Run experiments
if [ "$1" == "--check" ]; then
    echo "[MODE] Checking experiment status..."
    python run_baseline_experiments.py --check
elif [ "$1" == "--analyze" ]; then
    echo "[MODE] Generating analysis report..."
    python run_baseline_experiments.py --analyze
else
    echo "[MODE] Running all missing experiments..."
    python run_baseline_experiments.py
fi

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
