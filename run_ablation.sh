#!/bin/bash
# =============================================================================
# Ablation Experiments Runner
# =============================================================================
# Usage:
#   ./run_ablation.sh           # Run all missing experiments
#   ./run_ablation.sh --check   # Check experiment status only
#   ./run_ablation.sh --analyze # Generate analysis report only
# =============================================================================

cd "$(dirname "$0")"

echo "========================================"
echo "FedGpro Ablation Study Experiments"
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
    python run_ablation_experiments.py --check
elif [ "$1" == "--analyze" ]; then
    echo "[MODE] Generating analysis report..."
    python run_ablation_experiments.py --analyze
else
    echo "[MODE] Running all missing experiments..."
    python run_ablation_experiments.py
fi

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
