# FedGpro: Privacy-Preserving Federated Learning for Credit Scoring

## Overview

FedGpro is a privacy-preserving federated learning framework for credit scoring, combining VAE-based data augmentation, prototype learning, and importance-aware differential privacy.

## Project Structure

```
FedGpro/
├── run_baseline_experiments.py    # Baseline experiment runner
├── run_ablation_experiments.py    # Ablation experiment runner
├── dataset/                       # Dataset generation
│   ├── generate_all_datasets_auto.py
│   ├── generate_Uci.py
│   └── generate_Xinwang.py
├── system/                        # Core system
│   ├── main.py                    # Main entry point
│   ├── flcore/                    # FL core modules
│   │   ├── servers/               # Server implementations
│   │   ├── clients/               # Client implementations
│   │   └── trainmodel/            # Model definitions
│   ├── utils/                     # Utilities
│   │   └── analyze_results.py     # Result analysis
│   └── results/                   # Experiment results
└── docs/                          # Documentation
```

## Quick Start

### Environment Setup

```bash
conda env create -f env_cuda_latest.yaml
conda activate fedgpro
```

### Dataset Generation

```bash
python dataset/generate_all_datasets_auto.py
```

### Run Experiments

```bash
# Run baseline experiments
python run_baseline_experiments.py

# Check missing experiments
python run_baseline_experiments.py --check

# Generate analysis report only
python run_baseline_experiments.py --analyze

# Run ablation experiments
python run_ablation_experiments.py
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| FedAvg | Classic federated averaging |
| FedProx | Federated learning with proximal regularization |
| FedProto | Prototype-based federated learning |
| FedGpro | Proposed method (VAE + Prototype + Adaptive DP) |

## Ablation Configurations

| Config | Description |
|--------|-------------|
| Full_Model | Complete model (baseline) |
| No_VAE_Generation | Disable VAE data generation |
| No_Prototype | Disable prototype learning |
| Privacy_Epsilon_1.0 | Privacy budget ε=1.0 |
| Privacy_Epsilon_10.0 | Privacy budget ε=10.0 |
| Privacy_Utility_First | Utility-first adaptive encryption |
| Privacy_Privacy_First | Privacy-first adaptive encryption |
| Generalization_Reserve_2 | Reserve 20% clients for generalization |

## Datasets

| Dataset | Samples | Features | Positive Ratio |
|---------|---------|----------|----------------|
| UCI | 30,000 | 23 | 22% |
| Xinwang | 50,000+ | 37 | ~4% |

## Heterogeneity Types

- **Feature**: Feature distribution heterogeneity
- **Label**: Label distribution heterogeneity
- **Quantity**: Sample quantity heterogeneity
- **IID**: Independent and identically distributed (control)

## Output Files

Results are saved in `system/results/汇总/`:

```
汇总/
├── baseline_experiments.xlsx
├── ablation_experiments.xlsx
├── figures/
│   ├── ablation_convergence_*.png
│   └── generalization_ablation_*.png
└── heterogeneity_plots/
    ├── label_heterogeneity.png
    ├── feature_heterogeneity.png
    └── quantity_heterogeneity.png
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| -gr | Global rounds | 100 |
| -ls | Local epochs | 5 |
| -nc | Number of clients | 10 |
| -lr | Learning rate | 0.005-0.007 |
| -lbs | Batch size | 64/128 |
| -t | Repeat times | 5 |

## License

MIT License

