# VAE Models for FedVPS

## Overview

This directory contains Variational Autoencoder (VAE) models specifically designed for the FedVPS (Federated VAE-Prototype Sharing) algorithm.

## File Structure

- **credit_vae.py**: VAE implementations for credit scoring datasets
  - `CreditVAE`: Universal VAE class supporting both UCI (23 features) and Xinwang (100 features)
  - `CreditVAEWithClassifier`: Joint VAE + Classifier for Phase 1 training
  - `create_credit_vae()`: Factory function for automatic VAE configuration

## Import Path

The VAE models are imported in FedVPS client as follows:

```python
from flcore.trainmodel.credit_vae import CreditVAE, create_credit_vae
```

This import path is used in:
- `system/flcore/clients/clientvps.py` (Client-side Phase 1 training)

## Architecture

### UCI Credit Card Dataset (23 features)
```
Encoder: 23 → 64 → 32 → 16 (latent)
Decoder: 16 → 32 → 64 → 23
```

### Xinwang Dataset (100 features)
```
Encoder: 100 → 128 → 64 → 32 (latent)
Decoder: 32 → 64 → 128 → 100
```

## Usage Example

```python
# Automatic configuration based on dataset
from flcore.trainmodel.credit_vae import create_credit_vae

# For UCI dataset
vae_uci = create_credit_vae(
    input_dim=23, 
    latent_dim=16,  # Auto-configured if None
    dataset_name='UCI'
)

# For Xinwang dataset
vae_xinwang = create_credit_vae(
    input_dim=100,
    latent_dim=32,  # Auto-configured if None
    dataset_name='Xinwang'
)
```

## Integration with FedVPS

The VAE models are automatically loaded when running FedVPS algorithm:

```bash
# UCI dataset
python main.py -data Uci -m credit_uci -algo FedVPS -gr 200 -did 0

# Xinwang dataset
python main.py -data Xinwang -m credit_xinwang -algo FedVPS -gr 200 -did 0
```

The VAE architecture is automatically selected based on the classifier's input dimension.

## Key Features

1. **Auto-configuration**: Automatically adapts to dataset dimensions
2. **Reparameterization trick**: Stable gradient flow during training
3. **Loss components**: Reconstruction (MSE) + KL divergence
4. **Sampling capability**: Generate synthetic data from learned distribution

## References

- Kingma & Welling (2014). "Auto-Encoding Variational Bayes". ICLR 2014.
- Xu et al. (2019). "Modeling Tabular data using Conditional GAN". NeurIPS 2019.

---

**Last Updated**: 2025-12-16
