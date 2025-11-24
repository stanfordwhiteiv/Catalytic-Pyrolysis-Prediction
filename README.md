# Physics-Informed Transformer for TGA Prediction

A PyTorch implementation of a transformer model with **cross-attention** and **physics-informed loss** for thermogravimetric analysis (TGA) char yield prediction.

## Key Features

- **Cross-Attention Architecture**: Novel transformer design where dynamic time-series features (temperature, previous char values) attend to static material properties via cross-attention, enabling better feature integration than simple concatenation.

- **Physics-Informed Loss**: Domain-specific constraints ensure physically plausible predictions:
  - **Monotonicity penalty**: Char yield must decrease with increasing temperature
  - **Over-penalty**: Char yield cannot exceed 100%

- **Autoregressive Prediction**: Model predicts full TGA curves by feeding predictions back as inputs.

## Architecture Overview

```
                    ┌─────────────────────┐
                    │   Static Features   │
                    │ (Material Properties)│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Linear Projection   │
                    └──────────┬───────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    │   ┌──────────────────┐   │   ┌──────────────────┐   │
    │   │ Dynamic Features │   │   │                  │   │
    │   │ (Temp, prev_char)│   │   │                  │   │
    │   └────────┬─────────┘   │   │                  │   │
    │            │             │   │                  │   │
    │            ▼             │   │                  │   │
    │   ┌──────────────────┐   │   │                  │   │
    │   │Linear + Pos Enc  │   │   │                  │   │
    │   └────────┬─────────┘   │   │                  │   │
    │            │             │   │                  │   │
    │            ▼             │   │                  │   │
    │   ┌──────────────────┐   │   │                  │   │
    │   │  Self-Attention  │   │   │                  │   │
    │   └────────┬─────────┘   │   │                  │   │
    │            │             │   │                  │   │
    │            ▼             ▼   │                  │   │
    │   ┌──────────────────────────┐                  │   │
    │   │     Cross-Attention      │◄─────────────────┘   │
    │   │ (Query: Dynamic,         │                      │
    │   │  Key/Value: Static)      │                      │
    │   └────────┬─────────────────┘                      │
    │            │                                        │
    │            ▼                                        │
    │   ┌──────────────────┐                              │
    │   │   Feedforward    │                              │
    │   └────────┬─────────┘                              │
    │            │                                        │
    └────────────┼────────────────────────────────────────┘
                 │  (Repeat for N layers)
                 ▼
        ┌──────────────────┐
        │  Output Linear   │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Predicted Char  │
        └──────────────────┘
```

## Installation

```bash
git clone https://github.com/yourusername/physics-informed-tga-transformer.git
cd physics-informed-tga-transformer
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Training

```python
from train import train_model
from config import get_default_config

# Configure and train
config = get_default_config()
config.training.epochs = 100
config.training.use_physics_loss = True

model, history, preprocessor = train_model(
    data_folder="./data",
    config=config
)
```

Or use the command line:
```bash
python train.py --data_folder ./data --epochs 100 --use_physics_loss
```

### Evaluation

```python
from evaluate import load_model_from_checkpoint, evaluate_on_curves

# Load trained model
model, config = load_model_from_checkpoint("checkpoints/best_model.pt")

# Evaluate
results = evaluate_on_curves(model, test_df, preprocessor)
print(f"R² Score: {results['R2']:.4f}")
```

### Using the Model Directly

```python
import torch
from models import TransformerWithStaticNet

# Create model
model = TransformerWithStaticNet(
    dynamic_input_dim=2,    # (temperature, prev_char)
    static_input_dim=8,     # material properties
    hidden_dim=64,
    num_layers=2,
    nhead=4,
    dim_feedforward=16
)

# Forward pass
dynamic_x = torch.randn(32, 10, 2)  # (batch, seq_len, features)
static_x = torch.randn(32, 8)       # (batch, static_features)
predictions = model(dynamic_x, static_x)  # (batch, 1)
```

## Data Format

The model expects TGA data in CSV format with:

- **Temperature column**: Temperature values (e.g., `Temp`)
- **Target column**: Char yield percentage (default: `Char`)
- **Static features**: Material properties (e.g., graphite content, hardener type)
- **Filename column**: For grouping measurements from the same sample

Example data structure:
```
Temp,Char,graphite_content,hardener,ct,ct2,...
200,95.2,0.15,A,1,0,...
250,92.1,0.15,A,1,0,...
300,88.5,0.15,A,1,0,...
```

## Configuration

Customize training via the `Config` class:

```python
from config import Config, DataConfig, ModelConfig, TrainingConfig

config = Config(
    data=DataConfig(
        sequence_length=10,
        target_column="Char"
    ),
    model=ModelConfig(
        hidden_dim=64,
        num_layers=2,
        use_cross_attention=True
    ),
    training=TrainingConfig(
        epochs=150,
        use_physics_loss=True,
        monotonicity_lambda=10.0,
        over_penalty_lambda=1000.0
    )
)
```

## Ablation Studies

Run ablation experiments to compare architectures:

```python
from config import get_ablation_config
from train import train_model

# Full model (cross-attention + physics loss)
config_full = get_ablation_config(use_cross_attention=True, use_physics_loss=True)

# Without physics loss
config_no_physics = get_ablation_config(use_cross_attention=True, use_physics_loss=False)

# Without cross-attention (concatenation baseline)
config_concat = get_ablation_config(use_cross_attention=False, use_physics_loss=True)
```

## Model Variants

| Model | Cross-Attention | Physics Loss | Description |
|-------|----------------|--------------|-------------|
| `TransformerWithStaticNet` | ✓ | Optional | Full model with cross-attention |
| `TransformerStandardNet` | ✗ | Optional | Baseline with feature concatenation |

## Physics-Informed Loss

The physics loss ensures predictions respect domain constraints:

```python
from models import compute_physics_loss

# During training
predictions = model(dynamic_x, static_x)
over_penalty, mono_penalty = compute_physics_loss(predictions, dynamic_x)

total_loss = mae_loss + 1000 * over_penalty + 10 * mono_penalty
```

**Constraints:**
1. **Over-penalty**: `ReLU(prediction - 100)` — penalizes predictions > 100%
2. **Monotonicity**: `ReLU(prediction - prev_char)` — penalizes increasing char with temperature

## Project Structure

```
physics-informed-tga-transformer/
├── __init__.py          # Package exports
├── models.py            # Neural network architectures
├── data_utils.py        # Data loading and preprocessing
├── train.py             # Training loop and utilities
├── evaluate.py          # Evaluation and visualization
├── config.py            # Configuration dataclasses
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Citation

If you use this code in your research, please cite:

>will add bibtex when this research finds a home :)

## Acknowledgments

- This work was developed for thermogravimetric analysis prediction in materials science research.
