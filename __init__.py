"""
Physics-Informed Transformer for TGA Prediction

A PyTorch implementation of a transformer model with cross-attention
for thermogravimetric analysis (TGA) prediction.
"""

from .models import (
    TransformerWithStaticNet,
    TransformerStandardNet,
    PositionalEncoding,
    TransformerEncoderLayerWithCrossAttention,
    TransformerEncoderWithCrossAttention,
    compute_physics_loss,
    count_parameters
)

from .data_utils import (
    TGADataset,
    TGAPreprocessor,
    load_tga_data,
    get_train_test_split,
    get_kfold_splits,
    create_sequences,
    create_dataloaders
)

from .config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    get_default_config,
    get_fast_dev_config,
    get_ablation_config
)

from .train import Trainer, EarlyStopping, train_model
from .evaluate import (
    predict_autoregressive,
    evaluate_on_curves,
    load_model_from_checkpoint
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Models
    'TransformerWithStaticNet',
    'TransformerStandardNet',
    'PositionalEncoding',
    'TransformerEncoderLayerWithCrossAttention',
    'TransformerEncoderWithCrossAttention',
    'compute_physics_loss',
    'count_parameters',

    # Data utilities
    'TGADataset',
    'TGAPreprocessor',
    'load_tga_data',
    'get_train_test_split',
    'get_kfold_splits',
    'create_sequences',
    'create_dataloaders',

    # Config
    'Config',
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'get_default_config',
    'get_fast_dev_config',
    'get_ablation_config',

    # Training
    'Trainer',
    'EarlyStopping',
    'train_model',

    # Evaluation
    'predict_autoregressive',
    'evaluate_on_curves',
    'load_model_from_checkpoint',
]
