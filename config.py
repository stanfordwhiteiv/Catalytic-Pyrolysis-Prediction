"""
Configuration and Hyperparameters

All the magic numbers that took forever to tune, now in one convenient place.
Modify these values to adapt the model to your specific dataset.

These defaults are the result of extensive hyperparameter search (read: many
sleepless nights and hundreds of failed experiments). Change at your own risk.

Author: [Your Name]
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DataConfig:
    """
    Data-related configuration.

    These settings control how data is loaded, split, and preprocessed.
    The defaults work well for our TGA dataset but you'll need to adjust
    column names for your own data.
    """

    # Data paths - update these for your setup
    data_folder: str = "./data"
    output_folder: str = "./outputs"

    # Column names - MUST match your CSV headers exactly
    target_column: str = "Char"                                        # What we're predicting
    id_columns: List[str] = field(default_factory=lambda: ["ID", "ID "])  # Columns to drop (note the space in "ID " - legacy issue)
    categorical_columns: List[str] = field(default_factory=lambda: ["ct", "ct2"])  # Columns to ordinal encode

    # Train/test splitting - done at FILE level, not row level
    # This prevents data leakage between measurements from the same sample
    test_size: float = 0.2
    n_folds: int = 5               # For cross-validation
    random_state: int = 42         # For reproducibility. Always 42. It's the answer to everything.
    exclude_identifier: Optional[str] = "Co"  # Files containing "Co" are held out entirely for final validation

    # Sequence parameters for the sliding window approach
    sequence_length: int = 10      # How many previous timesteps the model sees
    prediction_length: int = 1     # How many steps ahead to predict (just 1 for now)

    # Feature selection - removes highly correlated features
    # 0.9 is aggressive but helps prevent overfitting
    correlation_threshold: float = 0.9


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    These hyperparameters define the transformer architecture. The defaults
    are surprisingly small - turns out you don't need GPT-3 for TGA prediction.
    Tried larger models, they just overfitted.
    """

    # Transformer dimensions
    hidden_dim: int = 64           # Main model dimension. 64 is plenty for this task.
    num_layers: int = 2            # Number of transformer layers. 2 is enough, 3 didn't help.
    nhead: int = 4                 # Attention heads. Must divide hidden_dim evenly!
    dim_feedforward: int = 16      # FFN hidden size. Yes, 16 is tiny. Larger didn't help.
    dropout: float = 0.1           # Standard dropout. 0.2 hurt performance.
    max_seq_len: int = 5000        # For positional encoding. Way more than needed but costs nothing.

    # Architecture choice
    use_cross_attention: bool = True   # True = our model, False = concatenation baseline


@dataclass
class TrainingConfig:
    """
    Training configuration.

    Learning rate, batch size, early stopping, etc. These were tuned on our
    dataset - you might need different values for yours.

    The physics loss weights (lambdas) are particularly sensitive. If you're
    getting weird predictions, try adjusting these first.
    """

    # Core training params
    epochs: int = 150              # Max epochs. Usually stops early around 60-80.
    batch_size: int = 2048         # Large batch = stable gradients. GPU memory permitting.
    learning_rate: float = 0.0005  # AdamW LR. 0.001 was too aggressive, 0.0001 too slow.
    weight_decay: float = 0.0      # L2 regularization. Tried 1e-2, didn't help.

    # Early stopping - prevents overfitting and saves time
    patience_early_stopping: int = 40   # Epochs without improvement before stopping
    patience_lr_scheduler: int = 4      # Epochs before reducing LR
    lr_reduction_factor: float = 0.8    # Multiply LR by this when plateau detected

    # Physics-informed loss - THE SECRET SAUCE
    use_physics_loss: bool = True
    monotonicity_lambda: float = 10.0      # Penalty for non-decreasing char. Tune carefully.
    over_penalty_lambda: float = 1000.0    # Penalty for char > 100%. Be harsh here.

    # Device selection - auto-detects GPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging and checkpoints
    log_interval: int = 10         # Print progress every N epochs
    save_checkpoints: bool = True  # Save best model during training
    checkpoint_dir: str = "./checkpoints"


@dataclass
class Config:
    """
    Master configuration container.

    Holds all sub-configs. Use get_default_config() to get started,
    then modify specific fields as needed.

    Example:
        >>> config = get_default_config()
        >>> config.training.epochs = 50
        >>> config.model.hidden_dim = 128
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Create output directories if they don't exist."""
        import os
        os.makedirs(self.data.output_folder, exist_ok=True)
        if self.training.save_checkpoints:
            os.makedirs(self.training.checkpoint_dir, exist_ok=True)


def get_default_config() -> Config:
    """
    Get the default configuration.

    These are the settings that worked best in our experiments. Start here
    and adjust as needed.

    Returns:
        Default Config object
    """
    return Config()


def get_fast_dev_config() -> Config:
    """
    Get configuration for fast development/testing.

    Reduced epochs and folds for quick iteration when debugging.
    DO NOT use these settings for final results - they're just for
    making sure the code runs without waiting an hour.

    Returns:
        Config with minimal training for fast testing
    """
    config = Config()
    config.training.epochs = 5        # Just enough to see if it's learning
    config.training.batch_size = 256  # Smaller for faster iteration
    config.data.n_folds = 2           # Minimum for CV
    return config


def get_ablation_config(
    use_cross_attention: bool = True,
    use_physics_loss: bool = True
) -> Config:
    """
    Get configuration for ablation study.

    Ablation studies are critical for the paper - reviewers WILL ask
    why we need each component. This function makes it easy to systematically
    enable/disable the key innovations.

    Four configurations to run:
    1. (True, True)   - Full model (cross-attention + physics)
    2. (True, False)  - Cross-attention only
    3. (False, True)  - Physics loss only (concatenation architecture)
    4. (False, False) - Baseline (neither)

    Args:
        use_cross_attention: Whether to use cross-attention architecture
        use_physics_loss: Whether to use physics-informed loss

    Returns:
        Configuration for the specified ablation
    """
    config = Config()
    config.model.use_cross_attention = use_cross_attention
    config.training.use_physics_loss = use_physics_loss
    return config
