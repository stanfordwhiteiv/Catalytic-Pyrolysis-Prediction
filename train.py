"""
Training Script for Physics-Informed TGA Transformer

This script provides training functionality with optional physics-informed loss
and early stopping.

Usage:
    python train.py --data_folder ./data --epochs 100 --use_physics_loss
"""

import os
import argparse
import time
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import (
    TransformerWithStaticNet,
    TransformerStandardNet,
    compute_physics_loss,
    count_parameters
)
from data_utils import (
    load_tga_data,
    get_train_test_split,
    get_kfold_splits,
    TGAPreprocessor,
    create_sequences,
    create_dataloaders
)
from config import Config, get_default_config


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        delta: Minimum change to qualify as an improvement
        verbose: Whether to print messages
    """

    def __init__(self, patience: int = 10, delta: float = 0.0, verbose: bool = False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class Trainer:
    """
    Trainer class for TGA prediction models.

    Handles training loop, validation, checkpointing, and physics-informed loss.

    Args:
        model: PyTorch model to train
        config: Training configuration
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.device = device or config.training.device
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.L1Loss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.training.lr_reduction_factor,
            patience=config.training.patience_lr_scheduler,
            verbose=False
        )
        self.early_stopping = EarlyStopping(
            patience=config.training.patience_early_stopping
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_char_loss': [],
            'over_penalty': [],
            'mono_penalty': []
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        use_physics_loss: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            use_physics_loss: Whether to include physics penalties

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        running_loss = 0.0
        running_char_loss = 0.0
        running_over_penalty = 0.0
        running_mono_penalty = 0.0
        n_samples = 0

        for X_dynamic, X_static, y_batch in tqdm(train_loader, desc="Training", leave=False):
            X_dynamic = X_dynamic.to(self.device)
            X_static = X_static.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_dynamic, X_static)

            # Base loss
            loss_char = self.criterion(outputs, y_batch)

            if use_physics_loss:
                over_penalty, mono_penalty = compute_physics_loss(outputs, X_dynamic)
                loss = (
                    loss_char +
                    self.config.training.over_penalty_lambda * over_penalty +
                    self.config.training.monotonicity_lambda * mono_penalty
                )
                running_over_penalty += over_penalty.item() * X_dynamic.size(0)
                running_mono_penalty += mono_penalty.item() * X_dynamic.size(0)
            else:
                loss = loss_char

            loss.backward()
            self.optimizer.step()

            batch_size = X_dynamic.size(0)
            running_loss += loss.item() * batch_size
            running_char_loss += loss_char.item() * batch_size
            n_samples += batch_size

        return {
            'loss': running_loss / n_samples,
            'char_loss': running_char_loss / n_samples,
            'over_penalty': running_over_penalty / n_samples if use_physics_loss else 0.0,
            'mono_penalty': running_mono_penalty / n_samples if use_physics_loss else 0.0
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Validation loss
        """
        self.model.eval()
        running_loss = 0.0
        n_samples = 0

        for X_dynamic, X_static, y_batch in val_loader:
            X_dynamic = X_dynamic.to(self.device)
            X_static = X_static.to(self.device)
            y_batch = y_batch.to(self.device)

            outputs = self.model(X_dynamic, X_static)
            loss = self.criterion(outputs, y_batch)

            running_loss += loss.item() * X_dynamic.size(0)
            n_samples += X_dynamic.size(0)

        return running_loss / n_samples

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        use_physics_loss: Optional[bool] = None
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (uses config if None)
            use_physics_loss: Whether to use physics loss (uses config if None)

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.training.epochs
        use_physics_loss = (
            use_physics_loss if use_physics_loss is not None
            else self.config.training.use_physics_loss
        )

        print(f"\nTraining for {epochs} epochs")
        print(f"Physics-informed loss: {use_physics_loss}")
        print(f"Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print("-" * 50)

        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, use_physics_loss)

            # Validate
            val_loss = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_loss)
            self.history['train_char_loss'].append(train_metrics['char_loss'])
            self.history['over_penalty'].append(train_metrics['over_penalty'])
            self.history['mono_penalty'].append(train_metrics['mono_penalty'])

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.config.training.save_checkpoints:
                    self.save_checkpoint('best_model.pt')

            # Logging
            if epoch % self.config.training.log_interval == 0:
                if use_physics_loss:
                    print(
                        f"Epoch {epoch:3d}/{epochs} | "
                        f"Train: {train_metrics['loss']:.4f} "
                        f"(Char: {train_metrics['char_loss']:.4f}, "
                        f"Over: {train_metrics['over_penalty']:.4f}, "
                        f"Mono: {train_metrics['mono_penalty']:.4f}) | "
                        f"Val: {val_loss:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:3d}/{epochs} | "
                        f"Train: {train_metrics['loss']:.4f} | "
                        f"Val: {val_loss:.4f}"
                    )

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f}s")
        print(f"Best validation loss: {best_val_loss:.4f}")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.training.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.training.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']


def train_model(
    data_folder: str,
    config: Optional[Config] = None,
    verbose: bool = True
) -> Tuple[nn.Module, Dict, TGAPreprocessor]:
    """
    Train a model from scratch.

    Args:
        data_folder: Path to data folder
        config: Training configuration
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_model, history, preprocessor)
    """
    config = config or get_default_config()
    config.data.data_folder = data_folder

    # Load data
    if verbose:
        print("Loading data...")
    data = load_tga_data(data_folder, verbose=verbose)

    # Split data
    train_df, test_df, _, _ = get_train_test_split(
        data,
        test_size=config.data.test_size,
        exclude_identifier=config.data.exclude_identifier,
        random_state=config.data.random_state,
        target_column=config.data.target_column
    )

    if verbose:
        print(f"\nTrain samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Preprocess
    preprocessor = TGAPreprocessor(
        target_column=config.data.target_column,
        correlation_threshold=config.data.correlation_threshold
    )
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    if verbose:
        print(f"Dynamic features: {preprocessor.dynamic_cols}")
        print(f"Static features: {len(preprocessor.static_cols)} features")

    # Create sequences
    train_dynamic, train_static, train_targets = create_sequences(
        train_processed,
        preprocessor.dynamic_cols,
        preprocessor.static_cols,
        config.data.target_column,
        config.data.sequence_length,
        config.data.prediction_length
    )
    test_dynamic, test_static, test_targets = create_sequences(
        test_processed,
        preprocessor.dynamic_cols,
        preprocessor.static_cols,
        config.data.target_column,
        config.data.sequence_length,
        config.data.prediction_length
    )

    if verbose:
        print(f"\nTraining sequences: {len(train_dynamic)}")
        print(f"Test sequences: {len(test_dynamic)}")

    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        train_dynamic, train_static, train_targets,
        test_dynamic, test_static, test_targets,
        batch_size=config.training.batch_size
    )

    # Get input dimensions
    dynamic_input_dim = train_dynamic.shape[2]
    static_input_dim = train_static.shape[1]

    # Create model
    if config.model.use_cross_attention:
        model = TransformerWithStaticNet(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            nhead=config.model.nhead,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            output_dim=config.data.prediction_length
        )
    else:
        model = TransformerStandardNet(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            nhead=config.model.nhead,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            output_dim=config.data.prediction_length
        )

    # Train
    trainer = Trainer(model, config)
    history = trainer.train(
        train_loader,
        test_loader,
        use_physics_loss=config.training.use_physics_loss
    )

    return model, history, preprocessor


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train Physics-Informed TGA Transformer")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to data folder")
    parser.add_argument("--output_folder", type=str, default="./outputs", help="Output folder")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--use_physics_loss", action="store_true", help="Use physics-informed loss")
    parser.add_argument("--no_cross_attention", action="store_true", help="Disable cross-attention")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Create config
    config = get_default_config()
    config.data.data_folder = args.data_folder
    config.data.output_folder = args.output_folder
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.use_physics_loss = args.use_physics_loss
    config.model.hidden_dim = args.hidden_dim
    config.model.num_layers = args.num_layers
    config.model.use_cross_attention = not args.no_cross_attention

    if args.device:
        config.training.device = args.device

    # Train
    model, history, preprocessor = train_model(args.data_folder, config)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
