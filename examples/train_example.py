"""
Example: Training the Physics-Informed TGA Transformer

This script demonstrates how to train the model on your TGA data.
"""

import sys
sys.path.append('..')

from train import train_model, Trainer
from config import Config, get_default_config
from models import TransformerWithStaticNet, count_parameters
from data_utils import (
    load_tga_data,
    get_train_test_split,
    TGAPreprocessor,
    create_sequences,
    create_dataloaders
)


def example_basic_training():
    """
    Basic training example using the train_model function.
    """
    print("=" * 60)
    print("Example 1: Basic Training")
    print("=" * 60)

    # Get default configuration
    config = get_default_config()

    # Customize for your needs
    config.data.data_folder = "../data"  # Update this path
    config.training.epochs = 50
    config.training.use_physics_loss = True
    config.model.use_cross_attention = True

    # Train the model
    model, history, preprocessor = train_model(
        data_folder=config.data.data_folder,
        config=config,
        verbose=True
    )

    print(f"\nTraining complete!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

    return model, history


def example_custom_training():
    """
    Custom training example with more control over the process.
    """
    print("=" * 60)
    print("Example 2: Custom Training Pipeline")
    print("=" * 60)

    # Configuration
    config = get_default_config()
    config.data.data_folder = "../data"  # Update this path
    config.training.epochs = 100
    config.training.use_physics_loss = True

    # Step 1: Load data
    print("\n1. Loading data...")
    data = load_tga_data(config.data.data_folder)

    # Step 2: Split data
    print("2. Splitting data...")
    train_df, test_df, train_files, test_files = get_train_test_split(
        data,
        test_size=config.data.test_size,
        exclude_identifier=config.data.exclude_identifier,
        random_state=config.data.random_state
    )
    print(f"   Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    # Step 3: Preprocess
    print("3. Preprocessing...")
    preprocessor = TGAPreprocessor(
        target_column=config.data.target_column,
        correlation_threshold=config.data.correlation_threshold
    )
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    print(f"   Dynamic features: {preprocessor.dynamic_cols}")
    print(f"   Static features: {len(preprocessor.static_cols)} features")

    # Step 4: Create sequences
    print("4. Creating sequences...")
    train_dynamic, train_static, train_targets = create_sequences(
        train_processed,
        preprocessor.dynamic_cols,
        preprocessor.static_cols,
        config.data.target_column,
        config.data.sequence_length
    )
    test_dynamic, test_static, test_targets = create_sequences(
        test_processed,
        preprocessor.dynamic_cols,
        preprocessor.static_cols,
        config.data.target_column,
        config.data.sequence_length
    )
    print(f"   Train sequences: {len(train_dynamic)}")
    print(f"   Test sequences: {len(test_dynamic)}")

    # Step 5: Create data loaders
    print("5. Creating data loaders...")
    train_loader, test_loader = create_dataloaders(
        train_dynamic, train_static, train_targets,
        test_dynamic, test_static, test_targets,
        batch_size=config.training.batch_size
    )

    # Step 6: Create model
    print("6. Creating model...")
    dynamic_input_dim = train_dynamic.shape[2]
    static_input_dim = train_static.shape[1]

    model = TransformerWithStaticNet(
        dynamic_input_dim=dynamic_input_dim,
        static_input_dim=static_input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        nhead=config.model.nhead,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout
    )
    print(f"   Parameters: {count_parameters(model):,}")

    # Step 7: Train
    print("7. Training...")
    trainer = Trainer(model, config)
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=config.training.epochs,
        use_physics_loss=config.training.use_physics_loss
    )

    # Step 8: Save checkpoint
    print("\n8. Saving model...")
    trainer.save_checkpoint("trained_model.pt")
    print("   Saved to checkpoints/trained_model.pt")

    return model, history, preprocessor


def example_ablation():
    """
    Example of running ablation experiments.
    """
    print("=" * 60)
    print("Example 3: Ablation Study")
    print("=" * 60)

    from config import get_ablation_config

    configurations = [
        ("Cross-Attn + Physics", True, True),
        ("Cross-Attn Only", True, False),
        ("Concat + Physics", False, True),
        ("Concat Only", False, False),
    ]

    results = {}

    for name, use_cross_attn, use_physics in configurations:
        print(f"\nTraining: {name}")
        print("-" * 40)

        config = get_ablation_config(
            use_cross_attention=use_cross_attn,
            use_physics_loss=use_physics
        )
        config.data.data_folder = "../data"  # Update this path
        config.training.epochs = 50  # Reduced for example

        model, history, _ = train_model(
            data_folder=config.data.data_folder,
            config=config,
            verbose=False
        )

        results[name] = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }

    # Print results
    print("\n" + "=" * 60)
    print("Ablation Results")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"{name:25s} | Train: {metrics['final_train_loss']:.4f} | Val: {metrics['final_val_loss']:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training Examples")
    parser.add_argument("--example", type=int, default=1, choices=[1, 2, 3],
                       help="Which example to run (1=basic, 2=custom, 3=ablation)")

    args = parser.parse_args()

    if args.example == 1:
        example_basic_training()
    elif args.example == 2:
        example_custom_training()
    elif args.example == 3:
        example_ablation()
