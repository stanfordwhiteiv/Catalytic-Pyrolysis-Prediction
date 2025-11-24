"""
Example: Inference with the Physics-Informed TGA Transformer

This script demonstrates how to load a trained model and make predictions.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import torch

from models import TransformerWithStaticNet
from evaluate import (
    predict_autoregressive,
    evaluate_on_curves,
    load_model_from_checkpoint,
    plot_predictions,
    plot_scatter
)
from data_utils import load_tga_data, get_train_test_split, TGAPreprocessor


def example_load_and_predict():
    """
    Load a trained model and make predictions on test data.
    """
    print("=" * 60)
    print("Example: Load Model and Predict")
    print("=" * 60)

    # Configuration
    checkpoint_path = "../checkpoints/best_model.pt"
    data_folder = "../data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"\n1. Loading model from {checkpoint_path}")
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    print(f"   Model loaded successfully")
    print(f"   Using device: {device}")

    # Load and preprocess data
    print(f"\n2. Loading test data from {data_folder}")
    data = load_tga_data(data_folder, verbose=False)

    train_df, test_df, _, _ = get_train_test_split(
        data,
        test_size=config.data.test_size,
        exclude_identifier=config.data.exclude_identifier,
        random_state=config.data.random_state
    )

    preprocessor = TGAPreprocessor(target_column=config.data.target_column)
    preprocessor.fit(train_df)
    test_processed = preprocessor.transform(test_df)
    print(f"   Test samples: {len(test_df)}")

    # Evaluate on curves
    print("\n3. Evaluating on TGA curves...")
    results = evaluate_on_curves(
        model,
        test_processed,
        preprocessor,
        sequence_length=config.data.sequence_length,
        device=device,
        verbose=True
    )

    # Visualize
    print("\n4. Creating visualizations...")
    plot_predictions(results, n_curves=4, output_path="predictions.png")
    plot_scatter(results, output_path="scatter.png")

    return results


def example_single_curve_prediction():
    """
    Predict a single TGA curve step by step.
    """
    print("=" * 60)
    print("Example: Single Curve Prediction")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a simple model for demonstration
    model = TransformerWithStaticNet(
        dynamic_input_dim=2,
        static_input_dim=5,
        hidden_dim=64,
        num_layers=2,
        nhead=4
    ).to(device)
    model.eval()

    # Simulated initial sequence (10 timesteps)
    # Format: [temperature, previous_char]
    initial_sequence = np.array([
        [0.0, 100.0],   # Normalized temp, char at 100%
        [0.1, 99.5],
        [0.2, 99.0],
        [0.3, 98.2],
        [0.4, 97.1],
        [0.5, 95.8],
        [0.6, 94.2],
        [0.7, 92.5],
        [0.8, 90.5],
        [0.9, 88.0],
    ], dtype=np.float32)

    # Simulated static features (material properties)
    static_features = np.array([0.5, 0.3, 0.1, 0.8, 0.2], dtype=np.float32)

    # Temperatures to predict
    future_temps = np.linspace(1.0, 2.0, 20)

    print(f"\nInitial sequence shape: {initial_sequence.shape}")
    print(f"Static features shape: {static_features.shape}")
    print(f"Predicting {len(future_temps)} future steps...")

    # Predict
    predictions = predict_autoregressive(
        model,
        initial_sequence,
        static_features,
        future_temps,
        device=device
    )

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"First 5 predictions: {predictions[:5]}")

    # Visualize
    plt.figure(figsize=(10, 6))

    # Plot initial sequence
    init_temps = initial_sequence[:, 0]
    init_chars = initial_sequence[:, 1]
    plt.plot(init_temps, init_chars, 'b-o', label='Initial Sequence', markersize=4)

    # Plot predictions
    plt.plot(future_temps, predictions, 'r--s', label='Predictions', markersize=4)

    plt.xlabel('Temperature (normalized)')
    plt.ylabel('Char Yield (%)')
    plt.title('TGA Curve Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('single_curve_prediction.png', dpi=150)
    print("\nSaved plot to single_curve_prediction.png")
    plt.close()

    return predictions


def example_batch_inference():
    """
    Perform batch inference for multiple samples.
    """
    print("=" * 60)
    print("Example: Batch Inference")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = TransformerWithStaticNet(
        dynamic_input_dim=2,
        static_input_dim=5,
        hidden_dim=64,
        num_layers=2,
        nhead=4
    ).to(device)
    model.eval()

    # Batch of inputs
    batch_size = 16
    seq_len = 10
    dynamic_dim = 2
    static_dim = 5

    # Random inputs for demonstration
    dynamic_x = torch.randn(batch_size, seq_len, dynamic_dim).to(device)
    static_x = torch.randn(batch_size, static_dim).to(device)

    print(f"\nInput shapes:")
    print(f"  Dynamic: {dynamic_x.shape}")
    print(f"  Static: {static_x.shape}")

    # Inference
    with torch.no_grad():
        predictions = model(dynamic_x, static_x)

    print(f"\nOutput shape: {predictions.shape}")
    print(f"Predictions:\n{predictions.cpu().numpy().flatten()[:8]}...")

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference Examples")
    parser.add_argument("--example", type=int, default=2, choices=[1, 2, 3],
                       help="Which example to run (1=load_predict, 2=single_curve, 3=batch)")

    args = parser.parse_args()

    if args.example == 1:
        example_load_and_predict()
    elif args.example == 2:
        example_single_curve_prediction()
    elif args.example == 3:
        example_batch_inference()
