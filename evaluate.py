"""
Evaluation Script for Physics-Informed TGA Transformer

Provides evaluation metrics, autoregressive prediction, and visualization
for trained models.

Usage:
    python evaluate.py --checkpoint best_model.pt --data_folder ./data
"""

import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn

from models import TransformerWithStaticNet, TransformerStandardNet
from data_utils import (
    load_tga_data,
    get_train_test_split,
    TGAPreprocessor,
    create_sequences
)
from config import Config, get_default_config


def predict_autoregressive(
    model: nn.Module,
    initial_sequence: np.ndarray,
    static_features: np.ndarray,
    temperatures: np.ndarray,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Predict a full TGA curve autoregressively.

    Starting from an initial sequence, predicts each subsequent point
    and feeds the prediction back as input for the next step.

    Args:
        model: Trained model
        initial_sequence: Initial dynamic sequence (seq_len, dynamic_dim)
        static_features: Static features (static_dim,)
        temperatures: Array of temperatures to predict at
        device: Device for computation

    Returns:
        Array of predicted char values
    """
    model.eval()
    predictions = []
    current_sequence = initial_sequence.copy()

    with torch.no_grad():
        for temp in temperatures:
            # Prepare tensors
            X_dynamic = torch.tensor(
                current_sequence[np.newaxis, :, :],
                dtype=torch.float32
            ).to(device)
            X_static = torch.tensor(
                static_features[np.newaxis, :],
                dtype=torch.float32
            ).to(device)

            # Predict
            pred = model(X_dynamic, X_static).cpu().numpy()[0, 0]
            predictions.append(pred)

            # Update sequence for next step
            # Dynamic features: [Temp, prev_char]
            new_step = np.array([temp, pred], dtype=np.float32)
            current_sequence = np.vstack([current_sequence[1:], new_step])

    return np.array(predictions)


def evaluate_on_curves(
    model: nn.Module,
    test_df: pd.DataFrame,
    preprocessor: TGAPreprocessor,
    sequence_length: int = 10,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict:
    """
    Evaluate model on complete TGA curves.

    Performs autoregressive prediction on each test curve and computes
    metrics.

    Args:
        model: Trained model
        test_df: Test DataFrame (preprocessed)
        preprocessor: Fitted preprocessor
        sequence_length: Length of input sequence
        device: Device for computation
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation metrics and predictions
    """
    curve_results = []
    all_predictions = []
    all_actuals = []

    dynamic_cols = preprocessor.dynamic_cols
    static_cols = preprocessor.static_cols
    target_col = preprocessor.target_column
    temp_col = dynamic_cols[0]

    grouped = test_df.groupby('filename')
    n_curves = len(grouped)

    for idx, (filename, group) in enumerate(grouped):
        group = group.sort_values(temp_col).reset_index(drop=True)

        if len(group) < sequence_length + 10:
            continue

        # Prepare data
        group = group.copy()
        group['prev_char'] = group[target_col].shift(1)
        group.loc[group.index[0], 'prev_char'] = group[target_col].iloc[0]

        dyn_data = group[dynamic_cols + ['prev_char']].copy()
        dyn_data = dyn_data.apply(pd.to_numeric, errors='coerce')
        target_data = group[target_col].astype(float).reset_index(drop=True)

        mask = dyn_data.notna().all(axis=1)
        dyn_data = dyn_data[mask]
        target_data = target_data[mask]

        if len(dyn_data) < sequence_length + 10:
            continue

        static_values = group[static_cols].drop_duplicates().iloc[0].values.astype(np.float32)
        initial_sequence = dyn_data.iloc[:sequence_length].values.astype(np.float32)
        temperatures = dyn_data.iloc[sequence_length:, 0].values
        actual_values = target_data.iloc[sequence_length:].values

        try:
            predicted_values = predict_autoregressive(
                model, initial_sequence, static_values, temperatures, device
            )

            curve_mae = mean_absolute_error(actual_values, predicted_values)
            curve_rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            curve_r2 = r2_score(actual_values, predicted_values)

            curve_results.append({
                'filename': filename,
                'mae': curve_mae,
                'rmse': curve_rmse,
                'r2': curve_r2,
                'predictions': predicted_values,
                'actuals': actual_values,
                'temperatures': temperatures
            })

            all_predictions.extend(predicted_values)
            all_actuals.extend(actual_values)

            if verbose and (idx + 1) % 10 == 0:
                print(f"Evaluated {idx + 1}/{n_curves} curves")

        except Exception as e:
            if verbose:
                print(f"Error evaluating {filename}: {e}")
            continue

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    results = {
        'MAE': mean_absolute_error(all_actuals, all_predictions),
        'RMSE': np.sqrt(mean_squared_error(all_actuals, all_predictions)),
        'R2': r2_score(all_actuals, all_predictions),
        'Avg_Curve_MAE': np.mean([c['mae'] for c in curve_results]),
        'Avg_Curve_RMSE': np.mean([c['rmse'] for c in curve_results]),
        'predictions': all_predictions,
        'actuals': all_actuals,
        'curves': curve_results,
        'n_curves': len(curve_results)
    }

    if verbose:
        print(f"\nEvaluation Results:")
        print(f"  Curves evaluated: {results['n_curves']}")
        print(f"  Overall MAE: {results['MAE']:.4f}")
        print(f"  Overall RMSE: {results['RMSE']:.4f}")
        print(f"  Overall R²: {results['R2']:.4f}")
        print(f"  Avg Curve MAE: {results['Avg_Curve_MAE']:.4f}")

    return results


def plot_predictions(
    results: Dict,
    n_curves: int = 4,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot predicted vs actual curves.

    Args:
        results: Evaluation results dictionary
        n_curves: Number of curves to plot
        output_path: Path to save figure (optional)
        figsize: Figure size
    """
    curves = results['curves'][:n_curves]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, curve in enumerate(curves):
        ax = axes[idx]
        ax.plot(curve['temperatures'], curve['actuals'],
                'b-', label='Actual', linewidth=2)
        ax.plot(curve['temperatures'], curve['predictions'],
                'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Char Yield (%)')
        ax.set_title(f"{curve['filename']}\nR² = {curve['r2']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_scatter(
    results: Dict,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
):
    """
    Create scatter plot of predictions vs actuals.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save figure (optional)
        figsize: Figure size
    """
    predictions = results['predictions']
    actuals = results['actuals']

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(actuals, predictions, alpha=0.3, s=10)

    # Diagonal line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('Actual Char Yield (%)', fontsize=12)
    ax.set_ylabel('Predicted Char Yield (%)', fontsize=12)
    ax.set_title(f"Predictions vs Actuals\nR² = {results['R2']:.3f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to {output_path}")
    else:
        plt.show()

    plt.close()


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu'
) -> Tuple[nn.Module, Config]:
    """
    Load a model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Infer input dimensions from saved state
    state_dict = checkpoint['model_state_dict']
    dynamic_input_dim = state_dict['dynamic_input_fc.weight'].shape[1]
    static_input_dim = state_dict['static_input_fc.weight'].shape[1]

    if config.model.use_cross_attention:
        model = TransformerWithStaticNet(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            nhead=config.model.nhead,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout
        )
    else:
        model = TransformerStandardNet(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            nhead=config.model.nhead,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


def save_results_csv(results: Dict, output_path: str):
    """
    Save evaluation results to CSV.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save CSV
    """
    curve_data = []
    for curve in results['curves']:
        curve_data.append({
            'filename': curve['filename'],
            'MAE': curve['mae'],
            'RMSE': curve['rmse'],
            'R2': curve['r2']
        })

    df = pd.DataFrame(curve_data)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Physics-Informed TGA Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to data folder")
    parser.add_argument("--output_folder", type=str, default="./evaluation", help="Output folder")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--plot_curves", type=int, default=4, help="Number of curves to plot")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Determine device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    # Load and preprocess data
    print(f"\nLoading data from {args.data_folder}")
    data = load_tga_data(args.data_folder, verbose=False)

    _, test_df, _, _ = get_train_test_split(
        data,
        test_size=config.data.test_size,
        exclude_identifier=config.data.exclude_identifier,
        random_state=config.data.random_state
    )

    # Note: In practice, you'd save and load the preprocessor
    # Here we refit on test data which isn't ideal but works for demonstration
    preprocessor = TGAPreprocessor(target_column=config.data.target_column)
    # This would normally be loaded from training
    train_df, _, _, _ = get_train_test_split(
        data,
        test_size=config.data.test_size,
        exclude_identifier=config.data.exclude_identifier,
        random_state=config.data.random_state
    )
    preprocessor.fit(train_df)
    test_processed = preprocessor.transform(test_df)

    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_on_curves(
        model, test_processed, preprocessor,
        sequence_length=config.data.sequence_length,
        device=device
    )

    # Save results
    save_results_csv(
        results,
        os.path.join(args.output_folder, 'curve_results.csv')
    )

    # Create visualizations
    print("\nCreating visualizations...")
    plot_predictions(
        results,
        n_curves=args.plot_curves,
        output_path=os.path.join(args.output_folder, 'predictions.png')
    )
    plot_scatter(
        results,
        output_path=os.path.join(args.output_folder, 'scatter.png')
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
