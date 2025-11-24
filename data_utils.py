"""
Data Loading and Preprocessing Utilities for TGA Prediction

This module provides utilities for loading thermogravimetric analysis (TGA) data,
creating train/test splits, and preparing sequences for the transformer model.
"""

import os
import glob
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class TGADataset(Dataset):
    """
    PyTorch Dataset for TGA sequence data.

    Stores dynamic time-series features, static material properties,
    and target char values.

    Args:
        dynamic_inputs: Dynamic sequences (n_samples, seq_len, dynamic_dim)
        static_inputs: Static features (n_samples, static_dim)
        targets: Target values (n_samples, pred_len)
    """

    def __init__(
        self,
        dynamic_inputs: np.ndarray,
        static_inputs: np.ndarray,
        targets: np.ndarray
    ):
        self.X_dynamic = torch.tensor(dynamic_inputs, dtype=torch.float32)
        self.X_static = torch.tensor(static_inputs, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_dynamic[idx], self.X_static[idx], self.y[idx]


def load_tga_data(folder_path: str, verbose: bool = True) -> List[Tuple[str, pd.DataFrame]]:
    """
    Load all TGA CSV files from a folder.

    Each CSV file is expected to contain TGA measurements with columns for
    temperature, char yield, and material properties.

    Args:
        folder_path: Path to folder containing CSV files
        verbose: Whether to print loading progress

    Returns:
        List of (filename, DataFrame) tuples
    """
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    data = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['filename'] = os.path.basename(file)
            data.append((os.path.basename(file), df))
            if verbose:
                print(f"Loaded: {os.path.basename(file)} ({len(df)} records)")
        except Exception as e:
            if verbose:
                print(f"Error reading {file}: {e}")

    if verbose:
        print(f"\nTotal files loaded: {len(data)}")

    return data


def get_train_test_split(
    data: List[Tuple[str, pd.DataFrame]],
    test_size: float = 0.2,
    exclude_identifier: Optional[str] = None,
    random_state: int = 42,
    id_columns: List[str] = None,
    target_column: str = 'Char',
    categorical_columns: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Create train/test split at the file level.

    Splits data by files (not individual samples) to prevent data leakage
    between related measurements.

    Args:
        data: List of (filename, DataFrame) tuples
        test_size: Fraction of files to use for testing
        exclude_identifier: Substring to exclude files (e.g., "Co" for holdout)
        random_state: Random seed for reproducibility
        id_columns: Columns to drop (ID columns)
        target_column: Name of target column
        categorical_columns: Columns to encode as categorical

    Returns:
        Tuple of (train_df, test_df, train_files, test_files)
    """
    if id_columns is None:
        id_columns = ['ID', 'ID ']
    if categorical_columns is None:
        categorical_columns = ['ct', 'ct2']

    all_filenames = [item[0] for item in data]

    # Optionally exclude certain files
    if exclude_identifier:
        included_files = [
            fname for fname in all_filenames
            if exclude_identifier.lower() not in fname.lower()
        ]
        print(f"Excluded files containing '{exclude_identifier}'")
    else:
        included_files = all_filenames

    # Split files
    train_files, test_files = train_test_split(
        included_files,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

    # Collect DataFrames
    train_dfs = [df for fname, df in data if fname in train_files]
    test_dfs = [df for fname, df in data if fname in test_files]

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Clean data
    train_df = train_df.drop(columns=id_columns, errors='ignore')
    test_df = test_df.drop(columns=id_columns, errors='ignore')

    for col in categorical_columns:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str)

    train_df = train_df.dropna(subset=[target_column]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[target_column]).reset_index(drop=True)

    return train_df, test_df, train_files, test_files


def get_kfold_splits(
    data: List[Tuple[str, pd.DataFrame]],
    n_folds: int = 5,
    exclude_identifier: Optional[str] = None,
    random_state: int = 42,
    id_columns: List[str] = None,
    target_column: str = 'Char',
    categorical_columns: List[str] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]]:
    """
    Generate K-fold cross-validation splits at the file level.

    Args:
        data: List of (filename, DataFrame) tuples
        n_folds: Number of folds
        exclude_identifier: Substring to exclude files
        random_state: Random seed
        id_columns: Columns to drop
        target_column: Name of target column
        categorical_columns: Columns to encode as categorical

    Returns:
        List of (train_df, test_df, train_files, test_files) for each fold
    """
    if id_columns is None:
        id_columns = ['ID', 'ID ']
    if categorical_columns is None:
        categorical_columns = ['ct', 'ct2']

    all_filenames = [item[0] for item in data]

    if exclude_identifier:
        included_files = [
            fname for fname in all_filenames
            if exclude_identifier.lower() not in fname.lower()
        ]
    else:
        included_files = all_filenames

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    included_files_array = np.array(included_files)

    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(included_files_array)):
        train_files = included_files_array[train_idx].tolist()
        test_files = included_files_array[test_idx].tolist()

        train_dfs = [df for fname, df in data if fname in train_files]
        test_dfs = [df for fname, df in data if fname in test_files]

        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        train_df = train_df.drop(columns=id_columns, errors='ignore')
        test_df = test_df.drop(columns=id_columns, errors='ignore')

        for col in categorical_columns:
            if col in train_df.columns:
                train_df[col] = train_df[col].astype(str)
            if col in test_df.columns:
                test_df[col] = test_df[col].astype(str)

        train_df = train_df.dropna(subset=[target_column]).reset_index(drop=True)
        test_df = test_df.dropna(subset=[target_column]).reset_index(drop=True)

        splits.append((train_df, test_df, train_files, test_files))
        print(f"Fold {fold_idx + 1}: {len(train_files)} train, {len(test_files)} test files")

    return splits


class TGAPreprocessor:
    """
    Preprocessor for TGA data.

    Handles feature extraction, scaling, and sequence creation for the
    transformer model.

    Args:
        target_column: Name of target column
        categorical_columns: Columns to encode as categorical
        correlation_threshold: Threshold for removing highly correlated features
    """

    def __init__(
        self,
        target_column: str = 'Char',
        categorical_columns: List[str] = None,
        correlation_threshold: float = 0.9
    ):
        self.target_column = target_column
        self.categorical_columns = categorical_columns or ['ct', 'ct2']
        self.correlation_threshold = correlation_threshold

        self.preprocessor = None
        self.feature_names = None
        self.high_corr_features = None
        self.dynamic_cols = None
        self.static_cols = None

    def fit(self, train_df: pd.DataFrame) -> 'TGAPreprocessor':
        """
        Fit the preprocessor on training data.

        Args:
            train_df: Training DataFrame

        Returns:
            self
        """
        X_train = train_df.drop(columns=[self.target_column])
        filenames_train = X_train['filename']
        X_train = X_train.drop(columns=['filename'])

        # Identify numeric columns
        numeric_columns = list(X_train.select_dtypes(include=np.number).columns)

        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('ct_encoder', OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=-1
                ), self.categorical_columns),
                ('scaler', StandardScaler(), numeric_columns)
            ],
            remainder='passthrough'
        )

        X_train_processed = self.preprocessor.fit_transform(X_train)
        self.feature_names = self.preprocessor.get_feature_names_out()

        X_train_df = pd.DataFrame(X_train_processed, columns=self.feature_names)
        X_train_df['filename'] = filenames_train.values

        # Find and remove highly correlated features
        corr_matrix = X_train_df.drop(columns=['filename']).corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Features to exclude from correlation filtering
        exclude_cols = [
            col for col in self.categorical_columns if col in self.feature_names
        ] + ['remainder__Temp']

        self.high_corr_features = [
            column for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
            and column not in exclude_cols
        ]

        # Find temperature column
        temp_col = None
        for col in X_train_df.columns:
            if 'temp' in col.lower() and col != 'filename':
                temp_col = col
                break

        if temp_col is None:
            raise ValueError("Could not find temperature column")

        # Define dynamic and static columns
        self.dynamic_cols = [temp_col]
        self.static_cols = [
            col for col in X_train_df.columns
            if col not in self.dynamic_cols + ['filename', self.target_column]
            and col not in self.high_corr_features
        ]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a DataFrame using the fitted preprocessor.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame with filename and target columns
        """
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].astype(float)
        filenames = X['filename']
        X = X.drop(columns=['filename'])

        X_processed = self.preprocessor.transform(X)
        X_df = pd.DataFrame(X_processed, columns=self.feature_names)
        X_df['filename'] = filenames.values

        # Remove high correlation features
        X_filtered = X_df.drop(columns=self.high_corr_features, errors='ignore')

        # Add target back
        processed_df = X_filtered.copy()
        processed_df[self.target_column] = y.values

        return processed_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


def create_sequences(
    df: pd.DataFrame,
    dynamic_cols: List[str],
    static_cols: List[str],
    target_col: str = 'Char',
    sequence_length: int = 10,
    pred_len: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for time-series prediction.

    Groups data by filename, creates sliding window sequences of dynamic
    features, and extracts corresponding static features and targets.

    Args:
        df: Preprocessed DataFrame
        dynamic_cols: Columns for dynamic (time-varying) features
        static_cols: Columns for static features
        target_col: Name of target column
        sequence_length: Length of input sequence
        pred_len: Prediction horizon

    Returns:
        Tuple of (dynamic_inputs, static_inputs, targets)
            - dynamic_inputs: (n_samples, seq_len, dynamic_dim + 1)
              The +1 is for prev_char feature
            - static_inputs: (n_samples, static_dim)
            - targets: (n_samples, pred_len)
    """
    inputs_dynamic = []
    inputs_static = []
    targets = []

    grouped = df.groupby('filename')
    temp_col = dynamic_cols[0]

    for _, group in grouped:
        group = group.sort_values(temp_col).reset_index(drop=True)

        if len(group) < sequence_length + pred_len:
            continue

        # Create autoregressive prev_char feature
        group = group.copy()
        group['prev_char'] = group[target_col].shift(1)
        group.loc[group.index[0], 'prev_char'] = group[target_col].iloc[0]

        # Prepare dynamic and target data
        dyn_data = group[dynamic_cols + ['prev_char']].copy()
        dyn_data = dyn_data.apply(pd.to_numeric, errors='coerce')
        target_data = group[target_col].astype(float).reset_index(drop=True)

        # Drop NaN rows
        mask = dyn_data.notna().all(axis=1)
        dyn_data = dyn_data[mask]
        target_data = target_data[mask]

        group_length = len(dyn_data)

        # Static features (same for entire group)
        static_values = group[static_cols].drop_duplicates().iloc[0].values.astype(np.float32)

        # Create sliding window sequences
        for i in range(group_length - sequence_length - pred_len + 1):
            input_seq_dynamic = dyn_data.iloc[i:i+sequence_length].values.astype(np.float32)
            target_seq = target_data.iloc[
                i+sequence_length:i+sequence_length+pred_len
            ].values.astype(np.float32)

            inputs_dynamic.append(input_seq_dynamic)
            inputs_static.append(static_values)
            targets.append(target_seq)

    return (
        np.array(inputs_dynamic, dtype=np.float32),
        np.array(inputs_static, dtype=np.float32),
        np.array(targets, dtype=np.float32)
    )


def create_dataloaders(
    train_dynamic: np.ndarray,
    train_static: np.ndarray,
    train_targets: np.ndarray,
    test_dynamic: np.ndarray,
    test_static: np.ndarray,
    test_targets: np.ndarray,
    batch_size: int = 2048,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
        train_dynamic: Training dynamic sequences
        train_static: Training static features
        train_targets: Training targets
        test_dynamic: Test dynamic sequences
        test_static: Test static features
        test_targets: Test targets
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = TGADataset(train_dynamic, train_static, train_targets)
    test_dataset = TGADataset(test_dynamic, test_static, test_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader
