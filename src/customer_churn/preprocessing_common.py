"""Shared utilities for preprocessing pipelines."""

from typing import List, Tuple

import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from customer_churn.utils import Config


def extract_features_and_target(cleaned_data: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """Split cleaned data into features and target using project config."""
    target_names: List[str] = [target.new_name for target in config.target]
    target_name = config.target[0].new_name

    X = cleaned_data.drop(columns=target_names)
    y = cleaned_data[target_name]
    return X, y


def create_preprocessor(features_robust: List[str]) -> ColumnTransformer:
    """Create a column transformer with robust scaling for selected columns."""
    return ColumnTransformer(
        transformers=[("robust_scaler", RobustScaler(), features_robust)],
        remainder="passthrough",
    )


def log_processed_shapes(X: pd.DataFrame, y: pd.Series) -> None:
    """Log common preprocessing output details."""
    logger.info(f"Feature columns in X: {X.columns.tolist()}")
    logger.info(f"Data preprocessing completed. Shape of X: {X.shape}, Shape of y: {y.shape}")
