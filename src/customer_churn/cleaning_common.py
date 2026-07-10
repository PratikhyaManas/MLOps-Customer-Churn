"""Shared utilities for data cleaning implementations."""

from typing import List

import pandas as pd

from customer_churn.utils import Config, Target


def build_target_config(config: Config) -> Target:
    """Build target configuration from the project config model."""
    target_info = config.target[0]
    return Target(name=target_info.name, dtype=target_info.dtype, new_name=target_info.new_name)


def get_required_columns(config: Config, target: Target) -> List[str]:
    """Return required feature and target columns used by cleaners."""
    return [feature.name for feature in config.num_features] + [target.name]


def validate_required_columns(df: pd.DataFrame, config: Config, target: Target) -> None:
    """Validate that configured required columns are present in the dataframe."""
    required_columns = get_required_columns(config, target)
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise Exception(f"Missing required columns: {', '.join(missing_columns)}")
