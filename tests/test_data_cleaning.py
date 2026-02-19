"""Tests for data cleaning module"""

import pandas as pd
import pytest

from customer_churn.data_cleaning import DataCleaning
from customer_churn.utils import Config


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing"""
    return Config(
        catalog_name="churn",
        schema_name="default",
        pipeline_id="test_pipeline",
        parameters={"learning_rate": 0.05, "random_state": 42, "force_col_wise": True},
        ab_test={"learning_rate_a": 0.05, "learning_rate_b": 0.1, "force_col_wise": True},
        num_features=[
            {"name": "customer_id", "dtype": "int64"},
            {"name": "tenure", "dtype": "int64"},
            {"name": "monthly_charges", "dtype": "float64"},
            {"name": "total_charges", "dtype": "float64"},
        ],
        target=[{"name": "churn", "dtype": "int64", "new_name": "churn_label"}],
        features={
            "clean": ["customer_id", "tenure", "monthly_charges", "total_charges"],
            "robust": ["customer_id", "tenure_robust", "monthly_charges_robust", "total_charges_robust"],
        },
    )


@pytest.fixture
def sample_data(tmp_path):
    """Create sample data file for testing"""
    data = {
        "customer_id": [1, 2, 3, 4, 5],
        "tenure": [12, 24, 6, 36, 48],
        "monthly_charges": [50.0, 75.0, 45.0, 90.0, 65.0],
        "total_charges": [600.0, 1800.0, 270.0, 3240.0, 3120.0],
        "churn": [0, 1, 0, 0, 1],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_data_cleaning_initialization(sample_data, sample_config):
    """Test DataCleaning class initialization"""
    cleaner = DataCleaning(sample_data, sample_config)
    assert cleaner.df is not None
    assert len(cleaner.df) == 5
    assert cleaner.target_config.name == "churn"


def test_validate_file_exists(sample_config):
    """Test file existence validation"""
    with pytest.raises(FileNotFoundError):
        DataCleaning("nonexistent_file.csv", sample_config)


def test_handle_missing_values(sample_data, sample_config):
    """Test missing value handling"""
    cleaner = DataCleaning(sample_data, sample_config)
    result = cleaner.handle_missing_values()
    assert result is not None
    assert result.isnull().sum().sum() == 0


def test_handle_duplicates(sample_data, sample_config):
    """Test duplicate handling"""
    cleaner = DataCleaning(sample_data, sample_config)
    initial_count = len(cleaner.df)
    result = cleaner.handle_duplicates()
    assert len(result) <= initial_count


def test_rename_target(sample_data, sample_config):
    """Test target column renaming"""
    cleaner = DataCleaning(sample_data, sample_config)
    result = cleaner.rename_target()
    assert "churn_label" in result.columns
    assert "churn" not in result.columns


def test_clean_data_pipeline(sample_data, sample_config):
    """Test complete data cleaning pipeline"""
    cleaner = DataCleaning(sample_data, sample_config)
    result = cleaner.clean_data()
    assert result is not None
    assert "churn_label" in result.columns
    assert len(result) > 0
