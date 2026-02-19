"""Tests for data preprocessing module"""

import pandas as pd
import pytest

from customer_churn.data_preprocessing import DataPreprocessor
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
            "robust": ["tenure", "monthly_charges", "total_charges"],
        },
    )


@pytest.fixture
def sample_data(tmp_path):
    """Create sample data file for testing"""
    data = {
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "tenure": [12, 24, 6, 36, 48, 15, 30, 9, 42, 18],
        "monthly_charges": [50.0, 75.0, 45.0, 90.0, 65.0, 55.0, 80.0, 40.0, 85.0, 60.0],
        "total_charges": [600.0, 1800.0, 270.0, 3240.0, 3120.0, 825.0, 2400.0, 360.0, 3570.0, 1080.0],
        "churn": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_preprocessor_initialization(sample_data, sample_config):
    """Test DataPreprocessor initialization"""
    preprocessor = DataPreprocessor(sample_data, sample_config)
    assert preprocessor.X is not None
    assert preprocessor.y is not None
    assert len(preprocessor.X) == 10


def test_get_processed_data(sample_data, sample_config):
    """Test getting processed data"""
    preprocessor = DataPreprocessor(sample_data, sample_config)
    X, y, prep = preprocessor.get_processed_data()
    
    assert X is not None
    assert y is not None
    assert prep is not None
    assert len(X) == len(y)


def test_preprocessor_shapes(sample_data, sample_config):
    """Test that preprocessor returns correct shapes"""
    preprocessor = DataPreprocessor(sample_data, sample_config)
    X, y, _ = preprocessor.get_processed_data()
    
    assert X.shape[0] == 10  # Number of rows
    assert len(y) == 10  # Target length
    assert "churn_label" not in X.columns  # Target should not be in features


def test_feature_columns(sample_data, sample_config):
    """Test that correct feature columns are present"""
    preprocessor = DataPreprocessor(sample_data, sample_config)
    X, _, _ = preprocessor.get_processed_data()
    
    expected_features = ["customer_id", "tenure", "monthly_charges", "total_charges"]
    for feature in expected_features:
        assert feature in X.columns
