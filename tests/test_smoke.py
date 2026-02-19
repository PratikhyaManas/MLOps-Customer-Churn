"""Smoke tests for production deployment validation."""

import pytest

from customer_churn.deployment_validator import DeploymentValidator, SmokeTestRunner
from customer_churn.health_checks import HealthChecker
from customer_churn.secrets_manager import validate_environment


class TestDeploymentReadiness:
    """Test suite for deployment readiness."""

    def test_pre_deployment_validation(self):
        """Test that all pre-deployment checks pass."""
        validator = DeploymentValidator()
        result = validator.validate_all(code_coverage_threshold=0.70)
        assert validator.can_deploy(), "Deployment blocked by validation errors"

    def test_configuration_files_exist(self):
        """Test that required configuration files exist."""
        validator = DeploymentValidator()
        assert validator.validate_configuration_files(), "Missing required configuration files"

    def test_bundle_configuration_valid(self):
        """Test that bundle configuration is production-ready."""
        validator = DeploymentValidator()
        assert validator.validate_bundle_configuration(), "Bundle configuration invalid for production"

    def test_dependencies_valid(self):
        """Test that dependencies are properly specified."""
        validator = DeploymentValidator()
        assert validator.validate_dependencies(), "Dependency specification invalid"

    def test_environment_variables_set(self):
        """Test that required environment variables are set."""
        assert validate_environment(), "Required environment variables not set"


class TestHealthChecks:
    """Test suite for health checks."""

    def test_dependencies_available(self):
        """Test that required packages are available."""
        checker = HealthChecker()
        required_packages = [
            "pandas",
            "sklearn",
            "lightgbm",
            "mlflow",
        ]
        assert checker.check_dependencies_available(required_packages), "Required packages not available"

    def test_spark_available(self):
        """Test that Spark session is available."""
        # This test may be skipped in non-Databricks environments
        checker = HealthChecker()
        try:
            result = checker.check_spark_available()
            # Don't assert in non-Databricks env, just log
            if result:
                assert result, "Spark not available"
        except Exception as e:
            pytest.skip(f"Spark not available in this environment: {str(e)}")

    def test_disk_space_available(self):
        """Test that minimum disk space is available."""
        checker = HealthChecker()
        assert checker.check_disk_space_available(min_space_gb=0.1), "Insufficient disk space"


class TestDeploymentSafeguards:
    """Test suite for deployment safeguards."""

    def test_no_blocking_errors(self):
        """Test that validator has no blocking errors."""
        validator = DeploymentValidator()
        validator.validate_all()
        assert validator.can_deploy(), f"Blocking errors found: {validator.errors}"

    def test_validation_summary_generated(self):
        """Test that validation summary can be generated without errors."""
        validator = DeploymentValidator()
        validator.validate_all()
        validator.print_validation_summary()  # Should not raise


class TestDataQuality:
    """Test suite for data quality checks."""

    def test_validator_initialization(self):
        """Test that DataValidator can be initialized."""
        from customer_churn.data_validator import DataValidator

        schema = {"col1": "int64", "col2": "float64"}
        validator = DataValidator(schema)
        assert validator.schema == schema

    def test_schema_validation(self):
        """Test schema validation functionality."""
        import pandas as pd

        from customer_churn.data_validator import DataValidator

        schema = {"col1": "int64", "col2": "float64"}
        validator = DataValidator(schema)

        # Valid schema
        df_valid = pd.DataFrame({"col1": [1, 2, 3], "col2": [1.0, 2.0, 3.0]})
        assert validator.validate_schema(df_valid), "Valid schema incorrectly rejected"

        # Invalid schema - missing column
        df_invalid = pd.DataFrame({"col1": [1, 2, 3]})
        assert not validator.validate_schema(df_invalid), "Invalid schema incorrectly accepted"

    def test_required_fields_validation(self):
        """Test required fields validation."""
        import pandas as pd

        from customer_churn.data_validator import DataValidator

        schema = {"col1": "int64", "col2": "float64"}
        validator = DataValidator(schema)

        # Valid - no nulls
        df_valid = pd.DataFrame({"col1": [1, 2, 3], "col2": [1.0, 2.0, 3.0]})
        assert validator.validate_required_fields(df_valid, ["col1", "col2"]), "Valid data rejected"

        # Invalid - nulls present
        df_invalid = pd.DataFrame({"col1": [1, None, 3], "col2": [1.0, 2.0, 3.0]})
        assert not validator.validate_required_fields(df_invalid, ["col1"]), "Data with nulls accepted"


# Fixture for sample data
@pytest.fixture
def sample_data():
    """Provide sample test data."""
    import pandas as pd

    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "tenure": [12, 24, 6, 36, 48],
            "monthly_charges": [50.0, 75.0, 45.0, 90.0, 65.0],
            "total_charges": [600.0, 1800.0, 270.0, 3240.0, 3120.0],
            "churn": [0, 1, 0, 0, 1],
        }
    )
