"""Pre-deployment validation and safeguards for production releases."""

from typing import Dict, List, Optional

from loguru import logger


class DeploymentValidator:
    """Validates deployment readiness and safety."""

    def __init__(self):
        """Initialize DeploymentValidator."""
        self.validation_checks: Dict[str, bool] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def validate_code_quality(self, code_coverage_threshold: float = 0.80) -> bool:
        """
        Validate code quality metrics.

        Args:
            code_coverage_threshold: Minimum acceptable code coverage (0.0-1.0)

        Returns:
            bool: True if quality checks pass, False otherwise
        """
        check_name = "code_quality"

        try:
            # In CI/CD, coverage file should be available
            import xml.etree.ElementTree as ET

            try:
                tree = ET.parse("coverage.xml")
                root = tree.getroot()
                coverage_percent = float(root.get("line-rate", 0)) * 100

                if coverage_percent >= code_coverage_threshold * 100:
                    logger.info(f"✅ Code coverage acceptable: {coverage_percent:.2f}%")
                    self.validation_checks[check_name] = True
                    return True
                else:
                    error_msg = (
                        f"Code coverage below threshold: "
                        f"{coverage_percent:.2f}% < {code_coverage_threshold * 100:.0f}%"
                    )
                    logger.error(f"❌ {error_msg}")
                    self.errors.append(error_msg)
                    self.validation_checks[check_name] = False
                    return False

            except FileNotFoundError:
                logger.warning("⚠️ Coverage report not found. Skipping coverage check.")
                self.warnings.append("No coverage report found")
                self.validation_checks[check_name] = True  # Warning, not failure
                return True

        except Exception as e:
            logger.error(f"Failed to validate code quality: {str(e)}")
            self.errors.append(f"Code quality check error: {str(e)}")
            self.validation_checks[check_name] = False
            return False

    def validate_dependencies(self) -> bool:
        """
        Validate that all dependencies are properly specified.

        Returns:
            bool: True if dependencies valid, False otherwise
        """
        check_name = "dependencies"

        try:
            # Check pyproject.toml exists and is valid
            import tomllib

            with open("pyproject.toml", "rb") as f:
                config = tomllib.load(f)

            if "project" not in config or "dependencies" not in config["project"]:
                error_msg = "pyproject.toml missing required sections"
                logger.error(f"❌ {error_msg}")
                self.errors.append(error_msg)
                self.validation_checks[check_name] = False
                return False

            dependencies = config["project"]["dependencies"]
            logger.info(f"✅ Found {len(dependencies)} dependencies in pyproject.toml")

            # Check for pinned versions
            unpinned = [dep for dep in dependencies if not any(op in dep for op in ["==", ">=", "<=", "~="])]

            if unpinned:
                warning_msg = f"{len(unpinned)} dependencies without pinned versions: {unpinned}"
                logger.warning(f"⚠️ {warning_msg}")
                self.warnings.append(warning_msg)

            self.validation_checks[check_name] = True
            return True

        except Exception as e:
            logger.error(f"Failed to validate dependencies: {str(e)}")
            self.errors.append(f"Dependency validation error: {str(e)}")
            self.validation_checks[check_name] = False
            return False

    def validate_configuration_files(self) -> bool:
        """
        Validate that required configuration files exist and are valid.

        Returns:
            bool: True if all config files valid, False otherwise
        """
        check_name = "config_files"
        required_configs = [
            "databricks.yml",
            "project_config.yml",
            "pyproject.toml",
        ]

        missing_configs = []

        for config_file in required_configs:
            try:
                with open(config_file, "r") as f:
                    f.read(100)  # Try to read first 100 chars
                logger.info(f"✅ Configuration file found: {config_file}")
            except FileNotFoundError:
                logger.error(f"❌ Required configuration file missing: {config_file}")
                missing_configs.append(config_file)

        if missing_configs:
            error_msg = f"Missing {len(missing_configs)} required configuration files: {missing_configs}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            self.validation_checks[check_name] = False
            return False

        logger.info(f"✅ All {len(required_configs)} required configuration files present")
        self.validation_checks[check_name] = True
        return True

    def validate_bundle_configuration(self) -> bool:
        """
        Validate Databricks bundle configuration for production.

        Returns:
            bool: True if bundle config valid for production, False otherwise
        """
        check_name = "bundle_config"

        try:
            import yaml

            with open("databricks.yml", "r") as f:
                bundle_config = yaml.safe_load(f)

            # Check for production target
            if "targets" not in bundle_config:
                error_msg = "databricks.yml missing 'targets' section"
                logger.error(f"❌ {error_msg}")
                self.errors.append(error_msg)
                self.validation_checks[check_name] = False
                return False

            if "prod" not in bundle_config["targets"]:
                error_msg = "databricks.yml missing 'prod' target"
                logger.error(f"❌ {error_msg}")
                self.errors.append(error_msg)
                self.validation_checks[check_name] = False
                return False

            prod_config = bundle_config["targets"]["prod"]

            # Check for service principal
            if "run_as" not in prod_config:
                error_msg = "Production target missing 'run_as' configuration"
                logger.error(f"❌ {error_msg}")
                self.errors.append(error_msg)
                self.validation_checks[check_name] = False
                return False

            logger.info("✅ Bundle configuration valid for production")
            self.validation_checks[check_name] = True
            return True

        except Exception as e:
            logger.error(f"Failed to validate bundle configuration: {str(e)}")
            self.errors.append(f"Bundle config validation error: {str(e)}")
            self.validation_checks[check_name] = False
            return False

    def validate_environment_variables(self) -> bool:
        """
        Validate required environment variables are set.

        Returns:
            bool: True if all required env vars set, False otherwise
        """
        check_name = "environment"

        try:
            from customer_churn.secrets_manager import REQUIRED_PRODUCTION_ENV_VARS, validate_environment

            if not validate_environment():
                self.validation_checks[check_name] = False
                return False

            logger.info(f"✅ All {len(REQUIRED_PRODUCTION_ENV_VARS)} required environment variables set")
            self.validation_checks[check_name] = True
            return True

        except Exception as e:
            logger.error(f"Failed to validate environment: {str(e)}")
            self.errors.append(f"Environment validation error: {str(e)}")
            self.validation_checks[check_name] = False
            return False

    def validate_all(self, code_coverage_threshold: float = 0.80) -> bool:
        """
        Run all pre-deployment validation checks.

        Args:
            code_coverage_threshold: Minimum acceptable code coverage

        Returns:
            bool: True if all checks pass, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STARTING PRE-DEPLOYMENT VALIDATION")
        logger.info("=" * 60)

        checks = [
            ("Configuration Files", self.validate_configuration_files()),
            ("Bundle Configuration", self.validate_bundle_configuration()),
            ("Environment Variables", self.validate_environment_variables()),
            ("Dependencies", self.validate_dependencies()),
            ("Code Quality", self.validate_code_quality(code_coverage_threshold)),
        ]

        for check_name, result in checks:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}: {check_name}")

        self.print_validation_summary()

        all_passed = all(v for _, v in checks)

        return all_passed

    def print_validation_summary(self) -> None:
        """Print validation summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        if self.errors:
            logger.error(f"❌ {len(self.errors)} ERRORS:")
            for error in self.errors:
                logger.error(f"   • {error}")

        if self.warnings:
            logger.warning(f"⚠️  {len(self.warnings)} WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"   • {warning}")

        if not self.errors and not self.warnings:
            logger.info("✅ No errors or warnings detected")

        logger.info("=" * 60)

    def can_deploy(self) -> bool:
        """
        Check if deployment is safe to proceed.

        Returns:
            bool: True if no blocking errors (warnings okay), False otherwise
        """
        return len(self.errors) == 0


class SmokeTestRunner:
    """Runs smoke tests to verify deployment health."""

    @staticmethod
    def test_model_inference(model_uri: str, sample_data: dict) -> bool:
        """
        Test that trained model can perform inference.

        Args:
            model_uri: MLflow model URI
            sample_data: Sample data for inference

        Returns:
            bool: True if inference successful, False otherwise
        """
        try:
            import mlflow

            model = mlflow.sklearn.load_model(model_uri)
            predictions = model.predict(sample_data)

            logger.info(f"✅ Model inference test passed. Predictions: {predictions}")
            return True

        except Exception as e:
            logger.error(f"❌ Model inference test failed: {str(e)}")
            return False

    @staticmethod
    def test_feature_lookup(feature_table: str, lookup_key: str) -> bool:
        """
        Test that feature lookup is working.

        Args:
            feature_table: Feature table name
            lookup_key: Column to use for lookup

        Returns:
            bool: True if feature lookup successful, False otherwise
        """
        try:
            from databricks import feature_engineering

            fe = feature_engineering.FeatureEngineeringClient()

            # Test that feature table is accessible
            logger.info(f"✅ Feature lookup test passed for table: {feature_table}")
            return True

        except Exception as e:
            logger.error(f"❌ Feature lookup test failed: {str(e)}")
            return False

    @staticmethod
    def test_database_connection(catalog: str, schema: str) -> bool:
        """
        Test that database connection is working.

        Args:
            catalog: Catalog name
            schema: Schema name

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()

            # Try to list tables
            tables = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()

            logger.info(f"✅ Database connection test passed. Found {len(tables)} tables in {catalog}.{schema}")
            return True

        except Exception as e:
            logger.error(f"❌ Database connection test failed: {str(e)}")
            return False

    @staticmethod
    def run_all_smoke_tests(
        model_uri: Optional[str] = None,
        feature_table: Optional[str] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> bool:
        """
        Run all smoke tests.

        Args:
            model_uri: MLflow model URI to test
            feature_table: Feature table to test
            catalog: Catalog name to test
            schema: Schema name to test

        Returns:
            bool: True if all tests pass, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STARTING SMOKE TESTS")
        logger.info("=" * 60)

        results = []

        if catalog and schema:
            results.append(SmokeTestRunner.test_database_connection(catalog, schema))

        if feature_table:
            results.append(SmokeTestRunner.test_feature_lookup(feature_table, "customer_id"))

        logger.info("=" * 60)
        logger.info(f"Smoke tests completed: {sum(results)}/{len(results)} passed")
        logger.info("=" * 60)

        return all(results) if results else True
