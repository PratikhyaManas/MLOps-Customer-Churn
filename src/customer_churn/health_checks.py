"""Health checks and monitoring utilities for production pipelines."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger


class HealthChecker:
    """Performs health checks on pipeline components."""

    def __init__(self):
        """Initialize HealthChecker."""
        self.checks: Dict[str, bool] = {}
        self.check_run_times: Dict[str, float] = {}

    def check_data_freshness(
        self, last_update_time: datetime, max_age_hours: float = 24
    ) -> bool:
        """
        Check if data is fresh enough for pipeline execution.

        Args:
            last_update_time: Timestamp of last data update
            max_age_hours: Maximum acceptable age in hours

        Returns:
            bool: True if data is fresh, False otherwise
        """
        age = datetime.utcnow() - last_update_time
        max_age = timedelta(hours=max_age_hours)

        is_fresh = age < max_age

        if is_fresh:
            logger.info(f"✅ Data freshness check passed. Age: {age.total_seconds() / 3600:.2f} hours")
        else:
            logger.error(f"❌ Data freshness check failed. Age: {age.total_seconds() / 3600:.2f} hours (max: {max_age_hours})")

        self.checks["data_freshness"] = is_fresh
        return is_fresh

    def check_table_exists(self, catalog: str, schema: str, table: str) -> bool:
        """
        Check if a Databricks table exists.

        Args:
            catalog: Catalog name
            schema: Schema/database name
            table: Table name

        Returns:
            bool: True if table exists, False otherwise
        """
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()

            full_table_name = f"{catalog}.{schema}.{table}"

            # Try to describe the table
            spark.sql(f"DESCRIBE TABLE {full_table_name}")
            logger.info(f"✅ Table exists: {full_table_name}")
            self.checks[f"table_exists_{table}"] = True
            return True

        except Exception as e:
            logger.error(f"❌ Table check failed for {full_table_name}: {str(e)}")
            self.checks[f"table_exists_{table}"] = False
            return False

    def check_table_row_count(
        self, catalog: str, schema: str, table: str, min_rows: int = 1
    ) -> bool:
        """
        Check if table has minimum row count.

        Args:
            catalog: Catalog name
            schema: Schema/database name
            table: Table name
            min_rows: Minimum acceptable row count

        Returns:
            bool: True if row count meets minimum, False otherwise
        """
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()

            full_table_name = f"{catalog}.{schema}.{table}"

            row_count = spark.sql(f"SELECT COUNT(*) as count FROM {full_table_name}").collect()[0][0]

            if row_count >= min_rows:
                logger.info(f"✅ Table {table} has {row_count} rows (min: {min_rows})")
                self.checks[f"row_count_{table}"] = True
                return True
            else:
                logger.error(
                    f"❌ Table {table} has {row_count} rows (min: {min_rows})"
                )
                self.checks[f"row_count_{table}"] = False
                return False

        except Exception as e:
            logger.error(f"❌ Row count check failed for {full_table_name}: {str(e)}")
            self.checks[f"row_count_{table}"] = False
            return False

    def check_dependencies_available(self, required_packages: List[str]) -> bool:
        """
        Check if required Python packages are available.

        Args:
            required_packages: List of package names to check

        Returns:
            bool: True if all packages available, False otherwise
        """
        missing_packages = []

        for package_name in required_packages:
            try:
                __import__(package_name)
                logger.debug(f"✅ Package available: {package_name}")
            except ImportError:
                logger.error(f"❌ Required package not found: {package_name}")
                missing_packages.append(package_name)

        if missing_packages:
            logger.error(f"Missing {len(missing_packages)} required packages: {missing_packages}")
            self.checks["dependencies"] = False
            return False

        logger.info(f"✅ All {len(required_packages)} required packages available")
        self.checks["dependencies"] = True
        return True

    def check_databricks_connection(self) -> bool:
        """
        Check if Databricks connection is available.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            from databricks.sdk import WorkspaceClient

            workspace = WorkspaceClient()
            current_user = workspace.current_user.me()
            logger.info(f"✅ Databricks connection successful. User: {current_user.user_name}")
            self.checks["databricks_connection"] = True
            return True

        except Exception as e:
            logger.error(f"❌ Databricks connection failed: {str(e)}")
            self.checks["databricks_connection"] = False
            return False

    def check_spark_available(self) -> bool:
        """
        Check if Spark session is available.

        Returns:
            bool: True if Spark available, False otherwise
        """
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()

            # Run a simple test
            test_result = spark.sql("SELECT 1 as test").collect()[0][0]

            logger.info(f"✅ Spark session available and functional")
            self.checks["spark_available"] = True
            return True

        except Exception as e:
            logger.error(f"❌ Spark session check failed: {str(e)}")
            self.checks["spark_available"] = False
            return False

    def check_disk_space_available(self, min_space_gb: float = 1.0) -> bool:
        """
        Check if minimum disk space is available.

        Args:
            min_space_gb: Minimum required space in GB

        Returns:
            bool: True if sufficient space available, False otherwise
        """
        try:
            import shutil

            stat = shutil.disk_usage("/")
            available_gb = stat.free / (1024**3)

            if available_gb >= min_space_gb:
                logger.info(f"✅ Sufficient disk space available: {available_gb:.2f} GB (min: {min_space_gb} GB)")
                self.checks["disk_space"] = True
                return True
            else:
                logger.error(f"❌ Insufficient disk space: {available_gb:.2f} GB (min: {min_space_gb} GB)")
                self.checks["disk_space"] = False
                return False

        except Exception as e:
            logger.error(f"❌ Disk space check failed: {str(e)}")
            self.checks["disk_space"] = False
            return False

    def check_all(
        self,
        required_packages: Optional[List[str]] = None,
        check_databricks: bool = True,
        check_spark: bool = True,
        tables_to_check: Optional[List[tuple]] = None,
    ) -> bool:
        """
        Run all health checks.

        Args:
            required_packages: List of packages to check
            check_databricks: Whether to check Databricks connection
            check_spark: Whether to check Spark availability
            tables_to_check: List of (catalog, schema, table) tuples to check

        Returns:
            bool: True if all checks pass, False otherwise
        """
        logger.info("Starting comprehensive health checks...")

        results = []

        # Check dependencies
        if required_packages:
            results.append(self.check_dependencies_available(required_packages))

        # Check Databricks connection
        if check_databricks:
            results.append(self.check_databricks_connection())

        # Check Spark
        if check_spark:
            results.append(self.check_spark_available())

        # Check disk space
        results.append(self.check_disk_space_available())

        # Check specific tables
        if tables_to_check:
            for catalog, schema, table in tables_to_check:
                results.append(self.check_table_exists(catalog, schema, table))
                results.append(self.check_table_row_count(catalog, schema, table))

        all_passed = all(results)

        self.print_health_summary()

        return all_passed

    def print_health_summary(self) -> None:
        """Print summary of all health checks."""
        passed = sum(1 for v in self.checks.values() if v is True)
        failed = sum(1 for v in self.checks.values() if v is False)
        total = len(self.checks)

        logger.info("")
        logger.info("=" * 60)
        logger.info("HEALTH CHECK SUMMARY")
        logger.info("=" * 60)

        for check_name, status in sorted(self.checks.items()):
            icon = "✅" if status else "❌"
            logger.info(f"{icon} {check_name}: {'PASS' if status else 'FAIL'}")

        logger.info("=" * 60)
        logger.info(f"Results: {passed}/{total} passed, {failed}/{total} failed")
        logger.info("=" * 60)
        logger.info("")

    def get_health_status(self) -> Dict[str, bool]:
        """
        Get all check results.

        Returns:
            Dict mapping check names to pass/fail status
        """
        return self.checks.copy()

    def get_failed_checks(self) -> List[str]:
        """
        Get list of failed checks.

        Returns:
            List of failed check names
        """
        return [name for name, status in self.checks.items() if status is False]


class DataDriftDetector:
    """Detects data drift in model features and predictions."""

    @staticmethod
    def calculate_schema_drift(reference_df, current_df) -> Dict[str, any]:
        """
        Calculate schema drift between reference and current data.

        Args:
            reference_df: Reference DataFrame
            current_df: Current DataFrame

        Returns:
            Dict with drift metrics
        """
        reference_cols = set(reference_df.columns)
        current_cols = set(current_df.columns)

        missing_cols = reference_cols - current_cols
        new_cols = current_cols - reference_cols

        return {
            "missing_columns": list(missing_cols),
            "new_columns": list(new_cols),
            "has_drift": bool(missing_cols or new_cols),
        }

    @staticmethod
    def calculate_statistical_drift(reference_series, current_series, threshold: float = 0.05):
        """
        Calculate statistical drift using Kolmogorov-Smirnov test.

        Args:
            reference_series: Reference data series
            current_series: Current data series
            threshold: Significance threshold

        Returns:
            Dict with drift metrics
        """
        try:
            from scipy import stats

            if reference_series.dtype in ["float64", "int64"] and current_series.dtype in ["float64", "int64"]:
                statistic, p_value = stats.ks_2samp(reference_series.dropna(), current_series.dropna())

                return {
                    "ks_statistic": float(statistic),
                    "p_value": float(p_value),
                    "has_drift": p_value < threshold,
                }
            else:
                return {
                    "ks_statistic": None,
                    "p_value": None,
                    "has_drift": False,
                }

        except Exception as e:
            logger.error(f"Failed to calculate statistical drift: {str(e)}")
            return {"has_drift": None, "error": str(e)}
