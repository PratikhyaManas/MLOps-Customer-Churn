"""Data validation and quality checks for customer churn pipeline."""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


class DataValidator:
    """Validates data schema, types, and quality."""

    def __init__(self, schema: Dict[str, str]):
        """
        Initialize DataValidator with expected schema.

        Args:
            schema: Dictionary mapping column names to expected data types
                   Example: {"name": "object", "age": "int64", "salary": "float64"}
        """
        self.schema = schema
        self.validation_errors = []

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has all required columns with correct types.

        Args:
            df: DataFrame to validate

        Returns:
            bool: True if schema is valid, False otherwise
        """
        self.validation_errors = []

        # Check all required columns exist
        missing_columns = set(self.schema.keys()) - set(df.columns)
        if missing_columns:
            error = f"Missing required columns: {missing_columns}"
            logger.error(error)
            self.validation_errors.append(error)
            return False

        # Check column types
        for col, expected_dtype in self.schema.items():
            actual_dtype = str(df[col].dtype)
            if actual_dtype != expected_dtype:
                error = f"Column '{col}': expected {expected_dtype}, got {actual_dtype}"
                logger.error(error)
                self.validation_errors.append(error)

        return len(self.validation_errors) == 0

    def validate_required_fields(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """
        Validate that required columns have no null values.

        Args:
            df: DataFrame to validate
            required_cols: List of column names that must not be null

        Returns:
            bool: True if all required fields are non-null, False otherwise
        """
        null_counts = df[required_cols].isnull().sum()
        has_nulls = null_counts.sum() > 0

        if has_nulls:
            for col, count in null_counts[null_counts > 0].items():
                error = f"Column '{col}': contains {count} null values"
                logger.error(error)
                self.validation_errors.append(error)
            return False

        logger.info(f"All {len(required_cols)} required columns are non-null")
        return True

    def validate_numeric_ranges(
        self, df: pd.DataFrame, column_ranges: Dict[str, Tuple[float, float]]
    ) -> bool:
        """
        Validate numeric columns are within expected ranges.

        Args:
            df: DataFrame to validate
            column_ranges: Dict mapping column names to (min, max) tuples

        Returns:
            bool: True if all columns within range, False otherwise
        """
        all_valid = True

        for col, (min_val, max_val) in column_ranges.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue

            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            count = out_of_range.sum()

            if count > 0:
                actual_min = df[col].min()
                actual_max = df[col].max()
                error = (
                    f"Column '{col}': {count} values outside range [{min_val}, {max_val}]. "
                    f"Actual range [{actual_min}, {actual_max}]"
                )
                logger.error(error)
                self.validation_errors.append(error)
                all_valid = False
            else:
                logger.info(f"Column '{col}' values within range [{min_val}, {max_val}]")

        return all_valid

    def validate_unique_values(
        self, df: pd.DataFrame, unique_constraints: Dict[str, int]
    ) -> bool:
        """
        Validate columns don't have too many duplicates.

        Args:
            df: DataFrame to validate
            unique_constraints: Dict mapping column names to max duplicate ratio (0.0-1.0)

        Returns:
            bool: True if duplicate ratios acceptable, False otherwise
        """
        all_valid = True

        for col, max_dup_ratio in unique_constraints.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue

            unique_count = df[col].nunique()
            total_count = len(df)
            unique_ratio = unique_count / total_count

            if unique_ratio < (1 - max_dup_ratio):
                error = (
                    f"Column '{col}': {unique_count} unique values out of {total_count} rows. "
                    f"Duplicate ratio: {1 - unique_ratio:.2%}"
                )
                logger.error(error)
                self.validation_errors.append(error)
                all_valid = False
            else:
                logger.info(f"Column '{col}': {unique_count} unique values {unique_ratio:.2%}")

        return all_valid

    def validate_no_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> bool:
        """
        Validate that DataFrame has no duplicate rows.

        Args:
            df: DataFrame to validate
            subset: List of columns to consider for duplicates. If None, all columns used.

        Returns:
            bool: True if no duplicates, False otherwise
        """
        dup_count = df.duplicated(subset=subset).sum()

        if dup_count > 0:
            error = f"Found {dup_count} duplicate rows" + (
                f" (considering columns {subset})" if subset else ""
            )
            logger.error(error)
            self.validation_errors.append(error)
            return False

        logger.info("No duplicate rows found")
        return True

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        required_cols: Optional[List[str]] = None,
        column_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        unique_constraints: Optional[Dict[str, float]] = None,
        check_duplicates: bool = False,
    ) -> bool:
        """
        Run comprehensive data quality validation.

        Args:
            df: DataFrame to validate
            required_cols: Columns that must not be null
            column_ranges: Numeric column range constraints
            unique_constraints: Duplicate ratio constraints
            check_duplicates: Whether to check for exact row duplicates

        Returns:
            bool: True if all validations pass, False otherwise
        """
        logger.info(f"Starting data quality validation on DataFrame with shape {df.shape}")
        results = {"schema": False, "required": False, "ranges": False, "unique": False, "duplicates": False}

        # Validate schema
        if not self.validate_schema(df):
            logger.warning("Schema validation failed")
            return False
        results["schema"] = True

        # Validate required fields
        if required_cols:
            results["required"] = self.validate_required_fields(df, required_cols)

        # Validate numeric ranges
        if column_ranges:
            results["ranges"] = self.validate_numeric_ranges(df, column_ranges)

        # Validate uniqueness
        if unique_constraints:
            results["unique"] = self.validate_unique_values(df, unique_constraints)

        # Check for duplicates
        if check_duplicates:
            results["duplicates"] = self.validate_no_duplicates(df)

        all_passed = all(v for v in results.values() if v is not False)

        if all_passed:
            logger.info("✅ All data quality checks passed")
        else:
            logger.error(f"❌ Data quality validation failed. Results: {results}")

        return all_passed

    def get_validation_report(self) -> str:
        """
        Get a formatted validation error report.

        Returns:
            str: Formatted error report
        """
        if not self.validation_errors:
            return "✅ No validation errors"

        report = "❌ Validation Errors:\n"
        for i, error in enumerate(self.validation_errors, 1):
            report += f"  {i}. {error}\n"

        return report

    @staticmethod
    def get_expected_schema() -> Dict[str, str]:
        """
        Get the expected schema for customer churn dataset.

        Returns:
            Dict: Column name to dtype mapping
        """
        return {
            "customer_id": "object",
            "tenure": "int64",
            "monthly_charges": "float64",
            "total_charges": "float64",
            "contract_type": "int64",
            "payment_method": "int64",
            "internet_service": "int64",
            "online_security": "int64",
            "online_backup": "int64",
            "device_protection": "int64",
            "tech_support": "int64",
            "streaming_tv": "int64",
            "streaming_movies": "int64",
            "paperless_billing": "int64",
            "senior_citizen": "int64",
            "partner": "int64",
            "dependents": "int64",
            "phone_service": "int64",
            "multiple_lines": "int64",
            "gender": "int64",
            "churn": "int64",
        }
