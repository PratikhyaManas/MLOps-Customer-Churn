import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError

from customer_churn.utils import Config, Target, load_config, setup_logging

# Load environment variables
load_dotenv()

FILEPATH = os.environ.get("FILEPATH", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
CLEANING_LOGS = os.environ.get("CLEANING_LOGS", "logs/data_cleaning.log")


class DataCleaning:
    """
    A class for cleaning and preprocessing customer churn data.

    Attributes:
        config (Config): Configuration model containing preprocessing settings
        df (pd.DataFrame): DataFrame containing the data to be processed
        target_config (Target): Configuration for target variable
    """

    def __init__(self, filepath: str, config: Config):
        """
        Initializes the DataCleaning class.

        Args:
            filepath (str): Path to the CSV file containing the data
            config (Config): Configuration model containing preprocessing settings

        Raises:
            FileNotFoundError: If data file doesn't exist
            Exception: If data cleaning fails
        """
        self._validate_file_exists(filepath)
        self.config = config
        self.df = self._load_data(filepath)
        self._setup_target_config()

    @staticmethod
    def _validate_file_exists(filepath: str) -> None:
        """
        Validates that the input file exists.

        Args:
            filepath (str): Path to the CSV file containing the data

        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"The file {filepath} does not exist")

    def _setup_target_config(self) -> None:
        """Sets up target configuration from config."""
        target_info = self.config.target[0]
        self.target_config = Target(name=target_info.name, dtype=target_info.dtype, new_name=target_info.new_name)

    @staticmethod
    def _load_data(filepath: str) -> pd.DataFrame:
        """
        Loads and validates the input data.

        Args:
            filepath (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded DataFrame

        Raises:
            Exception: If data loading or validation fails
        """
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            if df.empty:
                raise Exception("Loaded DataFrame is empty")
            return df
        except pd.errors.EmptyDataError as e:
            raise Exception(f"Failed to load data: {str(e)}") from e

    def _validate_columns(self) -> None:
        """
        Validates that required columns exist in the DataFrame.

        Raises:
            Exception: If DataFrame validation fails
        """
        columns_to_check = [feature.name for feature in self.config.num_features] + [self.target_config.name]
        missing_columns = [col for col in columns_to_check if col not in self.df.columns]
        if missing_columns:
            raise Exception(f"Missing required columns: {', '.join(missing_columns)}")

    def _validate_data_types(self) -> None:
        """
        Validates and converts data types according to configuration.

        Raises:
            Exception: If data type conversion fails
        """
        try:
            logger.info("Validating and converting data types")
            
            # Convert numerical features
            for feature in self.config.num_features:
                if feature.name in self.df.columns:
                    self.df[feature.name] = self.df[feature.name].astype(feature.dtype)
            
            # Convert target variable
            if self.target_config.name in self.df.columns:
                self.df[self.target_config.name] = self.df[self.target_config.name].astype(self.target_config.dtype)
            
            logger.info("Data types validated and converted successfully")
        except Exception as e:
            logger.error(f"Failed to validate/convert data types: {str(e)}")
            raise

    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handles missing values in the dataset.

        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        # Check for missing values
        missing_count = self.df.isnull().sum()
        if missing_count.sum() > 0:
            logger.warning(f"Found {missing_count.sum()} missing values")
            logger.info(f"Missing values per column:\n{missing_count[missing_count > 0]}")
            
            # Drop rows with missing target
            if self.df[self.target_config.name].isnull().any():
                logger.info(f"Dropping rows with missing target variable: {self.target_config.name}")
                self.df = self.df.dropna(subset=[self.target_config.name])
            
            # Fill missing values for numerical features with median
            for feature in self.config.num_features:
                if feature.name in self.df.columns and self.df[feature.name].isnull().any():
                    median_value = self.df[feature.name].median()
                    self.df[feature.name].fillna(median_value, inplace=True)
                    logger.info(f"Filled missing values in {feature.name} with median: {median_value}")
        else:
            logger.info("No missing values found")
        
        return self.df

    def handle_duplicates(self) -> pd.DataFrame:
        """
        Identifies and removes duplicate rows.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        logger.info("Checking for duplicate rows")
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        final_count = len(self.df)
        duplicates_removed = initial_count - final_count
        
        if duplicates_removed > 0:
            logger.warning(f"Removed {duplicates_removed} duplicate rows")
        else:
            logger.info("No duplicate rows found")
        
        return self.df

    def handle_outliers(self, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """
        Handles outliers using IQR method.

        Args:
            method (str): Method to use for outlier detection (default: "iqr")
            threshold (float): IQR threshold multiplier (default: 1.5)

        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        logger.info(f"Handling outliers using {method} method with threshold {threshold}")
        
        numerical_cols = [f.name for f in self.config.num_features if f.dtype == "float64"]
        
        for col in numerical_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                if outliers > 0:
                    logger.info(f"Column {col}: {outliers} outliers detected")
                    # Cap outliers instead of removing
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return self.df

    def rename_target(self) -> pd.DataFrame:
        """
        Renames the target column according to configuration.

        Returns:
            pd.DataFrame: DataFrame with renamed target column
        """
        if self.target_config.name in self.df.columns:
            logger.info(f"Renaming target column from {self.target_config.name} to {self.target_config.new_name}")
            self.df = self.df.rename(columns={self.target_config.name: self.target_config.new_name})
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """
        Executes the complete data cleaning pipeline.

        Returns:
            pd.DataFrame: Cleaned DataFrame

        Raises:
            Exception: If data cleaning pipeline fails
        """
        try:
            logger.info("Starting data cleaning pipeline")
            
            # Validate columns and data types
            self._validate_columns()
            self._validate_data_types()
            
            # Execute cleaning steps
            self.handle_missing_values()
            self.handle_duplicates()
            self.handle_outliers()
            self.rename_target()
            
            logger.info(f"Data cleaning completed successfully. Final shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            logger.error(f"Data cleaning pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Setup logging
    setup_logging(log_file=CLEANING_LOGS)
    
    # Load configuration
    config = load_config("project_config.yml")
    
    # Initialize and run data cleaning
    cleaner = DataCleaning(filepath=FILEPATH, config=config)
    cleaned_data = cleaner.clean_data()
    
    # Save cleaned data
    output_path = "data/churn_data_cleaned.csv"
    cleaned_data.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")
