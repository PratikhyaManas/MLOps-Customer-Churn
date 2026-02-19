## Databricks notebook source
import os

import numpy as np
import pandas as pd
from databricks.connect import DatabricksSession
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError
from pyspark.sql import SparkSession

from customer_churn.utils import Config, Target

# Load environment variables
load_dotenv()

spark = DatabricksSession.builder.getOrCreate()

FILEPATH_DATABRICKS = os.environ.get("FILEPATH_DATABRICKS", "/databricks/driver/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
CLEANING_LOGS = os.environ.get("CLEANING_LOGS", "logs/data_cleaning_spark.log")


class DataCleaning:
    """
    A class for cleaning and preprocessing customer churn data using Spark.

    Attributes:
        config (Config): Configuration model containing preprocessing settings
        df (pd.DataFrame): DataFrame containing the data to be processed
        target_config (Target): Configuration for target variable
        spark (SparkSession): Spark session for data processing
    """

    def __init__(self, filepath: str, config: Config, spark: SparkSession):
        """
        Initializes the DataCleaning class with Spark support.

        Args:
            filepath (str): Path to the CSV file containing the data
            config (Config): Configuration model containing preprocessing settings
            spark (SparkSession): Active Spark session

        Raises:
            Exception: If data cleaning fails
        """
        self.config = config
        self.spark = spark
        self.df = self._load_data(filepath)
        self._setup_target_config()

    def _setup_target_config(self) -> None:
        """Sets up target configuration from config."""
        target_info = self.config.target[0]
        self.target_config = Target(name=target_info.name, dtype=target_info.dtype, new_name=target_info.new_name)

    def _load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads and validates the input data using Spark.

        Args:
            filepath (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded DataFrame

        Raises:
            Exception: If data loading or validation fails
        """
        try:
            logger.info(f"Loading data from {filepath}")
            df = self.spark.read.csv(filepath, header=True, inferSchema=True).toPandas()
            if df.empty:
                raise Exception("Loaded DataFrame is empty")
            return df
        except Exception as e:
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
        """Validates data types of key columns."""
        target_col = self.target_config.name
        if not np.issubdtype(self.df[target_col].dtype, np.number):
            raise Exception(f"Target column '{target_col}' must be numeric")

    def clean_data(self) -> pd.DataFrame:
        """
        Executes the complete data cleaning pipeline.

        Returns:
            pd.DataFrame: Cleaned DataFrame

        Raises:
            Exception: If data cleaning pipeline fails
        """
        try:
            logger.info("Starting data preprocessing with Spark")
            self._rename_and_capitalize_columns()
            self._validate_preprocessed_data()

            logger.info("Data cleaning completed successfully")
            logger.info(f"Final data shape: {self.df.shape}")
            return self.df

        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise

    def _rename_and_capitalize_columns(self) -> None:
        """Renames and capitalizes key columns."""
        self.df.rename(columns={self.target_config.name: self.target_config.new_name}, inplace=True)
        self.df.columns = [col.capitalize() if col else col for col in self.df.columns]
        if "customer_id" in self.df.columns or "Customer_id" in self.df.columns:
            id_col = "customer_id" if "customer_id" in self.df.columns else "Customer_id"
            self.df[id_col] = self.df[id_col].astype("str")
        logger.info("Renamed and capitalized columns")

    def _validate_preprocessed_data(self) -> None:
        """Validates the preprocessed data."""
        logger.info("Validating preprocessed data")
        if self.df.empty:
            raise Exception("Preprocessed DataFrame is empty")
        logger.info("Validation completed successfully")
