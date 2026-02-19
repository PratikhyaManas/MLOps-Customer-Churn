## Databricks notebook source
import os
from typing import Tuple

from databricks.connect import DatabricksSession
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from customer_churn.data_cleaning_spark import DataCleaning
from customer_churn.utils import Config

# Load environment variables
load_dotenv()

spark = DatabricksSession.builder.getOrCreate()

FILEPATH_DATABRICKS = os.environ.get("FILEPATH_DATABRICKS", "/databricks/driver/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PREPROCESSING_LOGS = os.environ.get("PREPROCESSING_LOGS", "logs/data_preprocessing_spark.log")


class DataPreprocessor:
    """
    A class for preprocessing customer churn data using Spark, including scaling features.

    Attributes:
        data_cleaning (DataCleaning): An instance of the DataCleaning class
        cleaned_data (pd.DataFrame): The cleaned DataFrame after preprocessing
        features_robust (list): List of feature names for robust scaling
        X (pd.DataFrame): Features DataFrame after cleaning
        y (pd.Series): Target Series after cleaning
        preprocessor (ColumnTransformer): ColumnTransformer for scaling the features
    """

    def __init__(self, filepath: str, config: Config, spark: SparkSession):
        """
        Initializes the DataPreprocessor class with Spark support.

        Args:
            filepath (str): The path to the CSV file containing the data
            config (Config): The configuration model containing preprocessing settings
            spark (SparkSession): Active Spark session
        """
        try:
            logger.info("Initializing data cleaning process with Spark")
            self.data_cleaning = DataCleaning(filepath, config, spark)
            self.cleaned_data = self.data_cleaning.clean_data()
            logger.info("Data cleaning process completed")

            # Define robust features for scaling from config
            self.features_robust = config.features.robust

            # Define features and target
            self.X = self.cleaned_data.drop(columns=[target.new_name for target in config.target])
            self.y = self.cleaned_data[config.target[0].new_name]

            # Set up the ColumnTransformer for scaling
            logger.info("Setting up ColumnTransformer for scaling")
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("robust_scaler", RobustScaler(), self.features_robust)
                ],
                remainder="passthrough",
            )
        except Exception as e:
            logger.error(f"An error occurred during initialization: {str(e)}")
            raise

    def get_processed_data(self) -> Tuple:
        """
        Retrieves the processed features, target, and preprocessor.

        Returns:
            Tuple: A tuple containing:
                - pd.DataFrame: The features DataFrame
                - pd.Series: The target Series
                - ColumnTransformer: The preprocessor for scaling
        """
        try:
            logger.info("Retrieving processed data and preprocessor")
            logger.info(f"Feature columns in X: {self.X.columns.tolist()}")
            logger.info(f"Data preprocessing completed. Shape of X: {self.X.shape}, Shape of y: {self.y.shape}")
            return self.X, self.y, self.preprocessor
        except Exception as e:
            logger.error(f"An error occurred during data preprocessing: {str(e)}")
            raise
