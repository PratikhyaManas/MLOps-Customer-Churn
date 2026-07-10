import os
from typing import Tuple

from dotenv import load_dotenv
from loguru import logger

from customer_churn.data_cleaning import DataCleaning
from customer_churn.preprocessing_common import create_preprocessor, extract_features_and_target, log_processed_shapes
from customer_churn.utils import Config, load_config, setup_logging

# Load environment variables
load_dotenv()

FILEPATH = os.environ.get("FILEPATH", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PREPROCESSING_LOGS = os.environ.get("PREPROCESSING_LOGS", "logs/data_preprocessing.log")


class DataPreprocessor:
    """
    A class for preprocessing customer churn data, including scaling features.

    Attributes:
        data_cleaning (DataCleaning): An instance of the DataCleaning class used for data preprocessing.
        cleaned_data (pd.DataFrame): The cleaned DataFrame after preprocessing.
        features_robust (list): List of feature names for robust scaling.
        X (pd.DataFrame): Features DataFrame after cleaning.
        y (pd.Series): Target Series after cleaning.
        preprocessor (ColumnTransformer): ColumnTransformer for scaling the features.
    """

    def __init__(self, filepath: str, config: Config):
        """
        Initializes the DataPreprocessor class.

        Args:
            filepath (str): The path to the CSV file containing the data.
            config (Config): The configuration model containing preprocessing settings.
        """
        try:
            # Initialize DataCleaning to preprocess data
            logger.info("Initializing data cleaning process")
            self.data_cleaning = DataCleaning(filepath, config)
            self.cleaned_data = self.data_cleaning.clean_data()
            logger.info("Data cleaning process completed")

            # Define robust features for scaling from config
            self.features_robust = config.features.robust

            # Define features and target
            self.X, self.y = extract_features_and_target(self.cleaned_data, config)

            # Set up the ColumnTransformer for scaling
            logger.info("Setting up ColumnTransformer for scaling")
            self.preprocessor = create_preprocessor(self.features_robust)
        except KeyError as e:
            logger.error(f"KeyError encountered during initialization: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"An error occurred during initialization: {str(e)}")
            raise

    def get_processed_data(self) -> Tuple:
        """
        Retrieves the processed features, target, and preprocessor.

        Returns:
            Tuple: A tuple containing:
                - pd.DataFrame: The features DataFrame.
                - pd.Series: The target Series.
                - ColumnTransformer: The preprocessor for scaling.
        """
        try:
            logger.info("Retrieving processed data and preprocessor")
            log_processed_shapes(self.X, self.y)

            return self.X, self.y, self.preprocessor

        except Exception as e:
            logger.error(f"An error occurred during data preprocessing: {str(e)}")
            raise


if __name__ == "__main__":
    # Configure logger using setup_logging
    setup_logging(PREPROCESSING_LOGS)

    # Load configuration from YAML file
    config = load_config("project_config.yml")

    # Test the DataPreprocessor class
    try:
        logger.info(f"Initializing DataPreprocessor with config: {config}")
        preprocessor = DataPreprocessor(FILEPATH, config)
        X, y, preprocessor_model = preprocessor.get_processed_data()

    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {str(e)}")

    logger.info("DataPreprocessor script completed")
