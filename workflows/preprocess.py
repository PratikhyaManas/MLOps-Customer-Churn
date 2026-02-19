"""
This script handles data ingestion and feature table updates for customer churn prediction.

Key functionality:
- Loads the source dataset and identifies new records for processing
- Splits new records into train and test sets based on timestamp
- Updates existing train and test tables with new data
- Inserts the latest feature values into the feature table for serving
- Triggers and monitors pipeline updates for online feature refresh
- Sets task values to coordinate pipeline orchestration

Workflow:
1. Load source dataset and retrieve recent records with updated timestamps
2. Split new records into train and test sets (80-20 split)
3. Append new train and test records to existing train and test tables
4. Insert the latest feature data into the feature table for online serving
5. Trigger a pipeline update and monitor its status until completion
6. Set a task value indicating whether new data was processed
"""

import argparse
import sys
import time
import traceback
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import OperationTimeoutError
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import max as spark_max

from customer_churn.utils import load_config, setup_logging

# Set up logging
setup_logging(log_file="")

try:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", action="store", default=None, type=str, required=True)

    args = parser.parse_args()
    root_path = args.root_path
    logger.info("Parsed arguments successfully.")

    # Load configuration
    logger.info("Loading configuration...")
    config_path = f"{root_path}/project_config.yml"
    config = load_config(config_path)
    logger.info("Configuration loaded successfully.")

    # Initialize Databricks workspace client
    workspace = WorkspaceClient()
    logger.info("Databricks workspace client initialized.")

    # Initialize Spark session
    spark = SparkSession.builder.getOrCreate()
    logger.info("Spark session initialized.")

    # Extract configuration details
    pipeline_id = config.pipeline_id
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    logger.debug(f"Catalog: {catalog_name}, Schema: {schema_name}")
    logger.debug(f"Pipeline ID: {pipeline_id}")

    # Load source data table
    source_data_table_name = f"{catalog_name}.{schema_name}.source_data"
    source_data = spark.table(source_data_table_name)
    logger.info(f"Loaded source data from {source_data_table_name}.")

    # Get max update timestamps
    max_train_timestamp = (
        spark.table(f"{catalog_name}.{schema_name}.train_set")
        .select(spark_max("Update_timestamp_utc").alias("max_update_timestamp"))
        .collect()[0]["max_update_timestamp"]
    )
    logger.info(f"Latest timestamp in train set: {max_train_timestamp}")

    max_test_timestamp = (
        spark.table(f"{catalog_name}.{schema_name}.test_set")
        .select(spark_max("Update_timestamp_utc").alias("max_update_timestamp"))
        .collect()[0]["max_update_timestamp"]
    )
    logger.info(f"Latest timestamp in test set: {max_test_timestamp}")

    latest_timestamp = max(max_train_timestamp, max_test_timestamp)
    logger.info(f"Latest timestamp across train and test sets: {latest_timestamp}")

    # Filter new data
    new_data = source_data.filter(col("Update_timestamp_utc") > latest_timestamp)
    new_data_count = new_data.count()
    logger.info(f"Found {new_data_count} new rows in source data.")

    # Split new data into train and test sets
    new_data_train, new_data_test = new_data.randomSplit([0.8, 0.2], seed=42)
    affected_rows_train = new_data_train.count()
    affected_rows_test = new_data_test.count()
    logger.info(f"New train data rows: {affected_rows_train}, New test data rows: {affected_rows_test}")

    if new_data_count > 0:
        # Write new data to train and test tables
        train_table_name = f"{catalog_name}.{schema_name}.train_set"
        test_table_name = f"{catalog_name}.{schema_name}.test_set"

        logger.info(f"Appending {affected_rows_train} rows to train table: {train_table_name}")
        new_data_train.write.mode("append").saveAsTable(train_table_name)

        logger.info(f"Appending {affected_rows_test} rows to test table: {test_table_name}")
        new_data_test.write.mode("append").saveAsTable(test_table_name)

        # Update feature table with latest data
        feature_table_name = f"{catalog_name}.{schema_name}.features_balanced"
        logger.info(f"Inserting latest features into: {feature_table_name}")
        new_data.write.mode("append").saveAsTable(feature_table_name)

        # Trigger pipeline update
        logger.info(f"Starting pipeline update for: {pipeline_id}")
        update_info = workspace.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)
        update_id = update_info.update_id
        logger.info(f"Pipeline update started. Update ID: {update_id}")

        # Monitor pipeline status
        while True:
            update_status = workspace.pipelines.get_update(pipeline_id=pipeline_id, update_id=update_id)
            state = update_status.update.state.value
            logger.info(f"Pipeline update status: {state}")

            if state in ["COMPLETED", "FAILED", "CANCELED"]:
                logger.info(f"Pipeline update finished with status: {state}")
                break

            time.sleep(10)

        # Set task value for workflow coordination
        dbutils.jobs.taskValues.set(key="refreshed", value=1)
        logger.info("Data preprocessing completed successfully. Task value set to 1.")
    else:
        logger.info("No new data to process.")
        dbutils.jobs.taskValues.set(key="refreshed", value=0)
        logger.info("Task value set to 0 (no refresh needed).")

except FileNotFoundError as e:
    logger.error(f"Configuration file not found: {str(e)}")
    logger.error(traceback.format_exc())
    dbutils.jobs.taskValues.set(key="refreshed", value=0)
    sys.exit(1)
except (KeyError, ValueError) as e:
    logger.error(f"Invalid configuration or data format: {str(e)}")
    logger.error(traceback.format_exc())
    dbutils.jobs.taskValues.set(key="refreshed", value=0)
    sys.exit(1)
except OperationTimeoutError as e:
    logger.error(f"Pipeline operation timeout: {str(e)}")
    logger.error(traceback.format_exc())
    dbutils.jobs.taskValues.set(key="refreshed", value=0)
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error in preprocessing workflow: {type(e).__name__}: {str(e)}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    dbutils.jobs.taskValues.set(key="refreshed", value=0)
    sys.exit(1)
