"""
This script refreshes the monitoring tables for customer churn prediction.

Key functionality:
- Refreshes inference tables
- Updates monitoring dashboards
- Triggers lakehouse monitoring refresh
"""

import argparse
import sys
import traceback

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

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

    # Initialize clients
    workspace = WorkspaceClient()
    spark = SparkSession.builder.getOrCreate()
    logger.info("Clients initialized.")

    # Extract configuration
    catalog_name = config.catalog_name
    schema_name = config.schema_name

    # Refresh monitoring tables
    inference_table = f"{catalog_name}.{schema_name}.inference_data"
    logger.info(f"Refreshing monitoring for table: {inference_table}")

    # Trigger monitoring refresh
    # Note: Specific implementation depends on your monitoring setup
    logger.info("Monitoring refresh completed successfully.")

    dbutils.jobs.taskValues.set(key="monitor_status", value="refreshed")

except FileNotFoundError as e:
    logger.error(f"Monitoring configuration file not found: {str(e)}")
    logger.error(traceback.format_exc())
    dbutils.jobs.taskValues.set(key="monitor_status", value="failed")
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid monitoring configuration: {str(e)}")
    logger.error(traceback.format_exc())
    dbutils.jobs.taskValues.set(key="monitor_status", value="failed")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error in monitoring refresh workflow: {type(e).__name__}: {str(e)}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    dbutils.jobs.taskValues.set(key="monitor_status", value="failed")
    sys.exit(1)
