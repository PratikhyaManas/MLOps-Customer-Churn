"""
This script deploys a customer churn model to production.

Key functionality:
- Retrieves model from MLflow
- Registers model to Unity Catalog
- Transitions model to production stage
- Creates model serving endpoint
"""

import argparse
import sys
import traceback

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
from loguru import logger
from pyspark.sql import SparkSession

from customer_churn.utils import load_config, setup_logging

# Set up logging
setup_logging(log_file="")

try:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
    parser.add_argument("--git_sha", action="store", default=None, type=str, required=True)
    parser.add_argument("--job_run_id", action="store", default=None, type=str, required=True)
    args = parser.parse_args()

    root_path = args.root_path
    git_sha = args.git_sha
    job_run_id = args.job_run_id
    logger.info("Parsed arguments successfully.")

    # Load configuration
    logger.info("Loading configuration...")
    config_path = f"{root_path}/project_config.yml"
    config = load_config(config_path)
    logger.info("Configuration loaded successfully.")

    # Initialize clients
    workspace = WorkspaceClient()
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    logger.info("Clients initialized.")

    # Get model info from previous tasks
    model_uri = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_uri")
    should_deploy = dbutils.jobs.taskValues.get(taskKey="evaluate_model", key="should_deploy")

    logger.info(f"Model URI: {model_uri}")
    logger.info(f"Should deploy: {should_deploy}")

    if should_deploy == 1:
        # Register model to Unity Catalog
        catalog_name = config.catalog_name
        schema_name = config.schema_name
        model_name = f"{catalog_name}.{schema_name}.customer_churn_model"

        logger.info(f"Registering model to: {model_name}")
        
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        logger.info(f"Model registered. Version: {model_version.version}")

        # Transition model to production
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=model_version.version
        )

        logger.info(f"Model transitioned to production. Version: {model_version.version}")

        # Set task values
        dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
        dbutils.jobs.taskValues.set(key="deployment_status", value="success")

        logger.info("Model deployment completed successfully.")
    else:
        logger.info("Model did not meet deployment criteria. Skipping deployment.")
        dbutils.jobs.taskValues.set(key="deployment_status", value="skipped")

except FileNotFoundError as e:
    logger.error(f"Model or configuration file not found: {str(e)}")
    logger.error(traceback.format_exc())
    dbutils.jobs.taskValues.set(key="deployment_status", value="failed")
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid model URI or deployment configuration: {str(e)}")
    logger.error(traceback.format_exc())
    dbutils.jobs.taskValues.set(key="deployment_status", value="failed")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error in model deployment workflow: {type(e).__name__}: {str(e)}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    dbutils.jobs.taskValues.set(key="deployment_status", value="failed")
    sys.exit(1)
