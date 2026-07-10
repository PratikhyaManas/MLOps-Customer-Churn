"""
This script deploys a customer churn model to production.

Key functionality:
- Retrieves model from MLflow
- Registers model to Unity Catalog
- Transitions model to production stage
- Creates model serving endpoint
"""

import sys

import mlflow
from loguru import logger

from customer_churn.utils import setup_logging
from customer_churn.workflow_common import (
    get_task_value_safe,
    load_workflow_context,
    log_workflow_failure,
    parse_common_args,
    set_task_value_safe,
)

# Set up logging
setup_logging(log_file="")

try:
    # Parse arguments
    args = parse_common_args(require_git_metadata=True)

    root_path = args.root_path
    git_sha = args.git_sha
    job_run_id = args.job_run_id
    logger.info("Parsed arguments successfully.")

    # Load configuration
    logger.info("Loading configuration...")
    config, _, _ = load_workflow_context(root_path, with_workspace=False, with_spark=False)
    logger.info("Configuration loaded successfully.")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    logger.info("Databricks clients initialized.")

    # Get model info from previous tasks
    model_uri = get_task_value_safe(task_key="train_model", key="model_uri")
    should_deploy = get_task_value_safe(task_key="evaluate_model", key="should_deploy", default=0)

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
        set_task_value_safe(key="model_version", value=model_version.version)
        set_task_value_safe(key="deployment_status", value="success")

        logger.info("Model deployment completed successfully.")
    else:
        logger.info("Model did not meet deployment criteria. Skipping deployment.")
        set_task_value_safe(key="deployment_status", value="skipped")

except FileNotFoundError as e:
    logger.error(f"Model or configuration file not found: {str(e)}")
    log_workflow_failure("model deployment workflow", e, task_key="deployment_status", task_value="failed")
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid model URI or deployment configuration: {str(e)}")
    log_workflow_failure("model deployment workflow", e, task_key="deployment_status", task_value="failed")
    sys.exit(1)
except Exception as e:
    log_workflow_failure("model deployment workflow", e, task_key="deployment_status", task_value="failed")
    sys.exit(1)
