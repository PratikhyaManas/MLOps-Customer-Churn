"""
This script evaluates a trained customer churn model.

Key functionality:
- Loads the trained model from MLflow
- Evaluates model performance on test set
- Logs evaluation metrics
- Determines if model should be deployed
"""

import argparse
import sys
import traceback

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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

    # Get model URI from previous task
    model_uri = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_uri")
    logger.info(f"Retrieved model URI: {model_uri}")

    # Load test data
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    target = config.target[0].new_name

    test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
    X_test = test_set.drop(columns=[target])
    y_test = test_set[target]

    # Load model and make predictions
    logger.info("Loading model and making predictions...")
    model = mlflow.sklearn.load_model(model_uri)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"Evaluation Metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  AUC: {auc:.4f}")

    # Set task values
    dbutils.jobs.taskValues.set(key="accuracy", value=accuracy)
    dbutils.jobs.taskValues.set(key="precision", value=precision)
    dbutils.jobs.taskValues.set(key="recall", value=recall)
    dbutils.jobs.taskValues.set(key="f1_score", value=f1)
    dbutils.jobs.taskValues.set(key="auc_score", value=auc)

    # Determine if model should be deployed (e.g., AUC > 0.75)
    deploy_threshold = 0.75
    should_deploy = 1 if auc >= deploy_threshold else 0
    dbutils.jobs.taskValues.set(key="should_deploy", value=should_deploy)

    logger.info(f"Model evaluation completed. Should deploy: {should_deploy}")

except FileNotFoundError as e:
    logger.error(f"Model or configuration file not found: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid model or metric value: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error in model evaluation workflow: {type(e).__name__}: {str(e)}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    sys.exit(1)
