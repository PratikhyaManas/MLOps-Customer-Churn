"""
This script evaluates a trained customer churn model.

Key functionality:
- Loads the trained model from MLflow
- Evaluates model performance on test set
- Logs evaluation metrics
- Determines if model should be deployed
"""

import sys

import mlflow
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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
    logger.info("Parsed arguments successfully.")

    # Load configuration
    logger.info("Loading configuration...")
    config, _, spark = load_workflow_context(root_path, with_workspace=False)
    logger.info("Configuration loaded successfully.")

    logger.info("Databricks clients initialized.")

    # Get model URI from previous task
    model_uri = get_task_value_safe(task_key="train_model", key="model_uri")
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

    logger.info("Evaluation Metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  AUC: {auc:.4f}")

    # Set task values
    set_task_value_safe(key="accuracy", value=accuracy)
    set_task_value_safe(key="precision", value=precision)
    set_task_value_safe(key="recall", value=recall)
    set_task_value_safe(key="f1_score", value=f1)
    set_task_value_safe(key="auc_score", value=auc)

    # Determine if model should be deployed (e.g., AUC > 0.75)
    deploy_threshold = 0.75
    should_deploy = 1 if auc >= deploy_threshold else 0
    set_task_value_safe(key="should_deploy", value=should_deploy)

    logger.info(f"Model evaluation completed. Should deploy: {should_deploy}")

except FileNotFoundError as e:
    logger.error(f"Model or configuration file not found: {str(e)}")
    log_workflow_failure("model evaluation workflow", e)
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid model or metric value: {str(e)}")
    log_workflow_failure("model evaluation workflow", e)
    sys.exit(1)
except Exception as e:
    log_workflow_failure("model evaluation workflow", e)
    sys.exit(1)
