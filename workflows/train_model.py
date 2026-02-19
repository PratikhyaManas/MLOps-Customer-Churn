"""
This script trains a LightGBM model for customer churn prediction with feature engineering.

Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM classifier
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups
- Outputs model URI for downstream tasks
"""

import argparse
import sys
import traceback

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

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
    logger.debug(f"Git SHA: {git_sha}")
    logger.debug(f"Job Run ID: {job_run_id}")
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
    fe = feature_engineering.FeatureEngineeringClient()
    logger.info("Spark session and Feature Engineering client initialized.")

    # Extract configuration details
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    target = config.target[0].new_name
    parameters = config.parameters
    features_robust = config.features.robust
    columns = config.features.clean
    columns_wo_id = columns.copy()
    columns_wo_id.remove("customer_id")

    # Convert test set to Pandas DataFrame
    test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

    # Create training set from feature table
    feature_table_name = f"{catalog_name}.{schema_name}.features_balanced"
    columns_to_drop = columns_wo_id + ["Update_timestamp_utc"]
    train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(*columns_to_drop)
    logger.info(f"Train set columns for feature table: {train_set.columns}")

    # Set MLflow tracking
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    training_set = fe.create_training_set(
        df=train_set,
        label=target,
        feature_lookups=[
            FeatureLookup(
                table_name=feature_table_name,
                lookup_key="customer_id",
            )
        ],
        exclude_columns=["Update_timestamp_utc"],
    )

    training_df = training_set.load_df().toPandas()
    logger.info(f"Training set loaded. Shape: {training_df.shape}")

    # Prepare features and target
    X_train = training_df.drop(columns=[target])
    y_train = training_df[target]
    X_test = test_set.drop(columns=[target])
    y_test = test_set[target]

    logger.info(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    logger.info(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Create preprocessing and model pipeline
    preprocessor = ColumnTransformer(
        transformers=[("robust_scaler", RobustScaler(), features_robust)],
        remainder="passthrough",
    )

    model = LGBMClassifier(
        learning_rate=parameters["learning_rate"],
        random_state=parameters["random_state"],
        force_col_wise=parameters["force_col_wise"],
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    # Start MLflow run
    experiment_name = f"/Users/{workspace.current_user.me().user_name}/customer-churn-experiment"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(tags={"git_sha": git_sha, "job_run_id": job_run_id}) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started. Run ID: {run_id}")

        # Train the model
        logger.info("Training model...")
        pipeline.fit(X_train, y_train)
        logger.info("Model training completed.")

        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"Model AUC Score: {auc_score:.4f}")

        # Log parameters and metrics
        mlflow.log_params(parameters)
        mlflow.log_metric("auc", auc_score)

        # Infer signature and log model
        signature = infer_signature(X_train, y_train)
        
        fe.log_model(
            model=pipeline,
            artifact_path="customer-churn-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            signature=signature,
            registered_model_name=f"{catalog_name}.{schema_name}.customer_churn_model",
        )

        logger.info(f"Model logged successfully. Run ID: {run_id}")
        logger.info(f"Model URI: runs:/{run_id}/customer-churn-model")

        # Set task value for workflow coordination
        dbutils.jobs.taskValues.set(key="model_uri", value=f"runs:/{run_id}/customer-churn-model")
        dbutils.jobs.taskValues.set(key="auc_score", value=auc_score)

    logger.info("Model training workflow completed successfully.")

except FileNotFoundError as e:
    logger.error(f"Configuration or data file not found: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid value in model training: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error in model training workflow: {type(e).__name__}: {str(e)}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    sys.exit(1)
