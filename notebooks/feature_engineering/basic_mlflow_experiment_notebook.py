# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Churn Prediction - Basic MLflow Experiment
# MAGIC 
# MAGIC This notebook demonstrates a basic MLflow experiment for customer churn prediction using LightGBM.
# MAGIC 
# MAGIC ## Objectives:
# MAGIC - Load and prepare customer churn data
# MAGIC - Train a LightGBM model
# MAGIC - Track experiment with MLflow
# MAGIC - Log model and metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Configuration

# COMMAND ----------

# Load project configuration
config_path = "/Workspace/.bundle/mlops-databricks-customer-churn/dev/files/project_config.yml"

import yaml

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print("Configuration loaded successfully")
print(f"Catalog: {config['catalog_name']}")
print(f"Schema: {config['schema_name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Data

# COMMAND ----------

# Load training data from catalog
catalog_name = config["catalog_name"]
schema_name = config["schema_name"]

train_data = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_data = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prepare Features and Target

# COMMAND ----------

# Define target column
target = config["target"][0]["new_name"]

# Split features and target
X_train = train_data.drop(columns=[target])
y_train = train_data[target]
X_test = test_data.drop(columns=[target])
y_test = test_data[target]

print(f"Features shape: {X_train.shape}")
print(f"Target distribution in training set:")
print(y_train.value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature Scaling

# COMMAND ----------

# Apply robust scaling to numerical features
scaler = RobustScaler()
numerical_features = ["tenure", "monthly_charges", "total_charges"]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

print("Feature scaling completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train Model with MLflow Tracking

# COMMAND ----------

# Set MLflow experiment
mlflow.set_experiment("/Users/your_email@example.com/customer-churn-basic-experiment")

# Start MLflow run
with mlflow.start_run(run_name="lightgbm-basic") as run:
    
    # Define model parameters
    params = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 100,
        "random_state": 42,
        "force_col_wise": True
    }
    
    # Train model
    model = LGBMClassifier(**params)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Log parameters
    mlflow.log_params(params)
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print("Model Evaluation Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"\nRun ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feature Importance

# COMMAND ----------

import matplotlib.pyplot as plt

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This notebook demonstrated:
# MAGIC - Loading customer churn data from Databricks
# MAGIC - Preparing features with scaling
# MAGIC - Training a LightGBM model
# MAGIC - Tracking experiments with MLflow
# MAGIC - Evaluating model performance
# MAGIC 
# MAGIC Next steps:
# MAGIC - Tune hyperparameters
# MAGIC - Try feature engineering
# MAGIC - Deploy model to serving endpoint
