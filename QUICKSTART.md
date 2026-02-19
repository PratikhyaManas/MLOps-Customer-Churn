# Quick Start Guide - Customer Churn MLOps Project

## 🎯 What Was Created

A complete MLOps project for **Customer Churn Prediction** based on the credit default template, including:

### ✅ Project Structure
- **27 files** organized in a production-ready structure
- Complete source code for data processing and ML workflows
- Comprehensive test suite
- CI/CD configuration ready
- Databricks asset bundle configuration

### 📂 Key Components Created

#### 1. **Configuration Files** ✓
- `project_config.yml` - Project settings and feature definitions
- `databricks.yml` - Databricks asset bundle configuration
- `pyproject.toml` - Python package configuration
- `Makefile` - Build automation commands
- `pytest.ini` - Test configuration
- `.pre-commit-config.yaml` - Code quality hooks

#### 2. **Source Code** ✓
Location: `src/customer_churn/`
- `utils.py` - Configuration management and logging
- `data_cleaning.py` - Data cleaning pipeline
- `data_cleaning_spark.py` - Spark-based data cleaning
- `data_preprocessing.py` - Feature preprocessing
- `data_preprocessing_spark.py` - Spark-based preprocessing

#### 3. **Workflow Scripts** ✓
Location: `workflows/`
- `preprocess.py` - Data preprocessing workflow
- `train_model.py` - Model training with MLflow
- `evaluate_model.py` - Model evaluation
- `deploy_model.py` - Model deployment
- `refresh_monitor.py` - Monitoring refresh

#### 4. **Tests** ✓
Location: `tests/`
- `test_data_cleaning.py` - Data cleaning tests
- `test_data_preprocessor.py` - Preprocessing tests

#### 5. **Notebooks** ✓
Location: `notebooks/feature_engineering/`
- `basic_mlflow_experiment_notebook.py` - Sample MLflow experiment

#### 6. **Documentation** ✓
- `README.md` - Comprehensive project documentation
- `data/README.md` - Data schema documentation

## 🚀 Next Steps

### 1. Set Up Your Environment

```bash
# Navigate to the project
cd mlops-databricks-customer-churn

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure Your Data

1. **Get customer churn dataset**:
   - Download from [Kaggle Telco Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
   - Or use your own customer data
   - Place in `data/churn_data.csv`

2. **Update configuration**:
   - Edit `project_config.yml` to match your data schema
   - Adjust feature names if needed

### 3. Configure Databricks

1. **Update `databricks.yml`**:
   - Set your workspace URL
   - Configure cluster settings
   - Set catalog and schema names

2. **Set up Unity Catalog**:
   ```sql
   CREATE CATALOG IF NOT EXISTS churn;
   CREATE SCHEMA IF NOT EXISTS churn.default;
   ```

### 4. Local Development

```bash
# Run tests
pytest tests/ -v

# Format code
make format

# Lint code
make lint

# Run all checks
make all
```

### 5. Deploy to Databricks

```bash
# Install Databricks CLI
pip install databricks-cli

# Configure authentication
databricks configure --token

# Validate bundle
databricks bundle validate

# Deploy to dev
databricks bundle deploy -t dev

# Run workflow
databricks bundle run customer-churn -t dev
```

## 🔑 Key Differences from Credit Default

| Aspect | Credit Default | Customer Churn |
|--------|----------------|----------------|
| **Domain** | Financial risk | Customer retention |
| **Target** | Default (payment failure) | Churn (customer leaving) |
| **Features** | Credit history, payments | Tenure, services, billing |
| **Use Case** | Loan approval decisions | Retention campaigns |
| **Key Metrics** | Precision (avoid bad loans) | Recall (catch churners) |

## 📊 Feature Schema

The project expects these features:

### Customer Features
- `customer_id`: Unique identifier
- `tenure`: Months as customer
- `senior_citizen`: Senior citizen flag
- `partner`: Has partner
- `dependents`: Has dependents

### Service Features
- `phone_service`: Has phone service
- `multiple_lines`: Multiple phone lines
- `internet_service`: Internet service type
- `online_security`: Online security service
- `online_backup`: Online backup service
- `device_protection`: Device protection
- `tech_support`: Tech support service
- `streaming_tv`: TV streaming service
- `streaming_movies`: Movie streaming service

### Account Features
- `contract_type`: Contract duration
- `paperless_billing`: Paperless billing
- `payment_method`: Payment method
- `monthly_charges`: Monthly bill
- `total_charges`: Total charges to date

### Target
- `churn`: Customer churned (0/1)

## 🎯 Expected Model Performance

Target metrics for deployment:
- **AUC-ROC**: > 0.75
- **Recall**: > 0.65 (important for churn)
- **Precision**: > 0.70
- **F1 Score**: > 0.70

## 📝 Customization Tips

### Adding New Features

1. Update `project_config.yml`:
```yaml
num_features:
  - name: your_new_feature
    dtype: float64
```

2. Update data cleaning logic in `data_cleaning.py`

3. Add to feature list:
```yaml
features:
  clean:
    - your_new_feature
```

### Changing the Model

Edit `workflows/train_model.py`:
```python
from xgboost import XGBClassifier  # Instead of LightGBM

model = XGBClassifier(
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100
)
```

### Adjusting Hyperparameters

Edit `project_config.yml`:
```yaml
parameters:
  learning_rate: 0.1  # Change from 0.05
  max_depth: 8        # Add new parameters
  n_estimators: 200
```

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the virtual environment
   ```bash
   .venv\Scripts\activate
   ```

2. **Missing data**: Check that `data/churn_data.csv` exists

3. **Databricks connection**: Verify your token and workspace URL

4. **Feature mismatch**: Ensure your data columns match `project_config.yml`

## 📚 Resources

- [Project README](README.md) - Full documentation
- [Databricks Docs](https://docs.databricks.com/)
- [MLflow Guide](https://mlflow.org/docs/latest/index.html)
- [Original Template](https://github.com/benitomartin/mlops-databricks-credit-default)

## 🎉 You're Ready!

Your customer churn MLOps project is fully set up and ready to:
- ✅ Process customer data
- ✅ Train churn prediction models
- ✅ Deploy to production
- ✅ Monitor model performance
- ✅ Scale with CI/CD

**Happy ML Engineering! 🚀**

---

For questions or issues, refer to the main [README.md](README.md) or open an issue.
