## MLOps Customer Churn Prediction 🎯

<p align="center">
<img width="737" alt="cover" src="https://github.com/user-attachments/assets/a1c18fba-9e39-45b5-8fcd-bceb1f5f5af9">
</p>

This is a comprehensive MLOps project for customer churn prediction built on Databricks. The project demonstrates end-to-end machine learning operations including data engineering, model training, deployment, monitoring, and CI/CD automation.

**Inspired by the credit default prediction MLOps template, this project adapts the same robust architecture for customer churn use cases.**

## 🎯 Project Overview

Customer churn (customer attrition) is when customers stop doing business with a company. This project predicts which customers are likely to churn, enabling proactive retention strategies.

**Key Features:**
- Production-ready MLOps pipeline on Databricks
- Feature engineering with Databricks Feature Store
- MLflow experiment tracking and model registry
- Automated model deployment and serving
- Real-time inference with feature lookup
- Model monitoring and drift detection
- Fully automated CI/CD with GitHub Actions
- Comprehensive testing and code quality checks
- Enhanced error handling with stack traces
- Data validation and quality checks
- Health monitoring and safeguards

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=Databricks&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-FFFFFF?style=for-the-badge&logo=apachespark&logoColor=#E35A16)
![MLflow](https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=MLflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## 📁 Project Structure

```
mlops-databricks-customer-churn/
├── .github/workflows/          # GitHub Actions CI/CD
│   └── pipeline.yml            # Unified CI/CD workflow
├── data/                       # Raw data directory
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/                  # Databricks notebooks
│   ├── create_source_data/
│   ├── feature_engineering/
│   ├── model_feature_serving/
│   └── monitoring/
├── src/                        # Source code modules
│   └── customer_churn/
│       ├── data_cleaning.py
│       ├── data_preprocessing.py
│       ├── data_validator.py       # Data quality validation
│       ├── health_checks.py        # Health monitoring
│       ├── secrets_manager.py      # Secrets management
│       ├── deployment_validator.py # Pre-deployment checks
│       └── utils.py
├── tests/                      # Unit & smoke tests
│   ├── test_data_cleaning.py
│   ├── test_data_preprocessor.py
│   └── test_smoke.py           # Deployment validation tests
├── workflows/                  # Databricks job workflows
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── deploy_model.py
│   └── refresh_monitor.py
├── .env.example                # Environment configuration
├── .gitignore
├── .pre-commit-config.yaml
├── bundle_monitoring.yml
├── databricks.yml              # Production-ready bundle config
├── Makefile
├── project_config.yml
├── pyproject.toml
├── pytest.ini
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Databricks workspace
- Git & GitHub account
- `uv` package manager (recommended)

### Local Development Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/yourusername/mlops-databricks-customer-churn.git
   cd mlops-databricks-customer-churn
   ```

2. **Create environment:**
   ```bash
   uv venv -p 3.11.0 .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -r pyproject.toml --all-extras
   uv lock
   ```

4. **Build package:**
   ```bash
   uv build
   ```

5. **Run tests:**
   ```bash
   make test
   # or: pytest tests/ -v
   ```

### Data Setup

Place your dataset in the `data/` folder. The pipeline expects the Telco Customer Churn dataset:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Update data paths in `src/customer_churn/data_*.py` files if needed.

## 🧪 Testing & Code Quality

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_data_cleaning.py -v

# Run with coverage
pytest tests/ --cov=src/customer_churn --cov-report=html
```

### Code Quality Checks

```bash
# Format code
make format

# Lint code
make lint

# Run all checks (format, lint, test)
make all
```

### Security Testing (SAST)

Static Application Security Testing (SAST) is automatically run in CI/CD:

```bash
# Run Semgrep SAST locally
semgrep --config=p/security-audit --config=p/owasp-top-ten --config=p/python src/

# Run Bandit for Python security issues
bandit -r src/

# Check for known vulnerabilities
safety check
```

**SAST Tools Used:**
- **Semgrep**: Pattern-based code scanner for security vulnerabilities, code quality, and OWASP compliance
- **Bandit**: Detects common security issues in Python code
- **Safety**: Checks for known vulnerabilities in dependencies
- **GitLeaks**: Detects hardcoded secrets and credentials

### Smoke Tests

Smoke tests validate deployment health:
```bash
pytest tests/test_smoke.py -v
```

## 🔄 MLOps Workflow

### 1. Data Preprocessing (`workflows/preprocess.py`)
- Load source data from Databricks catalog
- Validate data quality with schema checks
- Split into train/test sets (80/20)
- Update feature tables
- Trigger online feature refresh
- Handle errors gracefully with full logging

### 2. Model Training (`workflows/train_model.py`)
- Load training data with feature engineering
- Create preprocessing pipeline (scaling, encoding)
- Train LightGBM model with optimized hyperparameters
- Log experiments and metrics with MLflow
- Calculate AUC-ROC, precision, recall, F1
- Store model artifacts

### 3. Model Evaluation (`workflows/evaluate_model.py`)
- Load trained model from MLflow
- Evaluate on held-out test set
- Calculate comprehensive metrics
- Determine if meets deployment threshold (AUC > 0.75)
- Log evaluation results

### 4. Model Deployment (`workflows/deploy_model.py`)
- Register model to Unity Catalog
- Transition to production stage
- Create/update serving endpoint
- Configure real-time inference
- Only deploy if meets quality threshold

### 5. Monitoring (`workflows/refresh_monitor.py`)
- Refresh monitoring dashboards
- Track model performance over time
- Monitor for data and model drift
- Alert on anomalies
- Update inference tables

## 📊 Data & Features

### Dataset

Telco Customer Churn dataset with 7,043 customers and 20 features:
- **Churn**: Target variable (44.7% positive class)
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Tenure**: Months as customer (0-72 months)
- **Charges**: Monthly & total charges
- **Services**: Phone, Internet, Security, Backup, TV, Movies, Tech Support
- **Contract**: Month-to-month, One year, Two year

### Feature Schema

```yaml
Required Columns:
  - customer_id (int64)
  - tenure (int64): 0-72 months
  - monthly_charges (float64)
  - total_charges (float64)
  - contract_type, payment_method, internet_service (int64)
  - online_security, streaming_tv, tech_support (int64)
  - churn (int64): 0 or 1 (target)
```

### Data Quality Validation

The pipeline includes automatic validation:
```python
from customer_churn.data_validator import DataValidator

validator = DataValidator(DataValidator.get_expected_schema())
validator.validate_data_quality(
    df,
    required_cols=["customer_id", "churn"],
    column_ranges={"tenure": (0, 100)},
    check_duplicates=True
)
```

## 🤖 Model & Performance

### Model Configuration

LightGBM classifier with optimized hyperparameters:
```yaml
learning_rate: 0.05
random_state: 42
force_col_wise: true
max_depth: 7
num_leaves: 31
```

### Expected Performance

- **AUC-ROC**: > 0.75 (deployment threshold)
- **Precision**: > 0.70
- **Recall**: > 0.65
- **F1 Score**: > 0.70

### Feature Importance

Top features for churn prediction:
1. Contract type (month-to-month increases risk)
2. Tenure (longer tenure = lower churn)
3. Monthly charges
4. Internet service selection
5. Tech support availability

## 🔐 Production Deployment

### Configuration Management

Update `databricks.yml` with your settings:
```yaml
bundle:
  name: mlops-databricks-customer-churn
  
targets:
  dev:
    mode: development
    
  staging:
    mode: development
    
  prod:
    mode: production
    run_as:
      service_principal_name: ${env.DATABRICKS_SERVICE_PRINCIPAL_NAME}
```

### Environment Setup

Create `.env` file with required variables:
```bash
# Required for all environments
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=<your-token>

# Required for production
DATABRICKS_SERVICE_PRINCIPAL_NAME=<sp-client-id>
GIT_SHA=<git-commit-sha>
ENVIRONMENT=prod

# Optional
ALERT_EMAIL=team@example.com
LOG_LEVEL=INFO
```

### Databricks Deployment

```bash
# Install Databricks CLI
pip install databricks-cli

# Validate bundle
databricks bundle validate --target prod

# Deploy to production
databricks bundle deploy --target prod

# Run as job
databricks bundle run customer-churn --target prod
```

### Service Principal Setup

Create a service principal in Databricks:

1. **Go to Admin Settings → Service Principals**
2. **Create Service Principal** → Note Client ID
3. **Generate Token** → Save securely
4. **Grant Permissions:**
   ```sql
   GRANT USE_WORKSPACE ON WORKSPACE TO `<client-id>`;
   GRANT USE_CATALOG ON CATALOG churn TO `<client-id>`;
   GRANT USE_SCHEMA ON SCHEMA churn.default TO `<client-id>`;
   ```

## 🔄 CI/CD with GitHub Actions

### Workflow Overview

The `pipeline.yml` workflow automatically:

**On Every Push:**
- ✅ Format & lint code (Ruff)
- ✅ Run unit tests (pytest)
- ✅ Check code coverage (80% threshold)
- ✅ Run SAST security scans (Semgrep, Bandit, Safety)
- ✅ Check for hardcoded secrets (GitLeaks)
- ✅ Build wheel package

**On Merge to Main:**
- ✅ Deploy to Databricks
- ✅ Run smoke tests
- ✅ Send notifications
- ✅ Validate health checks

### GitHub Actions Setup

1. **Configure Secrets** in repository Settings → Secrets:

   | Secret | Value |
   |--------|-------|
   | `DATABRICKS_HOST` | Workspace URL |
   | `DATABRICKS_TOKEN` | Service principal token |
   | `DATABRICKS_SERVICE_PRINCIPAL_NAME` | SP Client ID |

2. **Enable Actions**:
   - Settings → Actions → General
   - Select "Read and write permissions"

3. **Create Environment** (optional):
   - Settings → Environments → New environment
   - Name: `databricks-production`
   - Add required reviewers for approval

4. **Branch Protection** (recommended):
   - Settings → Branches → Add rule for `main`
   - Require PR reviews (1+)
   - Require status checks to pass
   - Dismiss stale reviews

### Workflow Triggers

```yaml
# Trigger on push to main or develop
on:
  push:
    branches: [main, develop]
    
  # Trigger on pull requests
  pull_request:
    branches: [main, develop]
```

### Troubleshooting CI/CD

**Tests Failing Locally But Passing in CI:**
- Check Python version: Must be 3.11
- Verify no hardcoded paths
- Check .gitignore isn't excluding needed files

**Secrets Not Found:**
- Verify secret names match exactly (case-sensitive)
- Check secrets are in correct repository
- Re-run workflow after adding secrets

**Deployment Fails:**
- Verify Databricks host URL is correct
- Check service principal has required permissions
- Ensure no workspace/job name conflicts

## 🛡️ Production Safeguards

### Pre-Deployment Validation

Automatic checks before deployment:

```python
from customer_churn.deployment_validator import DeploymentValidator

validator = DeploymentValidator()
is_ready = validator.validate_all()

# Checks:
# - Code coverage threshold
# - Configuration files exist
# - Bundle is production-ready
# - All required env vars set
# - Dependencies specified correctly
```

### Health Checks

Comprehensive health monitoring:

```python
from customer_churn.health_checks import HealthChecker

checker = HealthChecker()
checker.check_all(
    required_packages=["pandas", "sklearn", "lightgbm"],
    check_databricks=True,
    check_spark=True,
    tables_to_check=[("churn", "default", "source_data")]
)
```

### Data Quality Validation

Automatic data validation during preprocessing:

```python
from customer_churn.data_validator import DataValidator

validator = DataValidator(schema)
validator.validate_data_quality(
    df,
    required_cols=["customer_id", "churn"],
    column_ranges={"tenure": (0, 100), "monthly_charges": (0, 200)},
    check_duplicates=True
)
```

### Secrets Management

Secure secret handling:

```python
from customer_churn.secrets_manager import SecretsManager, validate_environment

# Validate environment is properly configured
validate_environment()

# Use secrets manager
manager = SecretsManager()
token = manager.get_secret("DATABRICKS_TOKEN", required=True)
```

## 🔍 Error Handling & Logging

### Enhanced Error Handling

All workflow scripts include:
- Specific exception types (not generic catch-all)
- Full stack traces logged
- Graceful cleanup on failure
- Proper exit codes

Example:
```python
except FileNotFoundError as e:
    logger.error(f"Configuration not found: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid configuration: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)
```

### Logging Configuration

Logging setup with rotation:
```python
from customer_churn.utils import setup_logging

setup_logging(
    log_file="logs/pipeline.log",
    log_level="INFO"
)
```

## 📚 Available Modules

### Core Modules

| Module | Purpose |
|--------|---------|
| `data_cleaning.py` | Handle missing values, duplicates, outliers |
| `data_preprocessing.py` | Feature scaling and encoding |
| `data_validator.py` | Data quality and schema validation |
| `health_checks.py` | System health monitoring |
| `secrets_manager.py` | Secure secrets and config management |
| `deployment_validator.py` | Pre-deployment validation |
| `utils.py` | Configuration loading and utilities |

### Usage Examples

```python
# Data Validation
from customer_churn.data_validator import DataValidator
validator = DataValidator(expec ted_schema)
validator.validate_data_quality(df)

# Health Checks
from customer_churn.health_checks import HealthChecker
checker = HealthChecker()
checker.check_all()

# Secrets Management
from customer_churn.secrets_manager import SecretsManager
secrets = SecretsManager()
token = secrets.get_secret("DATABRICKS_TOKEN")

# Deployment Validation
from customer_churn.deployment_validator import DeploymentValidator
validator = DeploymentValidator()
validator.validate_all()
```

## 🚨 Monitoring & Alerts

### Built-in Alerts

The pipeline monitors:
- Data freshness (24-hour max age)
- Row count thresholds
- Schema changes
- Model performance degradation
- Disk space availability

### Optional Notifications

Configure email alerts:
```bash
export ALERT_EMAIL=team@example.com
```

Job failures automatically notify configured emails.

### Drift Detection

Detect data and model drift:
```python
from customer_churn.health_checks import DataDriftDetector

drift = DataDriftDetector.calculate_statistical_drift(
    reference_data, 
    current_data,
    threshold=0.05
)
```

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run checks: `make all`
5. Commit with clear messages
6. Push and create Pull Request

### Best Practices

- Write descriptive commit messages
- Add tests for new features
- Update documentation
- Follow code style (enforced by Ruff)
- Pass all CI/CD checks

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by MLOps Credit Default template
- Built on Databricks, MLflow, and LightGBM
- Community contributions and feedback

## 📧 Support

For issues, questions, or feedback:
1. Check existing GitHub issues
2. Open a new issue with detailed description
3. Contact team via email (see CODEOWNERS)

---

⭐ **Star this repo if you find it helpful!**
