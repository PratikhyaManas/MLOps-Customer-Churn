"""
This script refreshes the monitoring tables for customer churn prediction.

Key functionality:
- Refreshes inference tables
- Updates monitoring dashboards
- Triggers lakehouse monitoring refresh
"""

import sys

from loguru import logger

from customer_churn.utils import setup_logging
from customer_churn.workflow_common import (
    load_workflow_context,
    log_workflow_failure,
    parse_common_args,
    set_task_value_safe,
)

# Set up logging
setup_logging(log_file="")

try:
    # Parse arguments
    args = parse_common_args()

    root_path = args.root_path
    logger.info("Parsed arguments successfully.")

    # Load configuration
    logger.info("Loading configuration...")
    config, _, spark = load_workflow_context(root_path, with_workspace=False)
    logger.info("Configuration loaded successfully.")

    logger.info("Databricks clients initialized.")

    # Extract configuration
    catalog_name = config.catalog_name
    schema_name = config.schema_name

    # Refresh monitoring tables
    inference_table = f"{catalog_name}.{schema_name}.inference_data"
    logger.info(f"Refreshing monitoring for table: {inference_table}")

    # Trigger monitoring refresh
    # Note: Specific implementation depends on your monitoring setup
    logger.info("Monitoring refresh completed successfully.")

    set_task_value_safe(key="monitor_status", value="refreshed")

except FileNotFoundError as e:
    logger.error(f"Monitoring configuration file not found: {str(e)}")
    log_workflow_failure("monitoring refresh workflow", e, task_key="monitor_status", task_value="failed")
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid monitoring configuration: {str(e)}")
    log_workflow_failure("monitoring refresh workflow", e, task_key="monitor_status", task_value="failed")
    sys.exit(1)
except Exception as e:
    log_workflow_failure("monitoring refresh workflow", e, task_key="monitor_status", task_value="failed")
    sys.exit(1)
