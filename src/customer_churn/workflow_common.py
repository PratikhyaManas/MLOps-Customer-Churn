"""Shared helpers for Databricks workflow entrypoints."""

import argparse
import traceback
from typing import Any, Optional, Tuple

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from customer_churn.utils import Config, load_config


def parse_common_args(require_git_metadata: bool = False) -> argparse.Namespace:
    """Parse common workflow arguments used across all jobs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", action="store", default=None, type=str, required=True)

    if require_git_metadata:
        parser.add_argument("--git_sha", action="store", default=None, type=str, required=True)
        parser.add_argument("--job_run_id", action="store", default=None, type=str, required=True)

    return parser.parse_args()


def load_workflow_context(
    root_path: str,
    with_workspace: bool = True,
    with_spark: bool = True,
) -> Tuple[Config, Optional[WorkspaceClient], Optional[SparkSession]]:
    """Load workflow config and optional Databricks clients."""
    config_path = f"{root_path}/project_config.yml"
    config = load_config(config_path)

    workspace = WorkspaceClient() if with_workspace else None
    spark = SparkSession.builder.getOrCreate() if with_spark else None
    return config, workspace, spark


def set_task_value_safe(key: str, value: Any) -> None:
    """Set Databricks task value when running in Jobs context."""
    try:
        dbutils.jobs.taskValues.set(key=key, value=value)
    except NameError:
        logger.warning("dbutils is unavailable outside Databricks Jobs; skipping task value update")


def get_task_value_safe(task_key: str, key: str, default: Any = None) -> Any:
    """Get Databricks task value when running in Jobs context."""
    try:
        return dbutils.jobs.taskValues.get(taskKey=task_key, key=key)
    except NameError:
        logger.warning("dbutils is unavailable outside Databricks Jobs; returning default value")
        return default


def log_workflow_failure(workflow_name: str, exc: Exception, task_key: Optional[str] = None, task_value: Any = None) -> None:
    """Log a workflow exception and optionally set a fallback task value."""
    logger.error(f"Unexpected error in {workflow_name}: {type(exc).__name__}: {str(exc)}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")

    if task_key is not None:
        set_task_value_safe(task_key, task_value)
