"""Secrets management utilities for production deployments."""

import os
from typing import Any, Optional

from loguru import logger


class SecretsManager:
    """
    Manages secure handling of secrets and sensitive configuration.
    
    Supports both environment variables and Databricks Secrets API.
    """

    def __init__(self, use_databricks_secrets: bool = False):
        """
        Initialize SecretsManager.

        Args:
            use_databricks_secrets: If True, use Databricks Secrets API.
                                   Otherwise use environment variables.
        """
        self.use_databricks_secrets = use_databricks_secrets

        if use_databricks_secrets:
            try:
                from databricks.sdk import WorkspaceClient

                self.workspace_client = WorkspaceClient()
                logger.info("Initialized Databricks Secrets API client")
            except ImportError:
                logger.warning("databricks-sdk not available. Falling back to environment variables.")
                self.use_databricks_secrets = False
                self.workspace_client = None
        else:
            self.workspace_client = None

    def get_secret(self, key: str, scope: Optional[str] = None, required: bool = True) -> Optional[str]:
        """
        Retrieve a secret value.

        Args:
            key: Secret key/name
            scope: Scope name (for Databricks Secrets) or prefix (for env vars)
            required: If True, raises error if secret not found

        Returns:
            Secret value or None if not found and not required

        Raises:
            KeyError: If secret not found and required=True
        """
        if self.use_databricks_secrets and self.workspace_client:
            return self._get_databricks_secret(key, scope, required)
        else:
            return self._get_env_secret(key, scope, required)

    def _get_env_secret(self, key: str, prefix: Optional[str] = None, required: bool = True) -> Optional[str]:
        """
        Get secret from environment variable.

        Args:
            key: Environment variable name
            prefix: Optional prefix to add to key (e.g., "PROD_" -> looks for "PROD_KEY")
            required: If True, raises KeyError if not found

        Returns:
            Environment variable value or None
        """
        env_key = f"{prefix}_{key}" if prefix else key
        value = os.environ.get(env_key)

        if value is None and required:
            raise KeyError(f"Required environment variable '{env_key}' not set")

        if value is None:
            logger.warning(f"Optional environment variable '{env_key}' not set")
            return None

        logger.debug(f"Retrieved environment variable: {env_key}")
        return value

    def _get_databricks_secret(
        self, key: str, scope: str = "default", required: bool = True
    ) -> Optional[str]:
        """
        Get secret from Databricks Secrets API.

        Args:
            key: Secret key
            scope: Secret scope (default: "default")
            required: If True, raises error if not found

        Returns:
            Secret value or None
        """
        if not self.workspace_client:
            raise RuntimeError("Databricks Secrets API not initialized")

        try:
            secret_value = self.workspace_client.secrets.get_secret(scope=scope, key=key)
            logger.debug(f"Retrieved secret from Databricks: {scope}/{key}")
            return secret_value.key_value_string
        except Exception as e:
            if required:
                logger.error(f"Failed to retrieve required secret {scope}/{key}: {str(e)}")
                raise KeyError(f"Secret '{scope}/{key}' not found in Databricks Secrets") from e
            else:
                logger.warning(f"Optional secret {scope}/{key} not found: {str(e)}")
                return None

    def validate_required_secrets(self, required_keys: dict[str, str]) -> bool:
        """
        Validate that all required secrets are available.

        Args:
            required_keys: Dict mapping key names to descriptions
                          Example: {"DB_PASSWORD": "Database password", "API_KEY": "API token"}

        Returns:
            bool: True if all secrets found, False otherwise
        """
        missing_secrets = []

        for key, description in required_keys.items():
            try:
                self.get_secret(key, required=True)
                logger.info(f"✅ Secret '{key}' found ({description})")
            except KeyError:
                logger.error(f"❌ Missing required secret: '{key}' ({description})")
                missing_secrets.append(key)

        if missing_secrets:
            logger.error(f"Missing {len(missing_secrets)} required secrets: {missing_secrets}")
            return False

        logger.info(f"✅ All {len(required_keys)} required secrets validated")
        return True

    def get_production_secrets(self) -> dict[str, str]:
        """
        Retrieve all required production secrets.

        Returns:
            Dict mapping secret names to values

        Raises:
            KeyError: If any required secret not found
        """
        required_secrets = {
            "DATABRICKS_HOST": "Databricks workspace URL",
            "DATABRICKS_TOKEN": "Databricks API token",
            "DATABRICKS_SERVICE_PRINCIPAL_NAME": "Service principal for production runs",
        }

        # Validate all secrets exist
        if not self.validate_required_secrets(required_secrets):
            raise KeyError("Cannot start: missing required production secrets")

        # Retrieve all secrets
        secrets = {}
        for key in required_secrets:
            secrets[key] = self.get_secret(key, required=True)

        return secrets

    @staticmethod
    def create_databricks_secret(scope: str, key: str, value: str) -> bool:
        """
        Create a secret in Databricks Secrets.
        
        Requires appropriate permissions and Databricks SDK.

        Args:
            scope: Secret scope
            key: Secret key
            value: Secret value

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()
            workspace_client.secrets.put_secret(scope=scope, key=key, string_value=value)
            logger.info(f"Created secret in Databricks: {scope}/{key}")
            return True
        except Exception as e:
            logger.error(f"Failed to create secret {scope}/{key}: {str(e)}")
            return False

    @staticmethod
    def create_databricks_secret_scope(scope: str) -> bool:
        """
        Create a secret scope in Databricks.
        
        Requires appropriate permissions and Databricks SDK.

        Args:
            scope: Scope name to create

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()
            workspace_client.secrets.create_scope(scope=scope)
            logger.info(f"Created secret scope in Databricks: {scope}")
            return True
        except Exception as e:
            logger.error(f"Failed to create secret scope '{scope}': {str(e)}")
            return False


# Environment variables required for production
REQUIRED_PRODUCTION_ENV_VARS = {
    "DATABRICKS_HOST": "Databricks workspace URL (e.g., https://workspace.cloud.databricks.com)",
    "DATABRICKS_TOKEN": "Databricks personal access token or service principal token",
    "DATABRICKS_SERVICE_PRINCIPAL_NAME": "Service principal name/ID for production job execution",
    "GIT_SHA": "Git commit SHA for deployment tracking",
    "ENVIRONMENT": "Deployment environment (dev, staging, prod)",
}

OPTIONAL_PRODUCTION_ENV_VARS = {
    "ALERT_EMAIL": "Email for failure notifications",
    "LOG_LEVEL": "Logging level (DEBUG, INFO, WARNING, ERROR)",
}


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.

    Returns:
        bool: True if all required vars set, False otherwise
    """
    missing_vars = []

    for var, description in REQUIRED_PRODUCTION_ENV_VARS.items():
        if not os.environ.get(var):
            logger.error(f"❌ Missing required environment variable: {var}")
            logger.error(f"   Description: {description}")
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"Cannot proceed: missing {len(missing_vars)} required environment variables")
        return False

    logger.info(f"✅ All {len(REQUIRED_PRODUCTION_ENV_VARS)} required environment variables set")

    # Log optional vars status
    for var, description in OPTIONAL_PRODUCTION_ENV_VARS.items():
        if os.environ.get(var):
            logger.info(f"✅ Optional environment variable set: {var}")
        else:
            logger.debug(f"ℹ️ Optional environment variable not set: {var}")

    return True
