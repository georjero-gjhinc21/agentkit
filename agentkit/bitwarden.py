"""
agentkit.bitwarden
==================

Bitwarden integration for secure credential management.

Store API keys, passwords, and secrets in Bitwarden vault instead of .env files
or hardcoded strings. Retrieve them programmatically when needed.

SETUP:
    1. Install Bitwarden CLI: https://bitwarden.com/help/cli/
    2. Login: bw login
    3. Unlock: bw unlock (save the session key)
    4. Export session: export BW_SESSION="your-session-key"

USAGE:
    from agentkit.bitwarden import BitwardenSecrets

    secrets = BitwardenSecrets()

    # Get API key by item name
    api_key = secrets.get("OPENAI_API_KEY")

    # Get field from specific item
    anthropic_key = secrets.get_field("Anthropic API", "api_key")

    # Get all fields from an item
    item = secrets.get_item("AWS Credentials")
    access_key = item.get("access_key_id")
    secret_key = item.get("secret_access_key")

WHY BITWARDEN:
    ✓ Encrypted vault storage
    ✓ Cross-device sync
    ✓ Shared team access (for organizations)
    ✓ Audit logs
    ✓ Better than .env files sitting in your project
    ✓ Automatic rotation/updates propagate everywhere
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from .errors import ConfigurationError


class BitwardenSecrets:
    """Retrieve secrets from Bitwarden vault.

    Requires Bitwarden CLI to be installed and authenticated.
    Session key must be available via BW_SESSION environment variable.
    """

    def __init__(
        self,
        session_key: str | None = None,
        cache: bool = True,
    ):
        """Initialize Bitwarden secrets manager.

        Args:
            session_key: Bitwarden session key. If None, reads from BW_SESSION env var.
            cache: Cache vault items in memory to reduce CLI calls (default: True).
        """
        self.session_key = session_key or os.getenv("BW_SESSION")
        if not self.session_key:
            raise ConfigurationError(
                "Bitwarden session key not found. Set BW_SESSION environment variable or pass session_key.\n"
                "Get session key by running: bw unlock"
            )

        # Verify CLI is available
        if not self._cli_available():
            raise ConfigurationError(
                "Bitwarden CLI not found. Install from: https://bitwarden.com/help/cli/"
            )

        self._cache_enabled = cache
        self._vault_cache: dict[str, Any] | None = None

    @staticmethod
    def _cli_available() -> bool:
        """Check if Bitwarden CLI is installed."""
        try:
            subprocess.run(["bw", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _run_bw_command(self, *args: str) -> str:
        """Run a Bitwarden CLI command.

        Args:
            *args: Command arguments to pass to bw.

        Returns:
            Command output as string.

        Raises:
            ConfigurationError: If command fails.
        """
        env = os.environ.copy()
        env["BW_SESSION"] = self.session_key

        try:
            result = subprocess.run(
                ["bw", *args],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise ConfigurationError(
                f"Bitwarden CLI error: {e.stderr.strip()}\n"
                f"Command: bw {' '.join(args)}"
            ) from e

    def _get_vault_items(self) -> list[dict[str, Any]]:
        """Get all items from vault.

        Returns cached items if caching is enabled.
        """
        if self._cache_enabled and self._vault_cache is not None:
            return self._vault_cache

        output = self._run_bw_command("list", "items")
        items = json.loads(output)

        if self._cache_enabled:
            self._vault_cache = items

        return items

    def get(self, name: str, default: str | None = None) -> str | None:
        """Get a secret by item name.

        Searches for an item with matching name and returns its password field.
        If the item has a 'value' or 'api_key' field, returns that instead.

        Args:
            name: Name of the Bitwarden item.
            default: Default value if item not found.

        Returns:
            Secret value or default.

        Raises:
            ConfigurationError: If item not found and no default provided.
        """
        items = self._get_vault_items()

        # Find item by name
        for item in items:
            if item.get("name") == name:
                # Try common field names
                if item.get("login", {}).get("password"):
                    return item["login"]["password"]

                # Check fields for api_key or value
                for field in item.get("fields", []):
                    if field.get("name") in ("api_key", "value", "secret"):
                        return field.get("value")

                # Fallback to notes
                if item.get("notes"):
                    return item["notes"]

        if default is not None:
            return default

        raise ConfigurationError(
            f"Secret '{name}' not found in Bitwarden vault.\n"
            f"Create an item named '{name}' with a password or api_key field."
        )

    def get_field(self, item_name: str, field_name: str, default: str | None = None) -> str | None:
        """Get a specific field from a Bitwarden item.

        Args:
            item_name: Name of the Bitwarden item.
            field_name: Name of the field to retrieve.
            default: Default value if field not found.

        Returns:
            Field value or default.

        Raises:
            ConfigurationError: If item or field not found and no default provided.
        """
        items = self._get_vault_items()

        for item in items:
            if item.get("name") == item_name:
                # Check custom fields
                for field in item.get("fields", []):
                    if field.get("name") == field_name:
                        return field.get("value")

                # Check login fields
                if field_name == "username":
                    return item.get("login", {}).get("username")
                if field_name == "password":
                    return item.get("login", {}).get("password")

                # Not found in this item
                if default is not None:
                    return default

                raise ConfigurationError(
                    f"Field '{field_name}' not found in item '{item_name}'.\n"
                    f"Available fields: {[f['name'] for f in item.get('fields', [])]}"
                )

        if default is not None:
            return default

        raise ConfigurationError(f"Item '{item_name}' not found in Bitwarden vault.")

    def get_item(self, item_name: str) -> dict[str, str]:
        """Get all fields from a Bitwarden item as a dictionary.

        Args:
            item_name: Name of the Bitwarden item.

        Returns:
            Dictionary of field_name -> value.

        Raises:
            ConfigurationError: If item not found.
        """
        items = self._get_vault_items()

        for item in items:
            if item.get("name") == item_name:
                result: dict[str, str] = {}

                # Add login fields
                if item.get("login"):
                    if item["login"].get("username"):
                        result["username"] = item["login"]["username"]
                    if item["login"].get("password"):
                        result["password"] = item["login"]["password"]

                # Add custom fields
                for field in item.get("fields", []):
                    result[field["name"]] = field.get("value", "")

                return result

        raise ConfigurationError(f"Item '{item_name}' not found in Bitwarden vault.")

    def clear_cache(self) -> None:
        """Clear the vault items cache."""
        self._vault_cache = None

    def list_items(self) -> list[str]:
        """List all item names in the vault.

        Returns:
            List of item names.
        """
        items = self._get_vault_items()
        return [item["name"] for item in items if item.get("name")]


# Convenience function for quick access
def get_secret(name: str, default: str | None = None) -> str | None:
    """Quick access to get a secret from Bitwarden.

    Creates a BitwardenSecrets instance and retrieves the secret.

    Args:
        name: Name of the Bitwarden item.
        default: Default value if not found.

    Returns:
        Secret value or default.
    """
    secrets = BitwardenSecrets()
    return secrets.get(name, default)
