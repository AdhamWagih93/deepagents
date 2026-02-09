"""
Secrets Management for DeepAgents

This module provides secure storage and management of secrets (API keys, tokens, etc.)
using Fernet symmetric encryption. Secrets are stored encrypted in SQLite and can be
injected into tool execution environments.
"""

import os
import json
import sqlite3
import base64
import hashlib
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


@dataclass
class SecretMetadata:
    """Metadata about a secret (without the actual value)."""
    id: int
    name: str
    category: str
    description: str
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SecretsManager:
    """
    Manages encrypted secrets with Fernet encryption.

    Secrets are stored encrypted in SQLite and can be:
    - Created, read, updated, deleted via CRUD operations
    - Injected into environment variables for tool execution
    - Masked in logs to prevent accidental exposure
    """

    # Default categories for organizing secrets
    CATEGORIES = [
        "api_keys",      # API keys (OpenAI, etc.)
        "tokens",        # Access/auth tokens
        "passwords",     # Passwords
        "credentials",   # Username/password pairs
        "certificates",  # Certificates and keys
        "webhooks",      # Webhook URLs with secrets
        "general",       # General/uncategorized
    ]

    def __init__(self, db_path: str, encryption_key: Optional[str] = None):
        """
        Initialize the SecretsManager.

        Args:
            db_path: Path to the SQLite database
            encryption_key: Optional master password. If not provided, uses machine-specific key.
        """
        self.db_path = db_path
        self._fernet = self._init_encryption(encryption_key)
        self._cached_secrets: Dict[str, str] = {}  # Runtime cache for performance
        self._init_db()

    def _init_encryption(self, master_password: Optional[str] = None) -> Optional[Any]:
        """Initialize Fernet encryption with a derived key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return None

        # Use master password or generate machine-specific key
        if master_password:
            password = master_password.encode()
        else:
            # Generate a machine-specific key based on username and hostname
            machine_id = f"{os.getenv('USERNAME', 'user')}@{os.getenv('COMPUTERNAME', 'local')}"
            password = machine_id.encode()

        # Salt for key derivation (fixed for this installation)
        salt = b"DeepAgentsSecrets2024"

        # Derive a key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)

    def _init_db(self):
        """Ensure the secrets table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS secrets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                encrypted_value BLOB NOT NULL,
                category TEXT DEFAULT 'general',
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_secrets_name ON secrets(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_secrets_category ON secrets(category)")
        conn.commit()
        conn.close()

    def _encrypt(self, value: str) -> bytes:
        """Encrypt a value."""
        if not self._fernet:
            # Fallback: base64 encoding (not secure, but functional)
            return base64.b64encode(value.encode())
        return self._fernet.encrypt(value.encode())

    def _decrypt(self, encrypted_value: bytes) -> str:
        """Decrypt a value."""
        if not self._fernet:
            # Fallback: base64 decoding
            return base64.b64decode(encrypted_value).decode()
        return self._fernet.decrypt(encrypted_value).decode()

    def set_secret(
        self,
        name: str,
        value: str,
        category: str = "general",
        description: str = ""
    ) -> bool:
        """
        Create or update a secret.

        Args:
            name: Unique name for the secret (e.g., 'OPENAI_API_KEY')
            value: The secret value to encrypt and store
            category: Category for organization
            description: Optional description

        Returns:
            True if successful
        """
        if not name or not value:
            return False

        encrypted = self._encrypt(value)
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO secrets (name, encrypted_value, category, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    encrypted_value = excluded.encrypted_value,
                    category = excluded.category,
                    description = excluded.description,
                    updated_at = excluded.updated_at
            """, (name, encrypted, category, description, now, now))
            conn.commit()

            # Update cache
            self._cached_secrets[name] = value
            return True

        except Exception as e:
            print(f"Error saving secret: {e}")
            return False
        finally:
            conn.close()

    def get_secret(self, name: str) -> Optional[str]:
        """
        Retrieve a decrypted secret value.

        Args:
            name: Name of the secret

        Returns:
            The decrypted secret value, or None if not found
        """
        # Check cache first
        if name in self._cached_secrets:
            return self._cached_secrets[name]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT encrypted_value FROM secrets WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()

            if row:
                value = self._decrypt(row[0])
                self._cached_secrets[name] = value
                return value
            return None

        except Exception as e:
            print(f"Error retrieving secret: {e}")
            return None
        finally:
            conn.close()

    def delete_secret(self, name: str) -> bool:
        """
        Delete a secret.

        Args:
            name: Name of the secret to delete

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM secrets WHERE name = ?", (name,))
            conn.commit()
            deleted = cursor.rowcount > 0

            # Remove from cache
            self._cached_secrets.pop(name, None)
            return deleted

        except Exception as e:
            print(f"Error deleting secret: {e}")
            return False
        finally:
            conn.close()

    def list_secrets(self, category: Optional[str] = None) -> List[SecretMetadata]:
        """
        List all secrets (metadata only, not values).

        Args:
            category: Optional category filter

        Returns:
            List of SecretMetadata objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            if category:
                cursor.execute("""
                    SELECT id, name, category, description, created_at, updated_at
                    FROM secrets WHERE category = ?
                    ORDER BY name
                """, (category,))
            else:
                cursor.execute("""
                    SELECT id, name, category, description, created_at, updated_at
                    FROM secrets ORDER BY category, name
                """)

            secrets = []
            for row in cursor.fetchall():
                secrets.append(SecretMetadata(
                    id=row[0],
                    name=row[1],
                    category=row[2],
                    description=row[3] or "",
                    created_at=datetime.fromisoformat(row[4]) if row[4] else None,
                    updated_at=datetime.fromisoformat(row[5]) if row[5] else None,
                ))
            return secrets

        except Exception as e:
            print(f"Error listing secrets: {e}")
            return []
        finally:
            conn.close()

    def get_categories(self) -> List[str]:
        """Get all unique categories in use."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT DISTINCT category FROM secrets ORDER BY category")
            categories = [row[0] for row in cursor.fetchall()]
            # Merge with default categories
            all_categories = set(self.CATEGORIES) | set(categories)
            return sorted(list(all_categories))
        except Exception:
            return self.CATEGORIES
        finally:
            conn.close()

    @contextmanager
    def inject_into_env(self, secret_names: List[str]):
        """
        Context manager that temporarily injects secrets into environment variables.

        Args:
            secret_names: List of secret names to inject

        Usage:
            with secrets_manager.inject_into_env(['API_KEY', 'TOKEN']):
                # Secrets are available as environment variables
                run_tool()
            # Secrets are cleaned up
        """
        original_env = {}
        injected = []

        try:
            for name in secret_names:
                value = self.get_secret(name)
                if value:
                    # Store original value if exists
                    original_env[name] = os.environ.get(name)
                    os.environ[name] = value
                    injected.append(name)

            yield injected

        finally:
            # Restore original environment
            for name in injected:
                if original_env.get(name) is not None:
                    os.environ[name] = original_env[name]
                else:
                    os.environ.pop(name, None)

    def get_secret_values(self, secret_names: List[str]) -> Dict[str, str]:
        """
        Get multiple secret values at once.

        Args:
            secret_names: List of secret names

        Returns:
            Dict mapping secret names to their values
        """
        result = {}
        for name in secret_names:
            value = self.get_secret(name)
            if value:
                result[name] = value
        return result

    def mask_in_logs(self, text: str) -> str:
        """
        Replace secret values in text with masked versions.

        Args:
            text: Text that may contain secret values

        Returns:
            Text with secrets replaced by '***'
        """
        if not text:
            return text

        result = text

        # Get all secret values for masking
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT name, encrypted_value FROM secrets")
            for row in cursor.fetchall():
                name = row[0]
                try:
                    value = self._decrypt(row[1])
                    if value and len(value) > 3:  # Only mask non-trivial secrets
                        # Replace the full value
                        result = result.replace(value, f"[{name}:***]")
                        # Also mask partial matches (first 4 chars + ...)
                        if len(value) > 8:
                            partial = value[:4]
                            result = result.replace(partial, f"[{name[:6]}...]")
                except Exception:
                    pass

        except Exception:
            pass
        finally:
            conn.close()

        return result

    def export_secrets(self, password: str) -> Optional[str]:
        """
        Export all secrets as an encrypted JSON string.

        Args:
            password: Password to encrypt the export

        Returns:
            Encrypted JSON string, or None on error
        """
        # Create a new Fernet with the export password
        export_fernet = self._init_encryption(password)
        if not export_fernet:
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT name, encrypted_value, category, description
                FROM secrets
            """)

            secrets_data = []
            for row in cursor.fetchall():
                secrets_data.append({
                    "name": row[0],
                    "value": self._decrypt(row[1]),
                    "category": row[2],
                    "description": row[3] or "",
                })

            # Encrypt the export
            json_data = json.dumps(secrets_data)
            encrypted = export_fernet.encrypt(json_data.encode())
            return base64.b64encode(encrypted).decode()

        except Exception as e:
            print(f"Error exporting secrets: {e}")
            return None
        finally:
            conn.close()

    def import_secrets(self, encrypted_data: str, password: str, overwrite: bool = False) -> int:
        """
        Import secrets from an encrypted export.

        Args:
            encrypted_data: The encrypted export string
            password: Password to decrypt
            overwrite: If True, overwrite existing secrets

        Returns:
            Number of secrets imported
        """
        # Create a Fernet with the import password
        import_fernet = self._init_encryption(password)
        if not import_fernet:
            return 0

        try:
            # Decrypt the import
            encrypted = base64.b64decode(encrypted_data)
            decrypted = import_fernet.decrypt(encrypted)
            secrets_data = json.loads(decrypted.decode())

            count = 0
            for secret in secrets_data:
                name = secret.get("name")
                value = secret.get("value")
                category = secret.get("category", "general")
                description = secret.get("description", "")

                if not name or not value:
                    continue

                # Check if exists
                if not overwrite and self.get_secret(name):
                    continue

                if self.set_secret(name, value, category, description):
                    count += 1

            return count

        except Exception as e:
            print(f"Error importing secrets: {e}")
            return 0

    def clear_cache(self):
        """Clear the runtime secrets cache."""
        self._cached_secrets.clear()

    def validate_secret(self, name: str) -> Dict[str, Any]:
        """
        Validate that a secret exists and is accessible.

        Returns:
            Dict with 'valid' bool and optional 'error' message
        """
        try:
            value = self.get_secret(name)
            if value:
                return {"valid": True, "length": len(value)}
            return {"valid": False, "error": f"Secret '{name}' not found"}
        except Exception as e:
            return {"valid": False, "error": str(e)}


class SecretInjector:
    """
    Helper class for injecting secrets into tool execution.

    This provides a cleaner interface for tools that need secrets.
    """

    def __init__(self, secrets_manager: SecretsManager):
        self.secrets_manager = secrets_manager
        self._active_secrets: Set[str] = set()

    def require(self, *secret_names: str) -> Dict[str, str]:
        """
        Get required secrets, raising an error if any are missing.

        Args:
            *secret_names: Names of required secrets

        Returns:
            Dict of secret name -> value

        Raises:
            ValueError if any secrets are missing
        """
        missing = []
        secrets = {}

        for name in secret_names:
            value = self.secrets_manager.get_secret(name)
            if value:
                secrets[name] = value
                self._active_secrets.add(name)
            else:
                missing.append(name)

        if missing:
            raise ValueError(f"Missing required secrets: {', '.join(missing)}")

        return secrets

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an optional secret.

        Args:
            name: Secret name
            default: Default value if not found

        Returns:
            Secret value or default
        """
        value = self.secrets_manager.get_secret(name)
        if value:
            self._active_secrets.add(name)
            return value
        return default

    def mask_output(self, text: str) -> str:
        """Mask any active secrets in the output text."""
        return self.secrets_manager.mask_in_logs(text)

    def get_active_secrets(self) -> Set[str]:
        """Get names of secrets that were accessed."""
        return self._active_secrets.copy()

    def reset(self):
        """Reset the active secrets tracking."""
        self._active_secrets.clear()
