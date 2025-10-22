"""
Utilities for safe config.json file operations.
Provides thread-safe JSON reading and writing with proper error handling.
"""

import os
import json
import tempfile
import shutil
from threading import Lock

# Global lock for config.json operations
_config_lock = Lock()

def load_config(config_path):
    """
    Safely load config.json with error handling.

    Args:
        config_path: Path to config.json file

    Returns:
        Dictionary containing config data, or empty dict if file doesn't exist
    """
    with _config_lock:
        if not os.path.exists(config_path):
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config from {config_path}: {e}")
            # Try to load backup if it exists
            backup_path = config_path + ".backup"
            if os.path.exists(backup_path):
                print(f"Attempting to restore from backup: {backup_path}")
                try:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError) as e2:
                    print(f"Error loading backup: {e2}")
            return {}


def save_config(config_path, config_data):
    """
    Safely save config.json with atomic write and backup.

    Args:
        config_path: Path to config.json file
        config_data: Dictionary containing config data to save

    Returns:
        True if successful, False otherwise
    """
    with _config_lock:
        try:
            # Create backup of existing file
            if os.path.exists(config_path):
                backup_path = config_path + ".backup"
                try:
                    shutil.copy2(config_path, backup_path)
                except IOError:
                    pass  # Backup creation is optional

            # Write to temporary file first (atomic operation)
            config_dir = os.path.dirname(config_path)
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=config_dir,
                delete=False,
                suffix='.tmp'
            ) as tmp_file:
                json.dump(config_data, tmp_file, indent=2, ensure_ascii=False)
                tmp_path = tmp_file.name

            # Replace original file with temporary file (atomic on most systems)
            shutil.move(tmp_path, config_path)
            return True

        except (IOError, OSError) as e:
            print(f"Error saving config to {config_path}: {e}")
            # Clean up temporary file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            return False


def update_config(config_path, updates):
    """
    Update specific keys in config.json safely.

    Args:
        config_path: Path to config.json file
        updates: Dictionary containing keys to update

    Returns:
        True if successful, False otherwise
    """
    config = load_config(config_path)
    config.update(updates)
    return save_config(config_path, config)


def update_nested_config(config_path, section, updates):
    """
    Update nested section in config.json safely.

    Args:
        config_path: Path to config.json file
        section: Section key (e.g., "realtime")
        updates: Dictionary containing keys to update in that section

    Returns:
        True if successful, False otherwise
    """
    config = load_config(config_path)
    if section not in config:
        config[section] = {}
    config[section].update(updates)
    return save_config(config_path, config)
