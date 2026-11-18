"""
File utility functions for NEXUS.

Provides safe file operations including atomic writes, safe path joining,
and workspace layout management.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileManager:
    """Safe file operations for NEXUS."""
    
    @staticmethod
    def atomic_write(file_path: str, content: str, mode: str = 'w') -> None:
        """Write content to file atomically using temp file + rename."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file in same directory
        fd, temp_path = tempfile.mkstemp(dir=str(file_path.parent), suffix='.tmp')
        try:
            with os.fdopen(fd, mode) as f:
                f.write(content)
            os.replace(temp_path, str(file_path))
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
    
    @staticmethod
    def atomic_write_json(file_path: str, data: Any, indent: int = 2) -> None:
        """Write JSON data to file atomically."""
        FileManager.atomic_write(file_path, json.dumps(data, indent=indent), mode='w')
    
    @staticmethod
    def safe_join(base_path: str, *paths: str) -> str:
        """Safely join paths, preventing directory traversal."""
        base = Path(base_path).resolve()
        joined = base
        
        for p in paths:
            joined = joined / p
        
        # Resolve and check that the result is still under base
        resolved = joined.resolve()
        
        # Check if resolved path is under base
        try:
            resolved.relative_to(base)
        except ValueError:
            # Path escapes base directory
            raise ValueError(f"Path traversal detected: {joined}")
        
        return str(resolved)
    
    @staticmethod
    def read_json(file_path: str) -> Optional[Dict[str, Any]]:
        """Read JSON file safely."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """Ensure a directory exists, creating it if necessary."""
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> List[Path]:
        """List files in directory matching pattern."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        return list(dir_path.glob(pattern))
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes."""
        return Path(file_path).stat().st_size