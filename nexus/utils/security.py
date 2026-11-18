"""
Security utilities for NEXUS.

Provides token redaction, path sanitization, and security-related functionality.
"""

import os
import re
from pathlib import Path
from typing import Any, Optional


class SecurityManager:
    """Security utilities for NEXUS."""
    
    # Sensitive environment variable names
    SENSITIVE_ENV_VARS = {
        'HF_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGINGFACEHUB_API_TOKEN',
        'KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN',
        'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY',
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
        'AZURE_STORAGE_KEY', 'GITHUB_TOKEN', 'GITLAB_TOKEN',
    }
    
    # Token patterns to redact
    TOKEN_PATTERNS = [
        re.compile(r'\bhf_[A-Za-z0-9]{20,}\b'),  # Hugging Face tokens
        re.compile(r'\bsk-[A-Za-z0-9]{20,}\b'),  # OpenAI-style keys
        re.compile(r'\bAKIA[A-Z0-9]{16}\b'),  # AWS access keys
        re.compile(r'\b[gh]_[A-Za-z0-9]{20,}\b'),  # GitHub tokens
        re.compile(r'\bglpat-[A-Za-z0-9]{20,}\b'),  # GitLab tokens
    ]
    
    def redact_sensitive(self, text: str) -> str:
        """Redact sensitive information from text."""
        if not isinstance(text, str):
            return str(text)
        
        redacted = text
        
        # Redact token patterns
        for pattern in self.TOKEN_PATTERNS:
            redacted = pattern.sub('[REDACTED]', redacted)
        
        # Redact environment variable assignments
        for env_var in self.SENSITIVE_ENV_VARS:
            # Match patterns like VAR=value or VAR="value"
            redacted = re.sub(
                rf'{env_var}\s*=\s*[^\s"\']+',
                f'{env_var}=[REDACTED]',
                redacted,
                flags=re.IGNORECASE,
            )
            redacted = re.sub(
                rf'{env_var}\s*=\s*"[^"]*"',
                f'{env_var}="[REDACTED]"',
                redacted,
                flags=re.IGNORECASE,
            )
        
        return redacted
    
    def sanitize_path(self, path: str) -> str:
        """Sanitize a path to prevent directory traversal attacks."""
        if not isinstance(path, str):
            return str(path)
        
        # Normalize the path
        normalized = os.path.normpath(path)
        
        # Remove any leading slashes to make it relative
        normalized = normalized.lstrip('/')
        
        # Remove any remaining .. components
        parts = []
        for part in normalized.split(os.sep):
            if part == '..':
                continue  # Skip parent directory references
            elif part == '.':
                continue  # Skip current directory references
            else:
                parts.append(part)
        
        return os.sep.join(parts) if parts else '.'
    
    def is_sensitive_env_var(self, var_name: str) -> bool:
        """Check if an environment variable is sensitive."""
        return var_name.upper() in self.SENSITIVE_ENV_VARS
    
    def get_safe_env(self) -> dict:
        """Get environment variables with sensitive ones redacted."""
        safe_env = {}
        
        for key, value in os.environ.items():
            if self.is_sensitive_env_var(key):
                safe_env[key] = '[REDACTED]'
            else:
                safe_env[key] = value
        
        return safe_env
    
    def mask_token(self, token: str) -> str:
        """Mask a token for safe display."""
        if not token or len(token) < 8:
            return '[REDACTED]'
        
        # Show first 4 and last 4 characters
        return f"{token[:4]}...{token[-4:]}"
    
    def validate_no_secrets_in_text(self, text: str) -> list:
        """Check text for potential secrets and return list of findings."""
        findings = []
        
        if not isinstance(text, str):
            return findings
        
        # Check for token patterns
        for i, pattern in enumerate(self.TOKEN_PATTERNS):
            matches = pattern.findall(text)
            if matches:
                findings.append({
                    "type": "token_pattern",
                    "pattern_index": i,
                    "matches_count": len(matches),
                })
        
        # Check for environment variable assignments
        for env_var in self.SENSITIVE_ENV_VARS:
            if re.search(rf'{env_var}\s*=', text, flags=re.IGNORECASE):
                findings.append({
                    "type": "env_var_assignment",
                    "variable": env_var,
                })
        
        return findings