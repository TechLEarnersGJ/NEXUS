"""
Hardware information utilities for NEXUS.

Provides stdlib-only CPU, RAM, and disk probing without ML dependencies.
"""

import os
import platform
from typing import Dict, Any, Optional


class HardwareInfo:
    """Hardware information for NEXUS."""
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU information."""
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1
        
        return {
            "count": cpu_count,
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get memory information."""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            mem_total = 0
            mem_available = 0
            
            for line in meminfo.splitlines():
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1]) * 1024  # Convert kB to bytes
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) * 1024
            
            return {
                "total_bytes": mem_total,
                "available_bytes": mem_available,
                "total_gb": mem_total / (1024 ** 3) if mem_total > 0 else 0,
                "available_gb": mem_available / (1024 ** 3) if mem_available > 0 else 0,
            }
        except Exception:
            return {
                "total_bytes": 0,
                "available_bytes": 0,
                "total_gb": 0,
                "available_gb": 0,
            }
    
    @staticmethod
    def get_disk_info(path: str = "/") -> Dict[str, Any]:
        """Get disk information for a path."""
        try:
            stat = os.statvfs(path)
            
            total = stat.f_blocks * stat.f_frsize
            available = stat.f_bavail * stat.f_frsize
            used = total - available
            
            return {
                "total_bytes": total,
                "used_bytes": used,
                "available_bytes": available,
                "total_gb": total / (1024 ** 3),
                "used_gb": used / (1024 ** 3),
                "available_gb": available / (1024 ** 3),
                "usage_percent": (used / total * 100) if total > 0 else 0,
            }
        except Exception:
            return {
                "total_bytes": 0,
                "used_bytes": 0,
                "available_bytes": 0,
                "total_gb": 0,
                "used_gb": 0,
                "available_gb": 0,
                "usage_percent": 0,
            }
    
    @staticmethod
    def get_all_info() -> Dict[str, Any]:
        """Get all hardware information."""
        return {
            "cpu": HardwareInfo.get_cpu_info(),
            "memory": HardwareInfo.get_memory_info(),
            "disk": HardwareInfo.get_disk_info(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }
    
    @staticmethod
    def has_gpu() -> bool:
        """Check if GPU is available (stdlib-only check)."""
        # Check for NVIDIA GPU via nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and "GPU" in result.stdout
        except Exception:
            return False
    
    @staticmethod
    def get_gpu_info() -> Optional[Dict[str, Any]]:
        """Get GPU information if available."""
        if not HardwareInfo.has_gpu():
            return None
        
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                gpus = []
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpus.append({
                            "name": parts[0],
                            "memory_total": parts[1],
                            "driver_version": parts[2],
                        })
                return {"gpus": gpus, "count": len(gpus)}
        except Exception:
            pass
        
        return None