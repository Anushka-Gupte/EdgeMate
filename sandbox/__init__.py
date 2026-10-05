"""Sandbox execution package for EdgeMate.
Provides isolated execution for untrusted candidate code.
"""

from .executor import SandboxExecutor, ExecutionResult, run_sandboxed

__all__ = ["SandboxExecutor", "ExecutionResult", "run_sandboxed"]
