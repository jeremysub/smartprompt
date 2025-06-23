"""
LLM Helper Module

A modular package for handling LLM interactions with tool calling capabilities.
"""

# Core classes
from .model_client import ModelClient
from .prompt import Prompt, PromptSettings
from .tool_runner import ToolRunner, ToolCallRequest
from .prompt_loader import PromptLoader
from .storage_providers import StorageProviderBase, MemoryProvider, BlobStorageProvider, FileSystemProvider

# Tool utilities
from .tools.datetime_tools import get_datetime_tools, GetCurrentTime, FormatDateTime, AddTime, GetTimeDifference, IsWeekend, GetDayOfWeek

# Convenience imports for common use cases
__all__ = [
    'ModelClient',
    'Prompt',
    'PromptSettings',
    'ToolRunner',
    'ToolCallRequest',
    'PromptLoader',
    'StorageProviderBase',
    'MemoryProvider',
    'BlobStorageProvider',
    'FileSystemProvider',
    'get_datetime_tools',
    # Datetime tools
    'GetCurrentTime',
    'FormatDateTime',
    'AddTime',
    'GetTimeDifference',
    'IsWeekend',
    'GetDayOfWeek'
]
