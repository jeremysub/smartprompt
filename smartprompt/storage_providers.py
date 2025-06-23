from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import json
from .blob_storage import BlobStorageClient
from .prompt import PromptSettings

class StorageProviderBase(ABC):
    """Abstract base class for prompt storage providers."""
    
    @abstractmethod
    def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the prompt directory."""
        pass
    
    @abstractmethod
    def download_file(self, file_name: str) -> Optional[bytes]:
        """Download a file from the prompt directory."""
        pass
    
    @abstractmethod
    def file_exists(self, file_name: str) -> bool:
        """Check if a file exists in the prompt directory."""
        pass


class MemoryProvider(StorageProviderBase):
    """In-memory storage provider for prompts. Default provider for development and testing."""
    
    def __init__(self, system_prompt: str, prompt_settings: PromptSettings, user_prompt: str = ""):
        """
        Initialize the memory provider with prompt content.
        
        Args:
            system_prompt: The system prompt content
            user_prompt: The user prompt content (optional)
            prompt_settings: Prompt settings. If None, defaults to OpenAI GPT-4
        """
        self._system_prompt = system_prompt
        self._prompt_settings = prompt_settings
        self._user_prompt = user_prompt
    
    def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the prompt directory."""
        files = [
            {
                "name": "settings.json",
                "filename": "settings.json",
                "size": len(json.dumps(self._prompt_settings.__dict__)),
                "created_at": 0,
                "modified_at": 0,
                "content_type": "application/json",
            },
            {
                "name": "system.md",
                "filename": "system.md",
                "size": len(self._system_prompt.encode("utf-8")),
                "created_at": 0,
                "modified_at": 0,
                "content_type": "text/markdown",
            }
        ]
        
        if self._user_prompt:
            files.append({
                "name": "user.md",
                "filename": "user.md",
                "size": len(self._user_prompt.encode("utf-8")),
                "created_at": 0,
                "modified_at": 0,
                "content_type": "text/markdown",
            })
        
        return files
    
    def download_file(self, file_name: str) -> Optional[bytes]:
        """Download a file from the prompt directory."""
        if file_name == "settings.json":
            return json.dumps(self._prompt_settings.__dict__).encode("utf-8")
        elif file_name == "system.md":
            return self._system_prompt.encode("utf-8")
        elif file_name == "user.md":
            return self._user_prompt.encode("utf-8") if self._user_prompt else None
        return None
    
    def file_exists(self, file_name: str) -> bool:
        """Check if a file exists in the prompt directory."""
        if file_name == "user.md":
            return bool(self._user_prompt)
        return file_name in ["settings.json", "system.md"]


class BlobStorageProvider(StorageProviderBase):
    """Azure Blob Storage provider for prompts."""
    
    _container_name = "prompts"
    
    def __init__(self, prompt_name: str):
        """
        Initialize the blob storage provider.
        
        Args:
            prompt_name: Name of the prompt directory
        """
        self._blob_client = BlobStorageClient(self._container_name, prompt_name)
    
    def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the prompt directory."""
        return self._blob_client.list_files()
    
    def download_file(self, file_name: str) -> Optional[bytes]:
        """Download a file from the prompt directory."""
        return self._blob_client.download_file(file_name)
    
    def file_exists(self, file_name: str) -> bool:
        """Check if a file exists in the prompt directory."""
        return self._blob_client.file_exists(file_name)


class FileSystemProvider(StorageProviderBase):
    """Local file system provider for prompts."""
    
    def __init__(self, prompt_name: str, base_path: str = "prompts"):
        """
        Initialize the file system provider.
        
        Args:
            prompt_name: Name of the prompt directory
            base_path: Base path where prompts are stored
        """
        self._prompt_path = os.path.join(base_path, prompt_name)
        
        # Check if prompt directory exists
        if not os.path.exists(self._prompt_path):
            raise ValueError(f"Prompts directory '{self._prompt_path}' does not exist")
        
        if not os.path.isdir(self._prompt_path):
            raise ValueError(f"'{self._prompt_path}' is not a directory")
    
    def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the prompt directory."""
        try:
            files = []
            for filename in os.listdir(self._prompt_path):
                file_path = os.path.join(self._prompt_path, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files.append({
                        "name": filename,
                        "filename": filename,
                        "size": stat.st_size,
                        "created_at": stat.st_ctime,
                        "modified_at": stat.st_mtime,
                        "content_type": self._get_content_type(filename),
                    })
            return files
        except Exception as e:
            return []
    
    def download_file(self, file_name: str) -> Optional[bytes]:
        """Download a file from the prompt directory."""
        try:
            file_path = os.path.join(self._prompt_path, file_name)
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception:
            return None
    
    def file_exists(self, file_name: str) -> bool:
        """Check if a file exists in the prompt directory."""
        file_path = os.path.join(self._prompt_path, file_name)
        return os.path.isfile(file_path)
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type based on file extension."""
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            '.json': 'application/json',
            '.md': 'text/markdown',
            '.txt': 'text/plain',
            '.yaml': 'text/yaml',
            '.yml': 'text/yaml',
        }
        return content_types.get(ext, 'application/octet-stream') 