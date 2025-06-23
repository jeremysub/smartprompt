import pytest
import json
import os
import tempfile
import shutil
from typing import Type, Any
from pydantic import BaseModel

from smartprompt.prompt_loader import PromptLoader
from smartprompt.storage_providers import MemoryProvider, FileSystemProvider, StorageProviderBase
from smartprompt.prompt import Prompt, PromptSettings


@pytest.mark.no_collect
class TestResponseSchema(BaseModel):
    """Test response schema for testing."""
    message: str
    status: str


class TestPromptLoader:
    """Test cases for PromptLoader class."""

    def test_memory_provider_basic_loading(self):
        """Test basic prompt loading with MemoryProvider."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        system_prompt = "You are a helpful assistant."
        user_prompt = "Hello, how are you?"
        
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_settings=settings
        )
        
        # Act
        loader = PromptLoader(provider, "test_prompt")
        prompt = loader.get_prompt()
        
        # Assert
        assert prompt.system_prompt == system_prompt
        assert prompt.user_prompt == user_prompt
        assert prompt.settings.provider == "openai"
        assert prompt.settings.model == "gpt-4"
        assert prompt.response_schema is None

    def test_memory_provider_without_user_prompt(self):
        """Test MemoryProvider without user prompt."""
        # Arrange
        settings = PromptSettings(provider="azure", model="gpt-4")
        system_prompt = "You are a coding assistant."
        
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt=system_prompt,
            prompt_settings=settings
        )
        
        # Act
        loader = PromptLoader(provider, "test_prompt")
        prompt = loader.get_prompt()
        
        # Assert
        assert prompt.system_prompt == system_prompt
        assert prompt.user_prompt == ""
        assert prompt.settings.provider == "azure"
        assert prompt.settings.model == "gpt-4"

    def test_memory_provider_with_response_schema(self):
        """Test MemoryProvider with response schema."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        system_prompt = "You are a helpful assistant."
        
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt=system_prompt,
            prompt_settings=settings
        )
        
        # Act
        loader = PromptLoader(provider, "test_prompt", response_schema=TestResponseSchema)
        prompt = loader.get_prompt()
        
        # Assert
        assert prompt.response_schema == TestResponseSchema

    def test_memory_provider_with_placeholders(self):
        """Test MemoryProvider with placeholder replacement."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        system_prompt = "You are a {role} assistant. You help with {task_type}."
        user_prompt = "Please help me with {specific_task}."
        
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_settings=settings
        )
        
        placeholders = {
            "role": "technical",
            "task_type": "programming",
            "specific_task": "debugging Python code"
        }
        
        # Act
        loader = PromptLoader(provider, "test_prompt", placeholders=placeholders)
        prompt = loader.get_prompt()
        
        # Assert
        assert prompt.system_prompt == "You are a technical assistant. You help with programming."
        assert prompt.user_prompt == "Please help me with debugging Python code."

    def test_memory_provider_partial_placeholders(self):
        """Test MemoryProvider with partial placeholder replacement."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        system_prompt = "You are a {role} assistant. You help with {task_type}."
        
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt=system_prompt,
            prompt_settings=settings
        )
        
        placeholders = {
            "role": "technical"
            # task_type is missing
        }
        
        # Act
        loader = PromptLoader(provider, "test_prompt", placeholders=placeholders)
        prompt = loader.get_prompt()
        
        # Assert
        assert prompt.system_prompt == "You are a technical assistant. You help with {task_type}."

    def test_memory_provider_empty_files_list(self):
        """Test MemoryProvider with empty files list (should raise error)."""
        # Arrange - Create a mock provider that returns empty list
        class EmptyProvider(MemoryProvider):
            def list_files(self):
                return []
        
        settings = PromptSettings(provider="openai", model="gpt-4")
        provider = EmptyProvider(
            prompt_name="test_prompt",
            system_prompt="test",
            prompt_settings=settings
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="Prompt 'test_prompt' does not exist in storage"):
            PromptLoader(provider, "test_prompt")

    def test_memory_provider_missing_settings_file(self):
        """Test MemoryProvider with missing settings.json (should raise error)."""
        # Arrange - Create a mock provider that doesn't return settings.json
        class NoSettingsProvider(MemoryProvider):
            def download_file(self, file_name: str):
                if file_name == "settings.json":
                    return None
                return super().download_file(file_name)
        
        settings = PromptSettings(provider="openai", model="gpt-4")
        provider = NoSettingsProvider(
            prompt_name="test_prompt",
            system_prompt="test",
            prompt_settings=settings
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="settings.json not found in prompt directory"):
            PromptLoader(provider, "test_prompt")

    def test_memory_provider_missing_system_prompt(self):
        """Test MemoryProvider with missing system.md (should raise error)."""
        # Arrange - Create a mock provider that doesn't return system.md
        class NoSystemProvider(MemoryProvider):
            def download_file(self, file_name: str):
                if file_name == "system.md":
                    return None
                return super().download_file(file_name)
        
        settings = PromptSettings(provider="openai", model="gpt-4")
        provider = NoSystemProvider(
            prompt_name="test_prompt",
            system_prompt="test",
            prompt_settings=settings
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="system.md not found in prompt directory"):
            PromptLoader(provider, "test_prompt")

    def test_get_prompt_settings(self):
        """Test getting prompt settings from loader."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt="test",
            prompt_settings=settings
        )
        
        # Act
        loader = PromptLoader(provider, "test_prompt")
        retrieved_settings = loader.get_prompt_settings()
        
        # Assert
        assert retrieved_settings == settings
        assert retrieved_settings.provider == "openai"
        assert retrieved_settings.model == "gpt-4"


class TestFileSystemProvider:
    """Test cases for FileSystemProvider with PromptLoader."""

    @pytest.fixture
    def temp_prompt_dir(self):
        """Create a temporary directory with test prompt files."""
        temp_dir = tempfile.mkdtemp()
        prompt_dir = os.path.join(temp_dir, "prompts", "test_prompt")
        os.makedirs(prompt_dir)
        
        # Create settings.json
        settings = PromptSettings(provider="openai", model="gpt-4")
        with open(os.path.join(prompt_dir, "settings.json"), "w") as f:
            json.dump(settings.__dict__, f)
        
        # Create system.md
        with open(os.path.join(prompt_dir, "system.md"), "w") as f:
            f.write("You are a helpful assistant.")
        
        # Create user.md
        with open(os.path.join(prompt_dir, "user.md"), "w") as f:
            f.write("Hello, how are you?")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_filesystem_provider_basic_loading(self, temp_prompt_dir):
        """Test basic prompt loading with FileSystemProvider."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        loader = PromptLoader(provider, "test_prompt")
        prompt = loader.get_prompt()
        
        # Assert
        assert prompt.system_prompt == "You are a helpful assistant."
        assert prompt.user_prompt == "Hello, how are you?"
        assert prompt.settings.provider == "openai"
        assert prompt.settings.model == "gpt-4"

    def test_filesystem_provider_without_user_prompt(self, temp_prompt_dir):
        """Test FileSystemProvider without user.md file."""
        # Arrange
        prompt_dir = os.path.join(temp_prompt_dir, "prompts", "test_prompt")
        os.remove(os.path.join(prompt_dir, "user.md"))
        
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        loader = PromptLoader(provider, "test_prompt")
        prompt = loader.get_prompt()
        
        # Assert
        assert prompt.system_prompt == "You are a helpful assistant."
        assert prompt.user_prompt == ""
        assert prompt.settings.provider == "openai"
        assert prompt.settings.model == "gpt-4"

    def test_filesystem_provider_with_placeholders(self, temp_prompt_dir):
        """Test FileSystemProvider with placeholder replacement."""
        # Arrange
        prompt_dir = os.path.join(temp_prompt_dir, "prompts", "test_prompt")
        
        # Update system.md with placeholders
        with open(os.path.join(prompt_dir, "system.md"), "w") as f:
            f.write("You are a {role} assistant. You help with {task_type}.")
        
        # Update user.md with placeholders
        with open(os.path.join(prompt_dir, "user.md"), "w") as f:
            f.write("Please help me with {specific_task}.")
        
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        placeholders = {
            "role": "technical",
            "task_type": "programming",
            "specific_task": "debugging Python code"
        }
        
        # Act
        loader = PromptLoader(provider, "test_prompt", placeholders=placeholders)
        prompt = loader.get_prompt()
        
        # Assert
        assert prompt.system_prompt == "You are a technical assistant. You help with programming."
        assert prompt.user_prompt == "Please help me with debugging Python code."

    def test_filesystem_provider_with_response_schema(self, temp_prompt_dir):
        """Test FileSystemProvider with response schema."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        loader = PromptLoader(provider, "test_prompt", response_schema=TestResponseSchema)
        prompt = loader.get_prompt()
        
        # Assert
        assert prompt.response_schema == TestResponseSchema

    def test_filesystem_provider_nonexistent_directory(self):
        """Test FileSystemProvider with nonexistent directory (should raise error)."""
        # Act & Assert
        with pytest.raises(ValueError, match="Prompts directory '.*nonexistent.*' does not exist"):
            FileSystemProvider("nonexistent", base_path="nonexistent_path")

    def test_filesystem_provider_file_is_directory(self, temp_prompt_dir):
        """Test FileSystemProvider when prompt path is a file, not directory (should raise error)."""
        # Arrange
        prompt_dir = os.path.join(temp_prompt_dir, "prompts", "test_prompt")
        os.remove(os.path.join(prompt_dir, "settings.json"))
        os.remove(os.path.join(prompt_dir, "system.md"))
        os.remove(os.path.join(prompt_dir, "user.md"))
        os.rmdir(prompt_dir)
        
        # Create a file with the same name
        with open(prompt_dir, "w") as f:
            f.write("test")
        
        # Act & Assert
        with pytest.raises(ValueError, match="'.*' is not a directory"):
            FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))

    def test_filesystem_provider_list_files(self, temp_prompt_dir):
        """Test FileSystemProvider list_files method."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        files = provider.list_files()
        
        # Assert
        assert len(files) == 3
        file_names = [f["name"] for f in files]
        assert "settings.json" in file_names
        assert "system.md" in file_names
        assert "user.md" in file_names
        
        # Check content types
        for file_info in files:
            if file_info["name"] == "settings.json":
                assert file_info["content_type"] == "application/json"
            elif file_info["name"].endswith(".md"):
                assert file_info["content_type"] == "text/markdown"

    def test_filesystem_provider_file_exists(self, temp_prompt_dir):
        """Test FileSystemProvider file_exists method."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act & Assert
        assert provider.file_exists("settings.json") is True
        assert provider.file_exists("system.md") is True
        assert provider.file_exists("user.md") is True
        assert provider.file_exists("nonexistent.md") is False

    def test_filesystem_provider_download_file(self, temp_prompt_dir):
        """Test FileSystemProvider download_file method."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        settings_content = provider.download_file("settings.json")
        system_content = provider.download_file("system.md")
        user_content = provider.download_file("user.md")
        nonexistent_content = provider.download_file("nonexistent.md")
        
        # Assert
        assert settings_content is not None
        assert system_content is not None
        assert user_content is not None
        assert nonexistent_content is None
        
        # Check content
        assert b"openai" in settings_content
        assert b"gpt-4" in settings_content
        assert b"You are a helpful assistant." in system_content
        assert b"Hello, how are you?" in user_content


class TestStorageProviderBase:
    """Test cases for StorageProviderBase abstract class."""

    def test_storage_provider_base_instantiation(self):
        """Test that StorageProviderBase cannot be instantiated directly."""
        # Act & Assert
        with pytest.raises(TypeError):
            StorageProviderBase()


class TestIntegration:
    """Integration tests for prompt loading functionality."""

    def test_memory_to_prompt_integration(self):
        """Test full integration from MemoryProvider to Prompt object."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        system_prompt = "You are a helpful assistant."
        user_prompt = "Hello, how are you?"
        
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_settings=settings
        )
        
        # Act
        loader = PromptLoader(provider, "test_prompt")
        prompt = loader.get_prompt()
        
        # Assert
        assert isinstance(prompt, Prompt)
        assert prompt.system_prompt == system_prompt
        assert prompt.user_prompt == user_prompt
        assert prompt.settings == settings
        assert prompt.response_schema is None

    def test_filesystem_to_prompt_integration(self, temp_prompt_dir):
        """Test full integration from FileSystemProvider to Prompt object."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        loader = PromptLoader(provider, "test_prompt")
        prompt = loader.get_prompt()
        
        # Assert
        assert isinstance(prompt, Prompt)
        assert prompt.system_prompt == "You are a helpful assistant."
        assert prompt.user_prompt == "Hello, how are you?"
        assert prompt.settings.provider == "openai"
        assert prompt.settings.model == "gpt-4"

    @pytest.fixture
    def temp_prompt_dir(self):
        """Create a temporary directory with test prompt files."""
        temp_dir = tempfile.mkdtemp()
        prompt_dir = os.path.join(temp_dir, "prompts", "test_prompt")
        os.makedirs(prompt_dir)
        
        # Create settings.json
        settings = PromptSettings(provider="openai", model="gpt-4")
        with open(os.path.join(prompt_dir, "settings.json"), "w") as f:
            json.dump(settings.__dict__, f)
        
        # Create system.md
        with open(os.path.join(prompt_dir, "system.md"), "w") as f:
            f.write("You are a helpful assistant.")
        
        # Create user.md
        with open(os.path.join(prompt_dir, "user.md"), "w") as f:
            f.write("Hello, how are you?")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir) 