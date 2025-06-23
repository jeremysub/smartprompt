import pytest
import json
import os
import tempfile
import shutil
from typing import List, Dict, Any

from smartprompt.storage_providers import MemoryProvider, FileSystemProvider, StorageProviderBase
from smartprompt.prompt import PromptSettings


class TestMemoryProvider:
    """Test cases for MemoryProvider class."""

    def test_memory_provider_initialization(self):
        """Test MemoryProvider initialization with all parameters."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        system_prompt = "You are a helpful assistant."
        user_prompt = "Hello, how are you?"
        
        # Act
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_settings=settings
        )
        
        # Assert
        assert provider.prompt_name == "test_prompt"
        assert provider._system_prompt == system_prompt
        assert provider._user_prompt == user_prompt
        assert provider._prompt_settings == settings

    def test_memory_provider_initialization_without_user_prompt(self):
        """Test MemoryProvider initialization without user prompt."""
        # Arrange
        settings = PromptSettings(provider="azure", model="gpt-4")
        system_prompt = "You are a coding assistant."
        
        # Act
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt=system_prompt,
            prompt_settings=settings
        )
        
        # Assert
        assert provider.prompt_name == "test_prompt"
        assert provider._system_prompt == system_prompt
        assert provider._user_prompt == ""
        assert provider._prompt_settings == settings

    def test_memory_provider_list_files_with_user_prompt(self):
        """Test MemoryProvider list_files method with user prompt."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt="test system",
            user_prompt="test user",
            prompt_settings=settings
        )
        
        # Act
        files = provider.list_files()
        
        # Assert
        assert len(files) == 3
        file_names = [f["name"] for f in files]
        assert "settings.json" in file_names
        assert "system.md" in file_names
        assert "user.md" in file_names
        
        # Check file metadata
        for file_info in files:
            assert "name" in file_info
            assert "filename" in file_info
            assert "size" in file_info
            assert "created_at" in file_info
            assert "modified_at" in file_info
            assert "content_type" in file_info
            
            if file_info["name"] == "settings.json":
                assert file_info["content_type"] == "application/json"
            elif file_info["name"].endswith(".md"):
                assert file_info["content_type"] == "text/markdown"

    def test_memory_provider_list_files_without_user_prompt(self):
        """Test MemoryProvider list_files method without user prompt."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt="test system",
            prompt_settings=settings
        )
        
        # Act
        files = provider.list_files()
        
        # Assert
        assert len(files) == 2
        file_names = [f["name"] for f in files]
        assert "settings.json" in file_names
        assert "system.md" in file_names
        assert "user.md" not in file_names

    def test_memory_provider_download_file_settings(self):
        """Test MemoryProvider download_file method for settings.json."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt="test system",
            prompt_settings=settings
        )
        
        # Act
        content = provider.download_file("settings.json")
        
        # Assert
        assert content is not None
        decoded_content = content.decode("utf-8")
        settings_dict = json.loads(decoded_content)
        assert settings_dict["provider"] == "openai"
        assert settings_dict["model"] == "gpt-4"

    def test_memory_provider_download_file_system_prompt(self):
        """Test MemoryProvider download_file method for system.md."""
        # Arrange
        system_prompt = "You are a helpful assistant."
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt=system_prompt,
            prompt_settings=PromptSettings(provider="openai", model="gpt-4")
        )
        
        # Act
        content = provider.download_file("system.md")
        
        # Assert
        assert content is not None
        decoded_content = content.decode("utf-8")
        assert decoded_content == system_prompt

    def test_memory_provider_download_file_user_prompt(self):
        """Test MemoryProvider download_file method for user.md."""
        # Arrange
        user_prompt = "Hello, how are you?"
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt="test system",
            user_prompt=user_prompt,
            prompt_settings=PromptSettings(provider="openai", model="gpt-4")
        )
        
        # Act
        content = provider.download_file("user.md")
        
        # Assert
        assert content is not None
        decoded_content = content.decode("utf-8")
        assert decoded_content == user_prompt

    def test_memory_provider_download_file_user_prompt_empty(self):
        """Test MemoryProvider download_file method for user.md when empty."""
        # Arrange
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt="test system",
            prompt_settings=PromptSettings(provider="openai", model="gpt-4")
        )
        
        # Act
        content = provider.download_file("user.md")
        
        # Assert
        assert content is None

    def test_memory_provider_download_file_nonexistent(self):
        """Test MemoryProvider download_file method for nonexistent file."""
        # Arrange
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt="test system",
            prompt_settings=PromptSettings(provider="openai", model="gpt-4")
        )
        
        # Act
        content = provider.download_file("nonexistent.md")
        
        # Assert
        assert content is None

    def test_memory_provider_file_exists(self):
        """Test MemoryProvider file_exists method."""
        # Arrange
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt="test system",
            user_prompt="test user",
            prompt_settings=PromptSettings(provider="openai", model="gpt-4")
        )
        
        # Act & Assert
        assert provider.file_exists("settings.json") is True
        assert provider.file_exists("system.md") is True
        assert provider.file_exists("user.md") is True
        assert provider.file_exists("nonexistent.md") is False

    def test_memory_provider_file_exists_without_user_prompt(self):
        """Test MemoryProvider file_exists method without user prompt."""
        # Arrange
        provider = MemoryProvider(
            prompt_name="test_prompt",
            system_prompt="test system",
            prompt_settings=PromptSettings(provider="openai", model="gpt-4")
        )
        
        # Act & Assert
        assert provider.file_exists("settings.json") is True
        assert provider.file_exists("system.md") is True
        assert provider.file_exists("user.md") is False
        assert provider.file_exists("nonexistent.md") is False


class TestFileSystemProvider:
    """Test cases for FileSystemProvider class."""

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

    def test_filesystem_provider_initialization(self, temp_prompt_dir):
        """Test FileSystemProvider initialization."""
        # Act
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Assert
        expected_path = os.path.join(temp_prompt_dir, "prompts", "test_prompt")
        assert provider._prompt_path == expected_path

    def test_filesystem_provider_initialization_nonexistent_directory(self):
        """Test FileSystemProvider initialization with nonexistent directory."""
        # Act & Assert
        with pytest.raises(ValueError, match="Prompts directory '.*nonexistent.*' does not exist"):
            FileSystemProvider("nonexistent", base_path="nonexistent_path")

    def test_filesystem_provider_initialization_file_not_directory(self, temp_prompt_dir):
        """Test FileSystemProvider initialization when path is a file, not directory."""
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
        
        # Check file metadata
        for file_info in files:
            assert "name" in file_info
            assert "filename" in file_info
            assert "size" in file_info
            assert "created_at" in file_info
            assert "modified_at" in file_info
            assert "content_type" in file_info
            
            if file_info["name"] == "settings.json":
                assert file_info["content_type"] == "application/json"
            elif file_info["name"].endswith(".md"):
                assert file_info["content_type"] == "text/markdown"

    def test_filesystem_provider_list_files_empty_directory(self, temp_prompt_dir):
        """Test FileSystemProvider list_files method with empty directory."""
        # Arrange
        prompt_dir = os.path.join(temp_prompt_dir, "prompts", "test_prompt")
        os.remove(os.path.join(prompt_dir, "settings.json"))
        os.remove(os.path.join(prompt_dir, "system.md"))
        os.remove(os.path.join(prompt_dir, "user.md"))
        
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        files = provider.list_files()
        
        # Assert
        assert len(files) == 0

    def test_filesystem_provider_download_file_settings(self, temp_prompt_dir):
        """Test FileSystemProvider download_file method for settings.json."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        content = provider.download_file("settings.json")
        
        # Assert
        assert content is not None
        decoded_content = content.decode("utf-8")
        settings_dict = json.loads(decoded_content)
        assert settings_dict["provider"] == "openai"
        assert settings_dict["model"] == "gpt-4"

    def test_filesystem_provider_download_file_system_prompt(self, temp_prompt_dir):
        """Test FileSystemProvider download_file method for system.md."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        content = provider.download_file("system.md")
        
        # Assert
        assert content is not None
        decoded_content = content.decode("utf-8")
        assert decoded_content == "You are a helpful assistant."

    def test_filesystem_provider_download_file_user_prompt(self, temp_prompt_dir):
        """Test FileSystemProvider download_file method for user.md."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        content = provider.download_file("user.md")
        
        # Assert
        assert content is not None
        decoded_content = content.decode("utf-8")
        assert decoded_content == "Hello, how are you?"

    def test_filesystem_provider_download_file_nonexistent(self, temp_prompt_dir):
        """Test FileSystemProvider download_file method for nonexistent file."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act
        content = provider.download_file("nonexistent.md")
        
        # Assert
        assert content is None

    def test_filesystem_provider_file_exists(self, temp_prompt_dir):
        """Test FileSystemProvider file_exists method."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act & Assert
        assert provider.file_exists("settings.json") is True
        assert provider.file_exists("system.md") is True
        assert provider.file_exists("user.md") is True
        assert provider.file_exists("nonexistent.md") is False

    def test_filesystem_provider_get_content_type(self, temp_prompt_dir):
        """Test FileSystemProvider _get_content_type method."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Act & Assert
        assert provider._get_content_type("test.json") == "application/json"
        assert provider._get_content_type("test.md") == "text/markdown"
        assert provider._get_content_type("test.txt") == "text/plain"
        assert provider._get_content_type("test.yaml") == "text/yaml"
        assert provider._get_content_type("test.yml") == "text/yaml"
        assert provider._get_content_type("test.unknown") == "application/octet-stream"

    def test_filesystem_provider_list_files_exception_handling(self, temp_prompt_dir):
        """Test FileSystemProvider list_files method with exception handling."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Remove the directory to simulate an error
        shutil.rmtree(os.path.join(temp_prompt_dir, "prompts", "test_prompt"))
        
        # Act
        files = provider.list_files()
        
        # Assert
        assert files == []

    def test_filesystem_provider_download_file_exception_handling(self, temp_prompt_dir):
        """Test FileSystemProvider download_file method with exception handling."""
        # Arrange
        provider = FileSystemProvider("test_prompt", base_path=os.path.join(temp_prompt_dir, "prompts"))
        
        # Remove the directory to simulate an error
        shutil.rmtree(os.path.join(temp_prompt_dir, "prompts", "test_prompt"))
        
        # Act
        content = provider.download_file("settings.json")
        
        # Assert
        assert content is None


class TestStorageProviderBase:
    """Test cases for StorageProviderBase abstract class."""

    def test_storage_provider_base_instantiation(self):
        """Test that StorageProviderBase cannot be instantiated directly."""
        # Act & Assert
        with pytest.raises(TypeError):
            StorageProviderBase()

    def test_storage_provider_base_interface(self):
        """Test that concrete providers implement the required interface."""
        # Arrange
        settings = PromptSettings(provider="openai", model="gpt-4")
        
        # Test MemoryProvider implements interface
        memory_provider = MemoryProvider(
            prompt_name="test",
            system_prompt="test",
            prompt_settings=settings
        )
        
        # Act & Assert
        assert hasattr(memory_provider, 'list_files')
        assert hasattr(memory_provider, 'download_file')
        assert hasattr(memory_provider, 'file_exists')
        
        # Test FileSystemProvider implements interface
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_dir = os.path.join(temp_dir, "prompts", "test")
            os.makedirs(prompt_dir)
            
            with open(os.path.join(prompt_dir, "settings.json"), "w") as f:
                json.dump(settings.__dict__, f)
            
            with open(os.path.join(prompt_dir, "system.md"), "w") as f:
                f.write("test")
            
            fs_provider = FileSystemProvider("test", base_path=os.path.join(temp_dir, "prompts"))
            
            assert hasattr(fs_provider, 'list_files')
            assert hasattr(fs_provider, 'download_file')
            assert hasattr(fs_provider, 'file_exists') 