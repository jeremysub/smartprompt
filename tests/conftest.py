import pytest
import json
import os
import tempfile
import shutil
from smartprompt.prompt import PromptSettings


@pytest.fixture
def temp_prompt_dir():
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


@pytest.fixture
def sample_prompt_settings():
    """Sample prompt settings for testing."""
    return PromptSettings(provider="openai", model="gpt-4")


@pytest.fixture
def sample_system_prompt():
    """Sample system prompt for testing."""
    return "You are a helpful assistant."


@pytest.fixture
def sample_user_prompt():
    """Sample user prompt for testing."""
    return "Hello, how are you?" 