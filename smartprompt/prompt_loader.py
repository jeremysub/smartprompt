import json
import re
from typing import Any, Optional, Dict, Type, List
from .prompt import Prompt, PromptSettings
from .storage_providers import StorageProviderBase

class PromptLoader:
    """
    Loads and processes prompts from a storage provider.
    The provider can be memory, blob storage, file system, or any custom implementation
    that implements the StorageProviderBase interface (list_files, download_file, etc).
    """
    def __init__(
        self,
        provider: StorageProviderBase,
        placeholders: Optional[Dict[str, str]] = None,
        response_schema: Optional[Type[Any]] = None,
        tools: Optional[List[Any]] = None
    ):
        """
        Args:
            prompt_name: Name of the prompt directory
            provider: Storage provider instance (required)
            placeholders: Dict of placeholder values to fill in prompt text
            response_schema: Optional response schema type
        """
        self._provider = provider
        self._response_schema = response_schema
        self._tools = tools

        # Check if prompt directory exists
        if not self._provider.list_files():
            raise ValueError(f"Prompt does not exist in storage")

        # Load prompt settings
        prompt_settings_content = self._provider.download_file("settings.json")
        if prompt_settings_content is None:
            raise ValueError("settings.json not found in prompt directory")
        prompt_settings_json = json.loads(prompt_settings_content.decode("utf-8"))
        self._prompt_settings = PromptSettings(**prompt_settings_json)

        # Load system prompt
        system_prompt_content = self._provider.download_file("system.md")
        if system_prompt_content is None:
            raise ValueError("system.md not found in prompt directory")
        self._system_prompt = system_prompt_content.decode("utf-8")

        # Load user prompt (optional)
        self._user_prompt = ""
        user_prompt_content = self._provider.download_file("user.md")
        if user_prompt_content is not None:
            self._user_prompt = user_prompt_content.decode("utf-8")

        # Fill placeholders if provided
        if placeholders:
            self._fill_placeholders(placeholders)

    def _fill_placeholders(self, placeholders: Dict[str, str]) -> None:
        """
        Replace placeholders in the prompt with the given values.
        This is a private method called only during initialization.
        Args:
            placeholders: Dictionary of placeholder name to value mappings
        """
        for placeholder in re.findall(r"{(.*?)}", self._system_prompt):
            if placeholder in placeholders:
                self._system_prompt = self._system_prompt.replace(f"{{{placeholder}}}", placeholders[placeholder])
        for placeholder in re.findall(r"{(.*?)}", self._user_prompt):
            if placeholder in placeholders:
                self._user_prompt = self._user_prompt.replace(f"{{{placeholder}}}", placeholders[placeholder])

    def get_prompt(self) -> Prompt:
        """
        Get the current prompt as a Prompt object.
        Returns:
            Prompt object with current system_prompt, user_prompt, model, and response_schema
        """
        return Prompt.from_definition(
            system_prompt=self._system_prompt,
            user_prompt=self._user_prompt,
            prompt_settings=self._prompt_settings,
            response_schema=self._response_schema
        )

    def get_prompt_settings(self) -> PromptSettings:
        """Get the prompt settings."""
        return self._prompt_settings 