from dataclasses import dataclass
from typing import Optional, Type, Any, List

@dataclass
class PromptSettings:
    provider: str
    model: str

@dataclass
class Prompt:
    """A data class that holds prompt information and expected response schema."""
    system_prompt: str
    user_prompt: str
    settings: PromptSettings
    response_schema: Optional[Type[Any]] = None
    tools: Optional[List[Any]] = None
    
    @classmethod
    def from_definition(cls, system_prompt: str, user_prompt: str, prompt_settings: PromptSettings, response_schema: Optional[Type[Any]] = None, tools: Optional[List[Any]] = None):
        """Create a Prompt from a PromptSettings."""
        return cls(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            settings=prompt_settings,
            response_schema=response_schema,
            tools=tools
        ) 