from typing import Dict, Type, List
from pydantic import BaseModel
from openai import pydantic_function_tool

class ToolRegistry:
    """Registry for managing all available tools."""
    def __init__(self):
        self._tools: Dict[str, Type[BaseModel]] = {}

    def register_tool(self, tool_class: Type[BaseModel]):
        """Register a new tool class."""
        self._tools[tool_class.__name__] = tool_class

    def get_tool(self, name: str) -> Type[BaseModel]:
        """Get a tool class by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found. Available tools: {list(self._tools.keys())}")
        return self._tools[name]

    def get_all_tools(self) -> Dict[str, Type[BaseModel]]:
        """Get all registered tools."""
        return self._tools.copy()

    def get_tool_names(self) -> List[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())

    def create_pydantic_tools(self, tool_names: List[str] = None) -> List:
        """Create pydantic_function_tool objects for selected or all registered tools."""
        if tool_names is None:
            tool_classes = self._tools.values()
        else:
            tool_classes = [self.get_tool(name) for name in tool_names]
        return [pydantic_function_tool(cls) for cls in tool_classes]

# Global registry instance
tool_registry = ToolRegistry()

def register_tool(tool_class):
    """Register a new tool with the global registry."""
    tool_registry.register_tool(tool_class)

def get_pydantic_tools(tool_names: List[str] = None):
    """Get pydantic_function_tool objects for selected or all registered tools."""
    return tool_registry.create_pydantic_tools(tool_names) 