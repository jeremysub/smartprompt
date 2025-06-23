from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ToolCallRequest:
    """A request to call a specific tool with arguments."""
    id: str
    function_name: str
    arguments: Dict[str, Any]

@dataclass
class ToolCallResponse:
    """A response from executing a tool call."""
    tool_call_id: str
    function_name: str
    arguments: Dict[str, Any]
    result: Any
    status: str  # 'success' or 'error'

class ToolRunner:
    """Helper class to manage tool execution workflow."""
    def __init__(self):
        """
        Initialize ToolRunner.
        """
        # Import the tool registry
        from smartprompt.tool_registry import tool_registry
        self._registry = tool_registry
    
    def run_tools(self, tool_call_requests: List[ToolCallRequest]) -> List[ToolCallResponse]:
        """
        Run a list of tool requests and return the results.
        
        Args:
            tool_requests: List of ToolCallRequest objects to execute
            
        Returns:
            List of ToolCallResponse objects with execution details
        """
        if not tool_call_requests:
            return []
        
        tool_call_responses = []
        
        for tool_call in tool_call_requests:
            response = self._dispatch_tool_call(tool_call)
            tool_call_responses.append(response)
            
        return tool_call_responses
    
    def _dispatch_tool_call(self, tool_request: ToolCallRequest) -> ToolCallResponse:
        """
        Dispatch a tool call to the appropriate tool using the ToolRegistry.
        This method handles the tool execution and returns a ToolCallResponse.
        """
        try:
            # Get the tool class from registry
            tool_class = self._registry.get_tool(tool_request.function_name)
            
            # Create tool instance and execute
            tool_instance = tool_class(**tool_request.arguments)
            result = tool_instance.execute()
            
            return ToolCallResponse(
                tool_call_id=tool_request.id,
                function_name=tool_request.function_name,
                arguments=tool_request.arguments,
                result=result,
                status='success'
            )
            
        except Exception as e:
            return ToolCallResponse(
                tool_call_id=tool_request.id,
                function_name=tool_request.function_name,
                arguments=tool_request.arguments,
                result=str(e),
                status='error'
            )