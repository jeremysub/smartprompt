import os
import json
from typing import Any, List
from openai import OpenAI, AzureOpenAI
from .prompt import Prompt
from .tool_runner import ToolRunner, ToolCallRequest

class ModelClient:
    """Handles common model interactions."""
    
    def __init__(self, prompt: Prompt):
        self._prompt = prompt
        
        # initialize the client
        if self._prompt.settings.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is not set")
            
            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self._prompt.settings.provider == "azure":
            if not os.getenv("AZURE_OPENAI_ENDPOINT_URL"):
                raise ValueError("AZURE_OPENAI_ENDPOINT_URL is not set")
            if not os.getenv("AZURE_OPENAI_API_KEY"):
                raise ValueError("AZURE_OPENAI_API_KEY is not set")
            if not os.getenv("AZURE_OPENAI_API_VERSION"):
                raise ValueError("AZURE_OPENAI_API_VERSION is not set")
            
            self._client = AzureOpenAI(
                base_url=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION")
            )
        elif self._prompt.settings.provider == "grok":
            if not os.getenv("GROK_API_KEY"):
                raise ValueError("GROK_API_KEY is not set")
            
            self._client = OpenAI(
                api_key=os.getenv("GROK_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
        else:
            raise ValueError(f"Invalid provider: {self._prompt.settings.provider}")

    def _build_base_messages(self) -> List[dict]:
        """
        Build base conversation messages.
        
        Returns:
            List of conversation messages
        """
        
        messages = [
            {
                'role': 'system',
                'content': self._prompt.system_prompt,
            },
            {
                'role': 'user',
                'content': self._prompt.user_prompt
            }
        ]

        return messages
    
    def get_structured_completion(self) -> Any:
        """
        Get a structured completion from the LLM.
        
        Returns:
            An instance of the response schema/object
        """
        # Use provided schema or get from prompt
        if self._prompt.response_schema is None:
            raise ValueError("No response schema provided and none defined in prompt")
            
        completion = self._client.beta.chat.completions.parse(
            model=self._prompt.settings.model,
            response_format=self._prompt.response_schema,
            messages = self._build_base_messages()
        )
        
        return completion.choices[0].message.parsed

    def get_text_completion(self) -> str:
        """
        Get a text completion from the LLM.
        
        Returns:
            Text response
        """
        
        # Get tools from the prompt
        tools = self._prompt.tools
        
        if tools is None:
            
            # if no tools, just return the text completion
            completion = self._client.chat.completions.create(
                model=self._prompt.settings.model,
                messages=self._build_base_messages()
            )
            
            return completion.choices[0].message.content
        
        else:
            # if tools, call the LLM twice
            
            # build the messages
            messages = self._build_base_messages()
            
            # first call: LLM determines what tools to call
            completion = self._client.chat.completions.create(
                model=self._prompt.settings.model,
                messages=messages,
                tools=tools
            )
            
            # get tool requests from the first call
            raw_tool_requests = completion.choices[0].message.tool_calls
            
            tool_requests = [ToolCallRequest(
                id=tool_call.id,
                function_name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments)
            ) for tool_call in raw_tool_requests]

            
            # execute tools with available tools mapping
            tool_runner = ToolRunner()
            tool_results = tool_runner.run_tools(tool_requests)

            # second call: make the same call, but with:
            # - previous messages
            # - last completion message
            # - tool results
            # - list of tools to call

            # append message from previous call            
            messages.append(completion.choices[0].message)

            # append tool results
            for result in tool_results:
                messages.append({
                    'role': 'tool',
                    'tool_call_id': result.tool_call_id,
                    'content': str(result.result)
                })
            
            completion = self._client.chat.completions.create(
                model=self._prompt.settings.model,
                messages=messages,
                tools=tools
            )
            
            return completion.choices[0].message.content
    
    def get_streamed_completion(self):
        """
        Get a streamed completion from the LLM.
        
        Returns:
            Streaming response
        """
        
        stream = self._client.chat.completions.create(
            model=self._prompt.settings.model,
            messages=self._build_base_messages(),
            stream=True
        )
        
        return stream 
