from smartprompt import ModelClient, Prompt, PromptSettings
from smartprompt.tools.datetime_tools import get_datetime_tools

def tool_calling_example():
    """Example showing the clean LLM -> Tools -> LLM workflow"""
    
    # Get datetime tools and debug
    datetime_tools = get_datetime_tools()
    
    # create LLM client for initial request
    my_settings = PromptSettings(provider="azure", model="gpt-4o")
    my_prompt = Prompt(
        system_prompt="You are a helpful assistant that can use tools to help users.",
        user_prompt="Can you get the current local time in Louisville, KY.",
        settings=my_settings,
        tools=datetime_tools
    )
    
    my_client = ModelClient(my_prompt)
    
    # call LLM with tools from the prompt
    results = my_client.get_text_completion()
    
    print(f"Tool call requests: {results}")

if __name__ == "__main__":
    tool_calling_example()