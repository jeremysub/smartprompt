from smartprompt import ModelClient, Prompt, PromptSettings
from math_tools import get_math_tools

def custom_tool_example():
    """Example showing how to use custom math tools with the LLM"""
    
    # Get our custom math tools
    math_tools = get_math_tools()
    
    # create LLM client for math operations
    my_settings = PromptSettings(provider="azure", model="gpt-4o")
    my_prompt = Prompt(
        system_prompt="You are a helpful math assistant that can perform calculations using available tools. Use the appropriate math tools to solve problems accurately.",
        user_prompt="Please help me with these calculations: 1) Add 15, 23, and 7 together. 2) Multiply 4, 5, and 6. 3) Calculate 10 to the power of 3. 4) Find the square root of 144. 5) Calculate the average of 85, 92, 78, and 95.",
        settings=my_settings,
        tools=math_tools
    )
    
    my_client = ModelClient(my_prompt)
    
    # call LLM with tools from the prompt
    try:
        results = my_client.get_text_completion()
        
        print("Custom math tools integration test:")
        print("=" * 50)
        print(results)
        print("=" * 50)
        
        # Verify that the response contains mathematical results
        assert "result" in results.lower() or "answer" in results.lower() or "calculation" in results.lower()
        
        print("✅ Custom math tools integration test completed successfully!")
        
    except Exception as e:
        print(f"Error during custom tool execution: {e}")
        raise

if __name__ == "__main__":

    # Then test integration with LLM
    custom_tool_example()
