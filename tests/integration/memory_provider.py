from smartprompt import ModelClient, Prompt, PromptSettings, get_all_tools, MemoryProvider, PromptLoader

def memory_provider_example():
    """Example showing the clean LLM -> Tools -> LLM workflow"""
    
    # create memory provider
    memory_provider = MemoryProvider(
        prompt_name="test-memory",
        system_prompt="You are a helpful assistant for Jeremy.",
        user_prompt="Write a short paragraph about the history of {city_state}.",
        prompt_settings=PromptSettings(provider="azure", model="gpt-4o")
    )
    
    # create loader
    loader = PromptLoader(memory_provider, "test-memory", placeholders={"city_state": "Louisville, KY"})
    
    # create LLM client for initial request
    prompt = loader.get_prompt()
    my_client = ModelClient(prompt)
    
    # call LLM with tools
    results = my_client.get_text_completion()
    
    print(f"Tool call requests: {results}")

if __name__ == "__main__":
    memory_provider_example()