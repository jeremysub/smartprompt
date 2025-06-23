from smartprompt import ModelClient, PromptLoader, BlobStorageProvider

def blob_provider_example():
    """Example showing the clean LLM -> Tools -> LLM workflow"""

    # create loader
    loader = PromptLoader(BlobStorageProvider(prompt_name="hello_llm"), placeholders={"city_state": "Louisville, KY"})
    
    # create LLM client for initial request
    prompt = loader.get_prompt()
    
    my_client = ModelClient(prompt)
    
    # call LLM with tools
    results = my_client.get_text_completion()
    
    print(f"Tool call requests: {results}")

if __name__ == "__main__":
    blob_provider_example()