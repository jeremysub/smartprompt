from pydantic import BaseModel, Field
from smartprompt import ModelClient, Prompt, PromptSettings

class WeatherReport(BaseModel):
    """A structured weather report response."""
    temperature: float = Field(..., description="Current temperature in Celsius")
    condition: str = Field(..., description="Weather condition (sunny, cloudy, rainy, etc.)")
    humidity: int = Field(..., description="Humidity percentage")
    wind_speed: float = Field(..., description="Wind speed in km/h")
    location: str = Field(..., description="Location name")
    forecast: str = Field(..., description="Brief weather forecast for the day")

def structured_completion_example():
    """Example showing structured completion with Pydantic model"""
    
    # create LLM client for structured request
    my_settings = PromptSettings(provider="azure", model="gpt-4o")
    my_prompt = Prompt(
        system_prompt="You are a weather assistant that provides structured weather reports. Always respond with accurate weather information in the requested format.",
        user_prompt="Provide a weather report for Louisville, KY. Make it realistic but you can use sample data.",
        settings=my_settings,
        response_schema=WeatherReport
    )
    
    my_client = ModelClient(my_prompt)
    
    # call LLM with structured completion
    try:
        weather_result = my_client.get_structured_completion()
        
        print(f"Structured completion result:")
        print(f"Temperature: {weather_result.temperature}°C")
        print(f"Condition: {weather_result.condition}")
        print(f"Humidity: {weather_result.humidity}%")
        print(f"Wind Speed: {weather_result.wind_speed} km/h")
        print(f"Location: {weather_result.location}")
        print(f"Forecast: {weather_result.forecast}")
        
        # Verify the result is a proper Pydantic model instance
        assert isinstance(weather_result, WeatherReport)
        assert isinstance(weather_result.temperature, float)
        assert isinstance(weather_result.humidity, int)
        assert isinstance(weather_result.wind_speed, float)
        assert isinstance(weather_result.condition, str)
        assert isinstance(weather_result.location, str)
        assert isinstance(weather_result.forecast, str)
        
        print("✅ All type assertions passed!")
        
    except Exception as e:
        print(f"Error during structured completion: {e}")
        raise

if __name__ == "__main__":
    structured_completion_example()
