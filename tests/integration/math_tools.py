"""
Math operations tools for basic arithmetic calculations.
This is a demonstration module for testing purposes only.
"""
from pydantic import BaseModel, Field
from typing import List
import math
from openai import pydantic_function_tool

class Add(BaseModel):
    numbers: List[float] = Field(..., description="List of numbers to add together")

    def execute(self) -> float:
        """Add multiple numbers together."""
        return sum(self.numbers)

class Subtract(BaseModel):
    a: float = Field(..., description="First number (minuend)")
    b: float = Field(..., description="Second number (subtrahend)")

    def execute(self) -> float:
        """Subtract b from a."""
        return self.a - self.b

class Multiply(BaseModel):
    numbers: List[float] = Field(..., description="List of numbers to multiply together")

    def execute(self) -> float:
        """Multiply multiple numbers together."""
        result = 1
        for num in self.numbers:
            result *= num
        return result

class Divide(BaseModel):
    a: float = Field(..., description="First number (dividend)")
    b: float = Field(..., description="Second number (divisor)")

    def execute(self) -> float:
        """Divide a by b."""
        if self.b == 0:
            raise ValueError("Cannot divide by zero")
        return self.a / self.b

class Power(BaseModel):
    base: float = Field(..., description="Base number")
    exponent: float = Field(..., description="Exponent")

    def execute(self) -> float:
        """Raise base to the power of exponent."""
        return math.pow(self.base, self.exponent)

class SquareRoot(BaseModel):
    number: float = Field(..., description="Number to find the square root of")

    def execute(self) -> float:
        """Calculate the square root of a number."""
        if self.number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return math.sqrt(self.number)

class Average(BaseModel):
    numbers: List[float] = Field(..., description="List of numbers to calculate average for")

    def execute(self) -> float:
        """Calculate the average (mean) of a list of numbers."""
        if not self.numbers:
            raise ValueError("Cannot calculate average of empty list")
        return sum(self.numbers) / len(self.numbers)

class Round(BaseModel):
    number: float = Field(..., description="Number to round")
    decimals: int = Field(default=0, description="Number of decimal places to round to")

    def execute(self) -> float:
        """Round a number to the specified number of decimal places."""
        return round(self.number, self.decimals)

# Register all math tools in the global registry
from smartprompt.tool_registry import register_tool
register_tool(Add)
register_tool(Subtract)
register_tool(Multiply)
register_tool(Divide)
register_tool(Power)
register_tool(SquareRoot)
register_tool(Average)
register_tool(Round)

def get_math_tools():
    """Get all math tools as pydantic_function_tool objects."""
    return [
        pydantic_function_tool(Add),
        pydantic_function_tool(Subtract),
        pydantic_function_tool(Multiply),
        pydantic_function_tool(Divide),
        pydantic_function_tool(Power),
        pydantic_function_tool(SquareRoot),
        pydantic_function_tool(Average),
        pydantic_function_tool(Round)
    ] 