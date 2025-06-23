"""
Date and time related tools for time operations.
"""
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List
from openai import pydantic_function_tool

class GetCurrentTime(BaseModel):
    """Get the current time in a specific timezone. Use this when asked for current time, local time, or what time it is in a particular location."""
    timezone: str = Field(default="UTC", description="Timezone to get current time in IANA format (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo')")

    def execute(self) -> str:
        """Get the current time in the specified timezone."""
        tz = ZoneInfo(self.timezone)
        current_time = datetime.now(tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

class FormatDateTime(BaseModel):
    """Format a datetime string from one format to another. Use this when you need to convert date/time strings between different formats."""
    datetime_str: str = Field(..., description="Date/time string to format")
    input_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Input format of the datetime string")
    output_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Output format for the datetime")

    def execute(self) -> str:
        """Format a datetime string from one format to another."""
        dt = datetime.strptime(self.datetime_str, self.input_format)
        return dt.strftime(self.output_format)

class AddTime(BaseModel):
    """Add time to a datetime string. Use this when you need to calculate future or past dates/times."""
    datetime_str: str = Field(..., description="Starting datetime string")
    days: int = Field(default=0, description="Days to add")
    hours: int = Field(default=0, description="Hours to add")
    minutes: int = Field(default=0, description="Minutes to add")
    input_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Format of the input datetime")

    def execute(self) -> str:
        """Add time to a datetime string."""
        dt = datetime.strptime(self.datetime_str, self.input_format)
        delta = timedelta(days=self.days, hours=self.hours, minutes=self.minutes)
        result = dt + delta
        return result.strftime(self.input_format)

class GetTimeDifference(BaseModel):
    """Calculate the difference between two datetime strings. Use this when you need to find the time span between two dates/times."""
    datetime1: str = Field(..., description="First datetime string")
    datetime2: str = Field(..., description="Second datetime string")
    format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Format of both datetime strings")

    def execute(self) -> dict:
        """Calculate the difference between two datetime strings."""
        dt1 = datetime.strptime(self.datetime1, self.format)
        dt2 = datetime.strptime(self.datetime2, self.format)
        diff = abs(dt2 - dt1)
        
        return {
            "total_seconds": diff.total_seconds(),
            "days": diff.days,
            "hours": diff.seconds // 3600,
            "minutes": (diff.seconds % 3600) // 60,
            "seconds": diff.seconds % 60
        }

class IsWeekend(BaseModel):
    """Check if a given date falls on a weekend. Use this when you need to determine if a date is Saturday or Sunday."""
    date_str: str = Field(..., description="Date string to check")
    format: str = Field(default="%Y-%m-%d", description="Format of the date string")

    def execute(self) -> bool:
        """Check if a given date falls on a weekend."""
        dt = datetime.strptime(self.date_str, self.format)
        return dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday

class GetDayOfWeek(BaseModel):
    """Get the day of the week for a given date. Use this when you need to know what day of the week a specific date falls on."""
    date_str: str = Field(..., description="Date string to get day of week for")
    format: str = Field(default="%Y-%m-%d", description="Format of the date string")

    def execute(self) -> str:
        """Get the day of the week for a given date."""
        dt = datetime.strptime(self.date_str, self.format)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[dt.weekday()]

# Register all datetime tools in the global registry
from smartprompt.tool_registry import register_tool
register_tool(GetCurrentTime)
register_tool(FormatDateTime)
register_tool(AddTime)
register_tool(GetTimeDifference)
register_tool(IsWeekend)
register_tool(GetDayOfWeek)

def get_datetime_tools():
    """Get all datetime tools as pydantic_function_tool objects."""
    return [
        pydantic_function_tool(GetCurrentTime),
        pydantic_function_tool(FormatDateTime),
        pydantic_function_tool(AddTime),
        pydantic_function_tool(GetTimeDifference),
        pydantic_function_tool(IsWeekend),
        pydantic_function_tool(GetDayOfWeek)
    ] 