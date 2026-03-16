"""
Sub-agent: location_image_agent

Runs inside a ParallelAgent alongside character_portrait_agent.
Its sole job is to call generate_location_images immediately.
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from film_generator_agent.tools.locations import generate_location_images

location_image_agent = Agent(
    name="location_image_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a location image generation sub-agent. "
        "When invoked, immediately call generate_location_images() with no arguments. "
        "Do not ask questions. Do not wait. Call the tool right away and return the result."
    ),
    tools=[
        FunctionTool(func=generate_location_images),
    ],
)
