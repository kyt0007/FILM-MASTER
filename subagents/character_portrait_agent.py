"""
Sub-agent: character_portrait_agent

Runs inside a ParallelAgent alongside location_image_agent.
Its sole job is to call generate_character_portraits immediately, then optionally
handle regeneration requests from the parent agent.
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from film_generator_agent.tools.characters import (
    generate_character_portraits,
    regenerate_character_portrait,
)

character_portrait_agent = Agent(
    name="character_portrait_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a character portrait generation sub-agent. "
        "When invoked, immediately call generate_character_portraits() with no arguments. "
        "Do not ask questions. Do not wait. Call the tool right away and return the result. "
        "If asked to regenerate a specific character, call regenerate_character_portrait(name, new_description)."
    ),
    tools=[
        FunctionTool(func=generate_character_portraits),
        FunctionTool(func=regenerate_character_portrait),
    ],
)
