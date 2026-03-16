"""Sub-agent: video_agent — generates all scene videos via Veo-3.0."""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from film_generator_agent.tools.video import generate_scene_videos, regenerate_scene_video

video_agent = Agent(
    name="video_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a scene video generation sub-agent. "
        "When invoked, immediately call generate_scene_videos() with no arguments. "
        "Do not ask questions. Return the result."
    ),
    tools=[
        FunctionTool(func=generate_scene_videos),
        FunctionTool(func=regenerate_scene_video),
    ],
)
