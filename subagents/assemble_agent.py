"""Sub-agent: assemble_agent — assembles the final film from videos and audio in state."""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from film_generator_agent.tools.assemble import assemble_film

assemble_agent = Agent(
    name="assemble_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a film assembly sub-agent. "
        "When invoked, immediately call assemble_film() with no arguments. "
        "It reads video_uris and bg_music_artifact from session state automatically. "
        "Return the final GCS URI when done. Do not ask questions."
    ),
    tools=[FunctionTool(func=assemble_film)],
)
