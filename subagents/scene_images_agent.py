"""Sub-agent: scene_images_agent — generates all scene images and stores URIs in state."""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from film_generator_agent.tools.images import generate_scene_images, regenerate_scene_image

scene_images_agent = Agent(
    name="scene_images_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a scene image generation sub-agent. "
        "When invoked, immediately call generate_scene_images() with no arguments. "
        "If some scenes failed, call regenerate_scene_image(shot_id, new_prompt) for each failed scene. "
        "Store scene_image_uris in state for the next step. Do not ask questions."
    ),
    tools=[
        FunctionTool(func=generate_scene_images),
        FunctionTool(func=regenerate_scene_image),
    ],
)
