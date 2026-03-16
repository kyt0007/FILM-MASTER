"""
media_gen_agent — ParallelAgent

Runs character portrait generation and location image generation simultaneously.
Wrap with AgentTool in the root agent so the LLM can invoke both with a single call.
"""

from google.adk.agents import ParallelAgent

from film_generator_agent.subagents.character_portrait_agent import character_portrait_agent
from film_generator_agent.subagents.location_image_agent import location_image_agent

media_gen_agent = ParallelAgent(
    name="media_gen_agent",
    description=(
        "Generates character portraits AND location reference images in parallel. "
        "Call this once after design_scenes is complete."
    ),
    sub_agents=[character_portrait_agent, location_image_agent],
)
