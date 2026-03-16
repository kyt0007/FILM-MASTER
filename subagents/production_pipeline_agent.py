"""
production_pipeline_agent — SequentialAgent

Runs the full production pipeline in order:
  1. scene_images_agent  — generates scene images, writes scene_image_uris to state
  2. video_agent         — Veo-3.0 videos with embedded audio, reads scene_image_uris,
                           writes video_uris
  3. assemble_agent      — downloads videos and concatenates into final MP4
                           reads video_uris, writes final_film_uri

State flows automatically between steps — each agent reads what the previous wrote.
"""

from google.adk.agents import SequentialAgent

from film_generator_agent.subagents.scene_images_agent import scene_images_agent
from film_generator_agent.subagents.video_agent        import video_agent
from film_generator_agent.subagents.assemble_agent     import assemble_agent

production_pipeline_agent = SequentialAgent(
    name="production_pipeline_agent",
    description=(
        "Runs the full production pipeline: "
        "scene images → Veo-3.0 videos (with embedded audio) → final film assembly. "
        "Call this once after portraits and location images are approved."
    ),
    sub_agents=[scene_images_agent, video_agent, assemble_agent],
)
