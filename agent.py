"""
film_generator_agent — Conversational film production assistant.

Architecture: single LlmAgent (follows content_gen_agent pattern).
All pipeline stages are FunctionTools; the LLM orchestrates them conversationally,
pausing for user confirmation between every stage.

Production pipeline
───────────────────
PRE-PRODUCTION  (requires user approval before production starts)
  1. brainstorm_film_concept   — expand idea → FilmConcept (title/genre/style/acts/sound)
  2. write_script              — FilmConcept → scene-by-scene script with dialogue
  3. design_scenes             — script → detailed shot list (camera/framing/action)
  4. media_gen_agent [ParallelAgent]
     ├─ generate_character_portraits — Imagen-4.0-ultra character portraits
     └─ generate_location_images     — one reference image per unique location
     └─ regenerate_character_portrait  — fix one portrait without redoing all

  ⏸ APPROVAL GATE — agent asks user to review and approve portraits + location images.

PRODUCTION  (runs only after user gives approval)
  production_pipeline_agent [SequentialAgent]
    5. scene_images_agent      — Gemini scene images (conditioned on portraits + locations)
       └─ regenerate_scene_image — fix one scene image (root agent tool)
    6. video_agent             — Veo-3.0 clips with embedded dialogue + ambient audio
       └─ regenerate_scene_video — fix one scene video (root agent tool)
    7. assemble_agent          — downloads videos, concatenates, uploads final MP4
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool, load_artifacts
from google.adk.tools.agent_tool import AgentTool

from film_generator_agent.tools.concept       import brainstorm_film_concept
from film_generator_agent.tools.script        import write_script
from film_generator_agent.tools.scene_design  import design_scenes
from film_generator_agent.tools.characters    import regenerate_character_portrait
from film_generator_agent.tools.images        import (
    generate_scene_images,
    regenerate_scene_image,
)
from film_generator_agent.tools.images        import regenerate_scene_image
from film_generator_agent.tools.video         import regenerate_scene_video
from film_generator_agent.subagents           import media_gen_agent, production_pipeline_agent

INSTRUCTION = """
You are a creative film director assistant guiding the user through producing a short film.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW — follow this order strictly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Film Concept  ⏸ REQUIRES USER APPROVAL
  Tool: brainstorm_film_concept(user_idea, genre, mood, style)
  Ask the user for their idea, preferred genre, mood, and visual style.
  Present the FilmConcept output (title, genre, style, acts, sound).
  Ask if they want any changes. Do NOT proceed until the user approves.

STEP 2 — Script  (run automatically after Step 1 is approved)
  Tool: write_script()
  Call immediately. Do not pause or inform the user — proceed directly to Step 3.

STEP 3 — Scene Design  (run automatically after Step 2)
  Tool: design_scenes()
  Call immediately. Do not pause or inform the user — proceed directly to Step 4.

STEP 4 — Character Portraits + Location Images  ⏸ REQUIRES USER APPROVAL
  Tool: media_gen_agent (single call — runs portraits AND location images in parallel)
  Call media_gen_agent() once. It will simultaneously generate all character portraits
  and all location reference images.
  List each character portrait and each location image artifact name.
  Ask: "Are you happy with the portraits and location images, or would you like to change any?
  Tell me the character name or location name and how you'd like them to look."
  Tool for character changes: regenerate_character_portrait(name, new_description)
  Do NOT proceed to Step 5 until the user approves portraits and location images.

STEP 5-8 — Production Pipeline  (run automatically after Step 4 is approved)
  Tool: production_pipeline_agent (single call — runs the full production in sequence)
  Warn the user this will take several minutes, then call immediately.

  The pipeline runs automatically in this order:
    Step 5 — Scene Images:  generates all scene images; URIs stored in state.
    Step 6 — Videos: Veo-3.0 clips with embedded audio (dialogue + ambient).
    Step 7 — Assembly: concatenates all clips, uploads final MP4 to GCS.

  Each step reads state written by the previous step — no manual handoff needed.
  Report the final GCS URI of the completed film when done.

  If the user wants to change a specific scene image or video after the pipeline:
    Tool: regenerate_scene_image(shot_id, new_prompt) — redo one scene image
    Tool: regenerate_scene_video(shot_id) — redo one scene video
  Then re-run production_pipeline_agent to reassemble with the updated assets.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL GUIDANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Keep responses concise. Show structured output (lists, short tables).
- Always tell the user which step they are on (e.g. "Step 3 of 8 — Scene Design").
- Only pause for user input at Step 1 (concept approval) and Step 4 (portraits + locations approval).
- Steps are numbered 1–7 to the user.
- At all other steps, call the tool immediately and report results.
- If the user asks to change something mid-pipeline, use the appropriate regeneration tool.
- If a tool fails, report the error clearly and offer to retry.
- Do not skip steps or re-order them.
- Never generate children in any prompt.
"""

root_agent = Agent(
    name="film_generator_agent",
    model="gemini-2.5-flash",
    instruction=INSTRUCTION,
    tools=[
        load_artifacts,
        # Pre-production
        FunctionTool(func=brainstorm_film_concept),
        FunctionTool(func=write_script),
        FunctionTool(func=design_scenes),
        AgentTool(agent=media_gen_agent),          # runs portraits + locations in parallel
        FunctionTool(func=regenerate_character_portrait),
        # Production pipeline (scene images → videos+audio → assembly)
        AgentTool(agent=production_pipeline_agent),
        FunctionTool(func=regenerate_scene_image),  # manual fix for one scene image
        FunctionTool(func=regenerate_scene_video),  # manual fix for one scene video
    ],
)
