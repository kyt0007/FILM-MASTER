"""
Tool 3 — design_scenes
Converts each script scene into a detailed shot with camera, framing, and character positions.
One shot per scene (keeps it simple and production-ready).
"""

import json
import logging

from google import genai
from google.adk.tools import ToolContext
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SHOT_MODEL = "gemini-2.5-pro"

SHOT_PROMPT = """
You are a professional film director designing a shot list from a screenplay.

Script:
{script}

Characters:
{characters}

Film concept (for visual style reference):
{film_concept}

Guidelines:

VISUAL STYLE
- Every shot must reflect the film's visual_style consistently.
  Same colour palette, lighting mood, and rendering aesthetic across the entire film.
  Do not let individual shots drift in tone or colour — treat the whole shot list as
  one coherent visual world.

SHOTS
- Design exactly ONE primary shot per scene.
- visual_description must be 50+ words. Include: setting, lighting, colour palette,
  AND for each character present — their exact outfit, hair colour/style, and expression
  exactly as defined in the Characters list. Be pixel-precise about appearance so the
  image generator can maintain visual consistency across all shots.
- coarse_action must be < 20 words, no character names, only actions
  (e.g. "Two figures walk toward a glowing door. Wind moves tall grass.").
- shot_type must include both type AND speed: e.g. "wide establishing shot, gradual",
  "medium close-up, slow", "extreme close-up, rapid".
- camera_movement options: "static", "slow pan left/right", "dolly in/out",
  "tilt up/down", "handheld tracking", "aerial descending".

CHARACTERS
- involving_characters: only include characters who actually appear in this shot.
  Each entry MUST repeat the character's full canonical description from the Characters
  list (outfit, hair, features) — do not abbreviate. This anchors their look across shots.
- Keep involving_characters to max 2 per shot for clarity.
- dialogue: copy dialogue from the script for this scene.

NARRATIVE CONTINUITY
- Each shot must logically follow from the previous shot in setting, time of day,
  and character state. Read every prior shot before writing the next.
- If a character's state changed (injured, wet, carrying something), reflect it in
  every subsequent shot they appear in.
- Lighting and time-of-day must progress consistently — if dusk in shot 4, night by shot 6.
- Emotional arc: the emotion field should build across the film's three acts, not reset
  randomly between shots.

LOCATIONS
- Reuse the same location_name across shots that share a location so a single reference
  image can be generated and reused for visual consistency.
- location_description must be word-for-word identical for the same location_name across
  all shots — copy-paste it, do not paraphrase.
- location_description must describe ONLY the physical environment — architecture,
  lighting, atmosphere, colour palette, time of day. NEVER mention characters, outfits,
  props held by characters, or anything that belongs to a person. The location image
  is generated without any characters present.

Return a JSON object:
{{
  "shots": {{
    "1": {{
      "scene_number": 1,
      "location_name": "snake_case_location_id (e.g. ancient_temple, city_rooftop, forest_clearing)",
      "location_description": "30+ word description of the ENVIRONMENT ONLY — architecture, lighting, atmosphere, colour palette, time of day. No characters, outfits, or people.",
      "involving_characters": [
        {{"name": "Character_Name", "visual_description": "outfit, action, expression, position"}},
        ...
      ],
      "visual_description": "Detailed 50+ word description of the entire shot",
      "coarse_action": "< 20 words, actions only, no names",
      "emotion": "...",
      "shot_type": "type + speed",
      "camera_movement": "...",
      "dialogue": [
        {{"character": "...", "line": "..."}}
      ]
    }},
    "2": {{ ... }},
    ...
  }}
}}
"""


async def design_scenes(tool_context: ToolContext) -> dict:
    """
    Design a detailed shot for each scene in the script.

    Call this after `write_script` has been confirmed by the user.
    Reads script, characters, and film_concept from session state.
    """
    script = tool_context.state.get("script")
    characters = tool_context.state.get("characters", [])
    film_concept = tool_context.state.get("film_concept")

    if not script:
        return {"status": "failed", "detail": "No script in state. Run write_script first."}

    client = genai.Client()
    prompt = SHOT_PROMPT.format(
        script=json.dumps(script, indent=2),
        characters=json.dumps(characters, indent=2),
        film_concept=json.dumps(film_concept, indent=2),
    )

    try:
        response = client.models.generate_content(
            model=SHOT_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        shot_data = json.loads(response.text)
        tool_context.state["shot_list"] = shot_data.get("shots", {})
        logging.info("Shot list designed: %d shots", len(shot_data.get("shots", {})))
        return {
            "status": "success",
            "num_shots": len(shot_data.get("shots", {})),
            "shot_list": shot_data,
        }
    except (json.JSONDecodeError, ValueError) as e:
        logging.error("Failed to design scenes: %s", e)
        return {"status": "failed", "detail": str(e)}
