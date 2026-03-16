"""
Tool 2 — write_script
Turns a FilmConcept into a scene-by-scene script with dialogue.
"""

import json
import logging

from google import genai
from google.adk.tools import ToolContext
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_MODEL = "gemini-2.5-pro"

SCRIPT_PROMPT = """
You are a professional screenwriter. Turn the following film concept into a
detailed, scene-by-scene script.

Film concept:
{film_concept}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Create as many scenes as the story needs (typically 6–14).
  Follow the 3-act structure from the film concept: act_1 → rising action → climax → act_3 resolution.
  Map the film concept's act descriptions directly to scenes — do not invent a different story.
- Each scene must take place in a SINGLE, continuous location — no cuts within a scene.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NARRATIVE COHERENCE  ← read this carefully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before writing each scene, re-read all previous scenes. Then ask yourself:
  1. Does this scene's plot logically follow from the previous scene's ending?
  2. Is the time of day / location transition plausible given the last scene?
  3. Are all character states (injuries, emotions, knowledge, relationships) consistent?
     A character who learned a secret in scene 3 cannot act ignorant in scene 5.
     A character who was injured in scene 4 shows it in scene 5.
  4. Does the emotional_tone escalate coherently toward the climax, then resolve?

- plot must explicitly reference what happened in the previous scene when relevant.
  Use phrases like "Following their confrontation in the temple..." to show the chain.
- Avoid reset scenes — scenes that ignore prior events and feel like a fresh start.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIALOGUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Every scene must have dialogue for every character present, OR a NARRATOR line
  if no characters are in the scene. Never leave dialogue empty.
- Each character must have a consistent, recognisable voice throughout the entire script.
  Decide their vocabulary, speech rhythm, and register upfront and maintain it.
  A formal character does not suddenly speak in slang. A quiet character does not monologue.
- Dialogue must advance the plot or reveal character — no filler lines.
- Lines must directly react to what was said or done in the same scene or prior scenes.
  Characters do not ignore bombshells; they respond, deflect, or escalate.
- Subtext is preferred over on-the-nose exposition.
  Show tension through what is NOT said as much as what is.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHARACTERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Characters list must cover ALL speaking or visible characters across all scenes.
  Use underscores for multi-word names (e.g. "Young_Alice").
- Character descriptions must be highly specific (exact outfit colours, hair, distinguishing
  features) — these are used as portrait generation prompts and must stay consistent across
  every scene the character appears in.
- A character's appearance does NOT change between scenes unless the script explicitly
  shows them changing clothes, ageing, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- visual_description must be 50+ words: setting, lighting, colour palette, time of day,
  and what each character is doing. Be cinematic and specific.
- Lighting and time-of-day must progress logically across scenes.
  If scene 4 is dusk, scene 6 cannot be midday unless time has passed in the story.

Return a JSON object with exactly this structure:
{{
  "scenes": {{
    "1": {{
      "scene_number": 1,
      "title": "...",
      "plot": "One paragraph. Reference prior events where relevant.",
      "visual_description": "50+ word cinematic description: setting, light, colour, character actions.",
      "dialogue": [
        {{"character": "Character_Name_or_NARRATOR", "line": "..."}},
        ...
      ],
      "emotional_tone": "...",
      "cinematography": "Camera technique description"
    }},
    "2": {{ ... }},
    ...
  }},
  "characters": [
    {{"name": "Character_Name", "description": "Highly specific visual description: age, gender, exact outfit colours, hair colour and style, distinguishing features."}},
    ...
  ]
}}
"""


async def write_script(tool_context: ToolContext) -> dict:
    """
    Generate a full scene-by-scene script from the film concept in session state.

    Call this after `brainstorm_film_concept` has been confirmed by the user.
    The film_concept must already be stored in session state.
    """
    film_concept = tool_context.state.get("film_concept")
    if not film_concept:
        return {"status": "failed", "detail": "No film_concept in state. Run brainstorm_film_concept first."}

    client = genai.Client()
    prompt = SCRIPT_PROMPT.format(film_concept=json.dumps(film_concept, indent=2))

    try:
        response = client.models.generate_content(
            model=SCRIPT_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        script_data = json.loads(response.text)
        tool_context.state["script"] = script_data.get("scenes", {})
        tool_context.state["characters"] = script_data.get("characters", [])
        logging.info(
            "Script written: %d scenes, %d characters",
            len(script_data.get("scenes", {})),
            len(script_data.get("characters", [])),
        )
        return {
            "status": "success",
            "num_scenes": len(script_data.get("scenes", {})),
            "num_characters": len(script_data.get("characters", [])),
            "script": script_data,
        }
    except (json.JSONDecodeError, ValueError) as e:
        logging.error("Failed to write script: %s", e)
        return {"status": "failed", "detail": str(e)}
