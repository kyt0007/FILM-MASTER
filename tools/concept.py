"""
Tool 1 — brainstorm_film_concept
Expands the user's rough idea into a structured FilmConcept.
"""

import json
import logging

from google import genai
from google.adk.tools import ToolContext
from google.genai import types

from film_generator_agent.utils.evaluate import EVALUATION_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CONCEPT_MODEL = "gemini-2.5-flash"

CONCEPT_PROMPT = """
You are a creative film producer. Expand the user's rough idea into a structured film concept.

User idea: {user_idea}
Preferred genre (optional): {genre}
Preferred mood (optional): {mood}
Preferred visual style (optional): {style}

Rules:
- visual_style must be NON-REALISTIC (anime, cartoon, stylised, painterly, etc.)
  if the film features characters.
- visual_style can be photorealistic ONLY for scenery/documentary films without characters.
- Be descriptive and specific in every field.
- title: short, evocative, words connected with underscores (e.g. "Ember_Rising").

Return a JSON object with exactly these keys:
{{
  "title": "...",
  "genre": "...",
  "visual_style": "...",
  "mood": "...",
  "setting": "...",
  "act_1": "...",
  "act_2": "...",
  "act_3": "...",
  "sound": "..."
}}
"""


async def brainstorm_film_concept(
    user_idea: str,
    tool_context: ToolContext,
    genre: str = "",
    mood: str = "",
    style: str = "",
) -> dict:
    """
    Expand the user's rough idea into a structured film concept.

    Args:
        user_idea: The user's description of what they want the film to be about.
        tool_context: ADK ToolContext.
        genre: Optional preferred genre (e.g. "sci-fi", "romance", "thriller").
        mood: Optional preferred mood (e.g. "dark", "hopeful", "whimsical").
        style: Optional preferred visual style (e.g. "anime", "watercolour").
    """
    client = genai.Client()
    prompt = CONCEPT_PROMPT.format(
        user_idea=user_idea,
        genre=genre or "not specified",
        mood=mood or "not specified",
        style=style or "not specified",
    )
    try:
        response = client.models.generate_content(
            model=CONCEPT_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )
        concept = json.loads(response.text)
        # Normalise title: replace spaces with underscores
        concept["title"] = concept.get("title", "Untitled_Film").replace(" ", "_")
        tool_context.state["film_concept"] = concept
        tool_context.state["title"] = concept["title"]
        logging.info("Film concept generated: %s", concept["title"])
        return {"status": "success", "film_concept": concept}
    except (json.JSONDecodeError, ValueError) as e:
        logging.error("Failed to generate film concept: %s", e)
        return {"status": "failed", "detail": str(e)}
