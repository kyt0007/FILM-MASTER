"""
Tool — generate_character_portraits / regenerate_character_portrait

Generates one Imagen-4.0-ultra portrait per character (parallel).
Falls back to Gemini image generation if Imagen fails.
Portraits are saved as ADK artifacts; consumed by scene image generation for continuity.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List

from google.adk.tools import ToolContext
from google.genai import types

IMAGEN_MODEL = "imagen-4.0-ultra-generate-001"

_imagen_model = None


def _normalize_name(name: str) -> str:
    """Canonical key for a character name — lowercase, spaces → underscores."""
    return name.strip().lower().replace(" ", "_")


def _get_imagen_model():
    global _imagen_model
    if _imagen_model is None:
        import vertexai
        from vertexai.preview.vision_models import ImageGenerationModel
        vertexai.init(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
        _imagen_model = ImageGenerationModel.from_pretrained(IMAGEN_MODEL)
    return _imagen_model


def _portrait_prompt(name: str, description: str, visual_style: str) -> str:
    return (
        f"ART STYLE: {visual_style}. Render this image entirely in {visual_style} style. "
        f"Full-body character portrait of a single person. "
        f"Subject: {name} — {description}. "
        f"The subject is the ONLY thing in the image. "
        f"Plain neutral studio background, no environment, no scenery, no props. "
        f"Front-facing reference pose, full body visible, high detail. "
        f"No text, watermarks, or multiple characters. "
        f"REMINDER: {visual_style} art style throughout."
    )


async def _generate_one_portrait(
    name: str,
    description: str,
    visual_style: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """Generate one portrait via Imagen, falling back to Gemini."""
    prompt = _portrait_prompt(name, description, visual_style)
    image_bytes = None

    # --- Attempt 1: Imagen-4.0-ultra ---
    try:
        response = await asyncio.to_thread(
            _get_imagen_model().generate_images,
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            person_generation="allow_adult",
            safety_filter_level="block_few",
            add_watermark=False,
        )
        if response.images:
            img = response.images[0]
            image_bytes = getattr(img, "image_bytes", None) or getattr(img, "_image_bytes", None)
    except Exception as e:
        logging.warning("Imagen failed for %s: %s", name, e)

    # --- Fallback: Gemini ---
    if not image_bytes:
        logging.info("Falling back to Gemini for %s", name)
        try:
            from film_generator_agent.utils.gemini import generate_best_image
            result = await generate_best_image([prompt], prompt, num_attempts=2)
            if result:
                image_bytes = result.get("image_bytes")
        except Exception as e:
            logging.error("Gemini fallback failed for %s: %s", name, e)

    if not image_bytes:
        logging.error("Both Imagen and Gemini failed for %s", name)
        return {"status": "failed", "name": name, "prompt": prompt}

    artifact_name = f"character_{_normalize_name(name)}.png"
    await tool_context.save_artifact(
        artifact_name,
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    )
    logging.info("Portrait saved: %s → %s", name, artifact_name)
    return {"status": "success", "name": name, "artifact": artifact_name}


async def generate_character_portraits(tool_context: ToolContext) -> dict:
    """
    Generate a reference portrait for every character using Imagen-4.0-ultra.

    Portraits are saved as ADK artifacts (character_{name}.png) and stored in
    state["character_portraits"] for use by generate_scene_images.
    Call this after design_scenes is confirmed.
    """
    characters: List[Dict] = tool_context.state.get("characters", [])
    visual_style: str = tool_context.state.get("film_concept", {}).get("visual_style", "cinematic")

    if not characters:
        return {"status": "skipped", "detail": "No characters defined. Proceeding without portraits."}

    results = await asyncio.gather(*[
        _generate_one_portrait(
            c["name"],
            c.get("description", c.get("visual_description", "")),
            visual_style,
            tool_context,
        )
        for c in characters
    ])

    # Single retry pass for failures
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        retry_results = await asyncio.gather(*[
            _generate_one_portrait(r["name"], r["prompt"], visual_style, tool_context)
            for r in failed
        ])
        results = [r for r in results if r["status"] == "success"] + retry_results

    successes = [r for r in results if r["status"] == "success"]
    failures  = [r for r in results if r["status"] == "failed"]

    # Key by normalized name so lookups in images.py are case/space insensitive
    tool_context.state["character_portraits"] = {
        _normalize_name(r["name"]): r["artifact"] for r in successes
    }

    return {
        "status": "success" if successes else "failed",
        "generated": [r["name"] for r in successes],
        "failed":    [r["name"] for r in failures],
        "detail": f"Generated {len(successes)} portrait(s)." + (f" {len(failures)} failed." if failures else ""),
    }


async def regenerate_character_portrait(
    name: str,
    new_description: str,
    tool_context: ToolContext,
) -> dict:
    """
    Re-generate the portrait for a single character.

    Args:
        name: Character's exact name as listed in state["characters"].
        new_description: Updated visual description.
    """
    visual_style = tool_context.state.get("film_concept", {}).get("visual_style", "cinematic")
    result = await _generate_one_portrait(name, new_description, visual_style, tool_context)

    if result["status"] == "success":
        tool_context.state.setdefault("character_portraits", {})[_normalize_name(name)] = result["artifact"]
        for c in tool_context.state.get("characters", []):
            if c["name"] == name:
                c["description"] = new_description
                break
        return {"status": "success", "detail": f"Portrait for '{name}' regenerated: {result['artifact']}"}

    return {"status": "failed", "detail": f"Failed to regenerate portrait for '{name}'."}
