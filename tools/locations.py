"""
Tool — generate_location_images

Generates one reference image per unique location in the shot list (parallel).
Each image is saved as an ADK artifact for use as a continuity reference by
generate_scene_images. No GCS upload — these are internal reference images only.
State key: location_images  (dict: location_name → artifact_name)
"""

import asyncio
import logging
from typing import Dict, List

from google.adk.tools import ToolContext
from google.genai import types as gtypes

from film_generator_agent.utils.gemini import generate_best_image


def _location_prompt(location_name: str, description: str, visual_style: str) -> str:
    # "ZERO characters" must come FIRST — before the description — so the model
    # applies the constraint before reading any character references that may have
    # leaked into the location_description from scene design.
    return " ".join([
        "ZERO characters in this image — absolutely no people, animals, figures, silhouettes,",
        "or body parts of any kind. This is a pure environment/background shot only.",
        "If the description below mentions any character, outfit, or person, IGNORE it entirely.",
        f"ART STYLE: {visual_style}. Every element must be rendered in {visual_style} style.",
        f"Wide establishing shot of the location: {location_name.replace('_', ' ')}.",
        description,
        "Fill the entire frame with the environment. No text or watermarks.",
        f"REMINDER: {visual_style} style, ZERO characters.",
    ])


async def _generate_one_location(
    location_name: str,
    description: str,
    visual_style: str,
    tool_context: ToolContext,
) -> Dict:
    prompt = _location_prompt(location_name, description, visual_style)

    print(f"\n[locations:{location_name}] ── PROMPT ──────────────────────────────")
    print(f"[locations:{location_name}] raw description from shot_list: {description!r}")
    print(f"[locations:{location_name}] full prompt:\n{prompt}")
    print(f"[locations:{location_name}] ────────────────────────────────────────\n")

    result = await generate_best_image([prompt], prompt, num_attempts=3)

    if not result:
        logging.error("Location image generation failed: %s", location_name)
        return {"status": "failed", "location_name": location_name}

    artifact_name = f"location_{location_name}.png"
    await tool_context.save_artifact(
        artifact_name,
        gtypes.Part.from_bytes(data=result["image_bytes"], mime_type="image/png"),
    )
    logging.info("Location artifact saved: %s", artifact_name)
    return {"status": "success", "location_name": location_name, "artifact_name": artifact_name}


async def generate_location_images(tool_context: ToolContext) -> dict:
    """
    Generate a reference image for every unique location in the shot list (parallel).

    Saves each as artifact (location_{name}.png) and uploads to GCS.
    Stores location_images dict in state for use by generate_scene_images.
    Call alongside generate_character_portraits after design_scenes completes.
    """
    shot_list: Dict = tool_context.state.get("shot_list", {})
    if not shot_list:
        return {"status": "failed", "detail": "No shot_list in state. Run design_scenes first."}

    visual_style: str = tool_context.state.get("film_concept", {}).get("visual_style", "cinematic")

    # Collect unique locations in scene order; first-seen description wins
    seen: Dict[str, str] = {}
    for shot_id in sorted(shot_list, key=lambda x: int(x)):
        shot = shot_list[shot_id]
        loc_name = shot.get("location_name", "").strip()
        if loc_name and loc_name not in seen:
            seen[loc_name] = shot.get("location_description", "").strip()

    if not seen:
        return {"status": "skipped", "detail": "No location_name fields found in shot_list."}

    results = await asyncio.gather(*[
        _generate_one_location(name, desc, visual_style, tool_context)
        for name, desc in seen.items()
    ])

    successes: List[str] = []
    failures:  List[str] = []
    location_images: Dict[str, str] = {}

    for r in results:
        if r["status"] == "success":
            location_images[r["location_name"]] = r["artifact_name"]
            successes.append(r["location_name"])
        else:
            failures.append(r["location_name"])

    tool_context.state["location_images"] = location_images

    return {
        "status": "success" if successes else "failed",
        "generated": successes,
        "failed":    failures,
        "location_images": location_images,
        "detail": f"Generated {len(successes)} location image(s)." + (f" {len(failures)} failed." if failures else ""),
    }
