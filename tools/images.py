"""
Tool — generate_scene_images / regenerate_scene_image

Generates one image per shot (parallel) using Gemini image generation.
Each scene is conditioned on character portrait artifacts and a location reference
artifact for visual continuity across shots.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from google.adk.tools import ToolContext

from film_generator_agent.utils.gemini import generate_best_image
from film_generator_agent.utils.storage import save_and_upload
from film_generator_agent.tools.characters import _normalize_name


def _scene_prompt(shot_info: Dict, visual_style: str, has_portraits: bool, has_location: bool) -> str:
    chars = shot_info.get("involving_characters", [])
    n = len(chars)
    parts = [
        f"ART STYLE: {visual_style}. Every element must be rendered in {visual_style} style.",
    ]

    # Character count is the most common failure — state it explicitly up front
    if n == 0:
        parts.append("NO characters in this scene.")
    else:
        names = " and ".join(f"'{c['name']}'" for c in chars)
        parts.append(
            f"EXACTLY {n} character{'s' if n > 1 else ''} in this scene: {names}. "
            f"Do NOT add or remove any characters."
        )

    if has_portraits:
        parts.append(
            f"CHARACTER REFERENCE IMAGES PROVIDED. Reproduce each character's face, hair, "
            f"outfit, and features EXACTLY as shown — in {visual_style} style."
        )

    # Character descriptions immediately after the count declaration
    for c in chars:
        parts.append(f"'{c['name']}' appearance: {c.get('visual_description', '')}.")

    if has_location:
        parts.append(
            f"LOCATION REFERENCE IMAGE PROVIDED. Scene must take place in this exact environment, "
            f"rendered in {visual_style} style."
        )

    parts += [
        f"Scene: {shot_info.get('visual_description', '')}",
        f"Action: {shot_info.get('coarse_action', '')}",
        f"Mood: {shot_info.get('emotion', '')}",
        "Fill the entire frame. No text, watermarks, or storyboard lines.",
        f"REMINDER: render everything in {visual_style} art style.",
    ]
    return " ".join(p for p in parts if p.strip())


async def _load_portrait_parts(
    chars: List[Dict],
    character_portraits: Dict[str, str],
    tool_context: ToolContext,
) -> list:
    parts = []
    for c in chars:
        artifact_name = character_portraits.get(_normalize_name(c["name"]))
        if not artifact_name:
            continue
        try:
            artifact = await tool_context.load_artifact(artifact_name)
            if artifact:
                parts.append(artifact)
        except Exception as e:
            logging.warning("Could not load portrait for %s: %s", c["name"], e)
    return parts


async def _generate_one_scene(
    shot_id: str,
    shot_info: Dict,
    visual_style: str,
    character_portraits: Dict[str, str],
    location_images: Dict[str, str],
    tool_context: ToolContext,
) -> Dict[str, Any]:
    chars = shot_info.get("involving_characters", [])
    portrait_parts = await _load_portrait_parts(chars, character_portraits, tool_context)

    location_part = None
    loc_artifact = location_images.get(shot_info.get("location_name", ""))
    if loc_artifact:
        try:
            location_part = await tool_context.load_artifact(loc_artifact)
        except Exception as e:
            logging.warning("Could not load location artifact %s: %s", loc_artifact, e)

    prompt = _scene_prompt(shot_info, visual_style, bool(portrait_parts), location_part is not None)
    contents = [prompt] + portrait_parts + ([location_part] if location_part else [])

    result = await generate_best_image(contents, prompt, num_attempts=3)
    if not result:
        logging.error("Scene image generation failed: shot %s", shot_id)
        return {"status": "failed", "shot_id": shot_id}

    title = tool_context.state.get("title", "film")
    artifact_name = f"scene_{shot_id}.png"
    blob_name = f"{tool_context.invocation_id}/{title}/{artifact_name}"

    uri = await save_and_upload(tool_context, result["image_bytes"], artifact_name, blob_name)

    scene_uris: Dict = tool_context.state.get("scene_image_uris", {})
    scene_uris[shot_id] = uri
    tool_context.state["scene_image_uris"] = scene_uris

    eval_decision = result["evaluation"].decision if result.get("evaluation") else "N/A"
    logging.info("Scene %s saved: %s (eval: %s)", shot_id, uri, eval_decision)
    return {"status": "success", "shot_id": shot_id, "uri": uri, "eval": eval_decision}


async def generate_scene_images(tool_context: ToolContext) -> dict:
    """
    Generate a high-quality image for every shot in the shot list (parallel).

    Conditions each image on character portrait artifacts and location reference
    images for visual continuity. Call after generate_character_portraits AND
    generate_location_images have completed.
    """
    shot_list: Dict = tool_context.state.get("shot_list", {})
    if not shot_list:
        return {"status": "failed", "detail": "No shot_list in state. Run design_scenes first."}

    visual_style = tool_context.state.get("film_concept", {}).get("visual_style", "cinematic")
    character_portraits: Dict = tool_context.state.get("character_portraits", {})
    location_images: Dict = tool_context.state.get("location_images", {})

    results = await asyncio.gather(*[
        _generate_one_scene(shot_id, shot_info, visual_style, character_portraits, location_images, tool_context)
        for shot_id, shot_info in sorted(shot_list.items(), key=lambda x: int(x[0]))
    ])

    successes = [r for r in results if r["status"] == "success"]
    failures  = [r for r in results if r["status"] == "failed"]
    return {
        "status": "success" if successes else "failed",
        "generated": [r["shot_id"] for r in successes],
        "failed":    [r["shot_id"] for r in failures],
        "scene_image_uris": tool_context.state.get("scene_image_uris", {}),
        "detail": f"Generated {len(successes)} scene image(s)." + (f" {len(failures)} failed." if failures else ""),
    }


async def regenerate_scene_image(
    shot_id: str,
    new_prompt: str,
    tool_context: ToolContext,
) -> dict:
    """
    Re-generate the image for a single scene.

    Args:
        shot_id: Shot identifier (e.g. "1", "3") as shown in the shot list.
        new_prompt: Updated visual description for the scene.
    """
    shot_list: Dict = tool_context.state.get("shot_list", {})
    shot_info = shot_list.get(shot_id)
    if not shot_info:
        return {"status": "failed", "detail": f"Shot '{shot_id}' not found in shot list."}

    shot_info = {**shot_info, "visual_description": new_prompt}
    visual_style = tool_context.state.get("film_concept", {}).get("visual_style", "cinematic")

    result = await _generate_one_scene(
        shot_id, shot_info, visual_style,
        tool_context.state.get("character_portraits", {}),
        tool_context.state.get("location_images", {}),
        tool_context,
    )
    if result["status"] == "success":
        return {"status": "success", "detail": f"Scene {shot_id} regenerated: {result['uri']}"}
    return {"status": "failed", "detail": f"Failed to regenerate scene {shot_id}."}
