"""
Tool 6 — generate_scene_videos
Generates one Veo-3.0 video clip per scene, image-conditioned on the scene image.
Dialogue from the shot design is passed as an audio hint so Veo generates matching speech.
Video URIs are stored in session state (ordered) for the assemble step.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from google import genai
from google.adk.tools import ToolContext
from google.genai import types

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VEO_MODEL        = "veo-3.0-generate-preview"
VIDEO_DURATION   = 8    # seconds per scene clip
POLL_INTERVAL    = 15   # seconds between operation status checks
MAX_CONCURRENT   = 3    # max parallel Veo requests (quota guard)


def _get_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def _build_video_prompt(shot_info: Dict, visual_style: str) -> str:
    """Compose a detailed Veo prompt from shot design data."""
    parts = [
        f"Cinematic {visual_style} scene.",
        f"Shot: {shot_info.get('shot_type', 'medium shot')}.",
        f"Camera: {shot_info.get('camera_movement', 'static')}.",
        shot_info.get("visual_description", ""),
        f"Action: {shot_info.get('coarse_action', '')}",
        f"Emotion: {shot_info.get('emotion', '')}.",
    ]
    # Append dialogue as audio hint — Veo-3.0 will generate matching speech
    dialogue = shot_info.get("dialogue", [])
    if dialogue:
        lines = "\n".join(f"{d['character']}: {d['line']}" for d in dialogue)
        parts.append(f"\n\nAudio / Dialogue:\n{lines}")
    return " ".join(p for p in parts if p.strip())


async def _generate_one_video(
    shot_id: str,
    shot_info: Dict,
    image_gcs_uri: str,
    visual_style: str,
    title: str,
    session_id: str,
    bucket: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Generate one scene video with Veo-3.0, image-conditioned."""
    async with semaphore:
        prompt = _build_video_prompt(shot_info, visual_style)
        output_gcs_prefix = f"gs://{bucket}/{session_id}/{title}/scene_{shot_id}"

        print(f"\n[video:scene_{shot_id}] ── START ──────────────────────────────")
        print(f"[video:scene_{shot_id}] model={VEO_MODEL}")
        print(f"[video:scene_{shot_id}] output_prefix={output_gcs_prefix}")
        print(f"[video:scene_{shot_id}] image_gcs_uri={image_gcs_uri!r}")
        print(f"[video:scene_{shot_id}] prompt={prompt[:120]}...")

        config = types.GenerateVideosConfig(
            aspect_ratio="16:9",
            output_gcs_uri=output_gcs_prefix,
            number_of_videos=1,
            duration_seconds=VIDEO_DURATION,
            person_generation="allow_adult",
            enhance_prompt=True,
            generate_audio=True,
        )
        try:
            client = _get_client()

            # Only pass image if it's a real GCS URI — artifact names will cause API errors
            use_image = image_gcs_uri.startswith("gs://")
            if not use_image and image_gcs_uri:
                print(f"[video:scene_{shot_id}] WARNING: image_gcs_uri is not gs:// — generating without image")

            print(f"[video:scene_{shot_id}] calling generate_videos (use_image={use_image})...")
            try:
                # generate_videos is synchronous — run in thread to avoid blocking the event loop
                if use_image:
                    operation = await asyncio.to_thread(
                        client.models.generate_videos,
                        model=VEO_MODEL,
                        prompt=prompt,
                        image=types.Image(gcs_uri=image_gcs_uri, mime_type="image/png"),
                        config=config,
                    )
                else:
                    operation = await asyncio.to_thread(
                        client.models.generate_videos,
                        model=VEO_MODEL,
                        prompt=prompt,
                        config=config,
                    )
            except Exception as e:
                print(f"[video:scene_{shot_id}] generate_videos FAILED: {type(e).__name__}: {e}")
                logging.error("Scene %s: generate_videos raised %s: %s", shot_id, type(e).__name__, e)
                return {"status": "failed", "shot_id": shot_id, "error": str(e)}

            print(f"[video:scene_{shot_id}] operation started: name={getattr(operation, 'name', 'N/A')} done={operation.done}")

            # Poll until complete — also run in thread so sleep/poll doesn't block
            poll = 0
            while not operation.done:
                poll += 1
                await asyncio.sleep(POLL_INTERVAL)
                try:
                    operation = await asyncio.to_thread(client.operations.get, operation)
                except Exception as e:
                    print(f"[video:scene_{shot_id}] poll #{poll} operations.get failed: {type(e).__name__}: {e}")
                    logging.warning("Scene %s poll %d failed: %s", shot_id, poll, e)
                    continue
                print(f"[video:scene_{shot_id}] poll #{poll}: done={operation.done}")

            print(f"[video:scene_{shot_id}] operation complete — inspecting result...")

            # Check for operation-level error
            op_error = getattr(operation, "error", None)
            if op_error and getattr(op_error, "code", None):
                print(f"[video:scene_{shot_id}] OPERATION ERROR: code={op_error.code} message={op_error.message}")
                logging.error("Scene %s: Veo error code=%s: %s", shot_id, op_error.code, op_error.message)
                return {"status": "failed", "shot_id": shot_id, "error": op_error.message}

            # Inspect result structure before accessing
            op_result = getattr(operation, "result", None)
            print(f"[video:scene_{shot_id}] operation.result type={type(op_result).__name__}")
            generated = getattr(op_result, "generated_videos", None) if op_result else None
            print(f"[video:scene_{shot_id}] generated_videos={generated}")

            try:
                video_uri = generated[0].video.uri
            except Exception as e:
                print(f"[video:scene_{shot_id}] could not read video URI: {type(e).__name__}: {e}")
                print(f"[video:scene_{shot_id}] full operation dump: {operation}")
                logging.error("Scene %s: result parse failed — %s: %s", shot_id, type(e).__name__, e)
                return {"status": "failed", "shot_id": shot_id, "error": str(e)}

            print(f"[video:scene_{shot_id}] SUCCESS → {video_uri}")
            logging.info("Scene %s video: %s", shot_id, video_uri)
            return {"status": "success", "shot_id": shot_id, "video_uri": video_uri}

        except Exception as e:
            print(f"[video:scene_{shot_id}] UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")
            logging.error("Scene %s: %s: %s", shot_id, type(e).__name__, e)
            return {"status": "failed", "shot_id": shot_id, "error": str(e)}


def _update_video_state(tool_context: ToolContext, new_results: List[Dict]) -> None:
    """Merge new successful results into state, preserving existing entries."""
    video_uri_map: Dict[str, str] = tool_context.state.get("video_uri_map", {})
    for r in new_results:
        if r["status"] == "success":
            video_uri_map[r["shot_id"]] = r["video_uri"]
    # Rebuild ordered list from sorted map
    ordered = [video_uri_map[k] for k in sorted(video_uri_map, key=int)]
    tool_context.state["video_uri_map"] = video_uri_map
    tool_context.state["video_uris"] = ordered


async def generate_scene_videos(tool_context: ToolContext) -> dict:
    """
    Generate one Veo-3.0 video clip for every scene, conditioned on its scene image.

    Dialogue from the shot design is embedded in the prompt so Veo generates
    matching in-video audio. All scene videos are generated concurrently, with one
    automatic retry pass for any failures.
    Video GCS URIs are stored in session state (ordered by shot_id) for assembly.

    Call this after generate_scene_images has completed.
    """
    shot_list: Dict = tool_context.state.get("shot_list", {})
    scene_image_uris: Dict = tool_context.state.get("scene_image_uris", {})
    visual_style: str = tool_context.state.get("film_concept", {}).get("visual_style", "cinematic")
    title: str = tool_context.state.get("title", "film")
    session_id: str = tool_context.invocation_id
    bucket: str = os.getenv("GCS_BUCKET_NAME", "")

    print(f"\n[video] ── generate_scene_videos ──────────────────────────────")
    print(f"[video] model={VEO_MODEL}  duration={VIDEO_DURATION}s  max_concurrent={MAX_CONCURRENT}")
    print(f"[video] title={title!r}  session_id={session_id!r}  bucket={bucket!r}")
    print(f"[video] visual_style={visual_style!r}")
    print(f"[video] shot_list keys: {sorted(shot_list.keys(), key=int) if shot_list else '(empty)'}")
    print(f"[video] scene_image_uris: {scene_image_uris}")
    for sid in sorted(shot_list.keys(), key=int) if shot_list else []:
        uri = scene_image_uris.get(sid, "")
        is_gcs = uri.startswith("gs://")
        print(f"[video]   scene {sid}: image_uri={uri!r}  is_gcs={is_gcs}")
    print(f"[video] ────────────────────────────────────────────────────────\n")

    if not shot_list:
        return {"status": "failed", "detail": "No shot_list in state."}

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    common = dict(visual_style=visual_style, title=title, session_id=session_id, bucket=bucket, semaphore=sem)

    results = await asyncio.gather(*[
        _generate_one_video(sid, shot_list[sid], scene_image_uris.get(sid, ""), **common)
        for sid in sorted(shot_list, key=int)
    ])

    # One automatic retry for failures
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        logging.info("Retrying %d failed video(s): %s", len(failed), [r["shot_id"] for r in failed])
        retry_results = await asyncio.gather(*[
            _generate_one_video(r["shot_id"], shot_list[r["shot_id"]], scene_image_uris.get(r["shot_id"], ""), **common)
            for r in failed
        ])
        results = [r for r in results if r["status"] == "success"] + list(retry_results)

    _update_video_state(tool_context, results)

    successes = [r for r in results if r["status"] == "success"]
    failures  = [r for r in results if r["status"] == "failed"]
    return {
        "status": "success" if successes else "failed",
        "generated": [r["shot_id"] for r in successes],
        "failed":    [r["shot_id"] for r in failures],
        "video_uris": tool_context.state.get("video_uris", []),
        "detail": (
            f"Generated {len(successes)} video clip(s) with embedded audio."
            + (f" {len(failures)} failed: scenes {[r['shot_id'] for r in failures]}." if failures else "")
        ),
    }


async def regenerate_scene_video(
    shot_id: str,
    tool_context: ToolContext,
) -> dict:
    """
    Re-generate the video for a single failed or unsatisfactory scene.

    Args:
        shot_id: Shot identifier (e.g. "1", "2") as shown in the shot list.
    """
    shot_list: Dict = tool_context.state.get("shot_list", {})
    shot_info = shot_list.get(shot_id)
    if not shot_info:
        return {"status": "failed", "detail": f"Shot '{shot_id}' not found in shot list."}

    result = await _generate_one_video(
        shot_id,
        shot_info,
        tool_context.state.get("scene_image_uris", {}).get(shot_id, ""),
        visual_style=tool_context.state.get("film_concept", {}).get("visual_style", "cinematic"),
        title=tool_context.state.get("title", "film"),
        session_id=tool_context.invocation_id,
        bucket=os.getenv("GCS_BUCKET_NAME", ""),
        semaphore=asyncio.Semaphore(1),
    )
    if result["status"] == "success":
        _update_video_state(tool_context, [result])
        return {"status": "success", "detail": f"Scene {shot_id} video regenerated: {result['video_uri']}"}
    return {"status": "failed", "detail": f"Scene {shot_id} video generation failed. Try again or skip this scene."}
