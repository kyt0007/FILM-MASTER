"""
Tool 8 — assemble_film
Downloads all scene videos from GCS and concatenates them into the final film.
Veo-3.0 generates audio (dialogue + ambient) natively in each clip, so no
additional audio mixing is needed.

  - Output: single MP4 uploaded to GCS + saved as ADK artifact.
"""

import asyncio
import datetime
import logging
import os
import tempfile
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions as api_exceptions, retry as api_retry
from google.adk.tools import ToolContext
from google.cloud import storage

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VIDEO_CODEC  = "libx264"
GCS_BUCKET   = os.getenv("GCS_BUCKET_NAME", "")
_GCS_RETRIES = api_retry.Retry(
    predicate=api_retry.if_transient_error,
    initial=1.0,
    maximum=30.0,
    multiplier=2.0,
    deadline=120.0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Split gs://bucket/blob into (bucket, blob)."""
    without_scheme = gcs_uri.removeprefix("gs://")
    bucket_name, _, blob_name = without_scheme.partition("/")
    return bucket_name, blob_name


def _gcs_download(gcs_uri: str, local_path: str) -> bool:
    """Download a GCS object to a local path with automatic retry on transient errors."""
    try:
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)

        if not blob.exists():
            logging.error("GCS object not found: %s", gcs_uri)
            return False

        blob.download_to_filename(local_path, retry=_GCS_RETRIES)
        logging.info("Downloaded %s → %s (%d bytes)", gcs_uri, local_path, os.path.getsize(local_path))
        return True
    except api_exceptions.NotFound:
        logging.error("GCS object not found: %s", gcs_uri)
    except api_exceptions.Forbidden:
        logging.error("Permission denied downloading %s — check service account roles", gcs_uri)
    except Exception as e:
        logging.error("GCS download failed for %s: %s: %s", gcs_uri, type(e).__name__, e)
    return False


def _gcs_upload(local_path: str, blob_name: str) -> str:
    """Upload a local file to GCS and return gs:// URI."""
    client = storage.Client()
    blob = client.bucket(GCS_BUCKET).blob(blob_name)
    blob.upload_from_filename(local_path, content_type="video/mp4")
    return f"gs://{GCS_BUCKET}/{blob_name}"


# ---------------------------------------------------------------------------
# Core assembly logic
# ---------------------------------------------------------------------------

async def _download_scene_video(
    shot_id: str,
    video_gcs_uri: str,
    temp_dir: str,
) -> Optional[str]:
    """Download one scene video from GCS. Returns local path, or None on failure."""
    local_path = os.path.join(temp_dir, f"scene_{shot_id}_raw.mp4")
    ok = await asyncio.to_thread(_gcs_download, video_gcs_uri, local_path)
    return local_path if ok else None


def _reencode_scene_clip(shot_id: str, video_local: str, temp_dir: str) -> Optional[str]:
    """Re-encode one downloaded scene clip to normalise codec/container. Returns output path or None."""
    from moviepy import VideoFileClip  # lazy
    try:
        clip = VideoFileClip(video_local)
        out_path = os.path.join(temp_dir, f"scene_{shot_id}_assembled.mp4")
        clip.write_videofile(out_path, codec=VIDEO_CODEC, audio_codec="aac", logger=None)
        clip.close()
        return out_path
    except Exception as e:
        logging.error("Scene %s encode failed: %s", shot_id, e)
        return None


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

async def assemble_film(tool_context: ToolContext) -> dict:
    """
    Assemble the final film by concatenating all Veo-3.0 scene videos.

    Veo generates audio natively in each clip — no mixing needed.
    Downloads all clips from GCS in parallel, re-encodes to normalise
    codec/container, then concatenates and uploads.

    Reads from session state: video_uris, video_uri_map.
    """
    from moviepy import VideoFileClip, concatenate_videoclips  # lazy

    ordered_video_uris: List[str] = tool_context.state.get("video_uris", [])
    video_uri_map: Dict[str, str] = tool_context.state.get("video_uri_map", {})
    title: str                    = tool_context.state.get("title", "film")
    session_id: str               = tool_context.invocation_id

    if not ordered_video_uris:
        return {"status": "failed", "detail": "No video_uris in state. Run generate_scene_videos first."}

    # Ordered shot IDs from video_uri_map (sorted numerically)
    ordered_shot_ids = sorted(video_uri_map.keys(), key=int)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download all videos in parallel (I/O-bound — safe to parallelize)
        download_tasks = [
            _download_scene_video(shot_id, video_uri_map[shot_id], temp_dir)
            for shot_id in ordered_shot_ids
            if video_uri_map.get(shot_id)
        ]
        downloaded: List[Optional[str]] = await asyncio.gather(*download_tasks)

        failed_downloads = [
            ordered_shot_ids[i] for i, p in enumerate(downloaded) if p is None
        ]
        if failed_downloads:
            logging.warning("Download failed for scene(s): %s", failed_downloads)

        # Re-encode sequentially to avoid OOM
        assembled_paths: List[str] = []
        for shot_id, video_local in zip(ordered_shot_ids, downloaded):
            if video_local is None:
                continue
            path = await asyncio.to_thread(
                _reencode_scene_clip, shot_id, video_local, temp_dir
            )
            if path:
                assembled_paths.append(path)

        if not assembled_paths:
            return {"status": "failed", "detail": "No scene clips could be assembled."}

        # Concatenate all scene clips
        clips = [VideoFileClip(p) for p in assembled_paths]
        final = concatenate_videoclips(clips, method="compose")
        now = datetime.datetime.now(datetime.timezone.utc)
        film_filename = f"film_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
        final_local = os.path.join(temp_dir, film_filename)
        final.write_videofile(final_local, codec=VIDEO_CODEC, audio_codec="aac", logger=None)
        for c in clips:
            c.close()
        final.close()

        # Upload to GCS
        blob_name = f"{session_id}/{title}/{film_filename}"
        gcs_uri = await asyncio.to_thread(_gcs_upload, final_local, blob_name)

        # Save as ADK artifact
        with open(final_local, "rb") as f:
            film_bytes = f.read()
        await tool_context.save_artifact(
            film_filename,
            genai.types.Part.from_bytes(data=film_bytes, mime_type="video/mp4"),
        )

        tool_context.state["final_film_uri"] = gcs_uri
        tool_context.state["final_film_artifact"] = film_filename

        logging.info("Final film: %s", gcs_uri)
        return {
            "status": "success",
            "num_scenes": len(assembled_paths),
            "gcs_uri": gcs_uri,
            "artifact_name": film_filename,
            "detail": f"Film assembled from {len(assembled_paths)} scene(s) and uploaded to GCS.",
        }
