"""Shared GCS upload helper for image tools."""

import logging
import os

from dotenv import load_dotenv
from google.adk.tools import ToolContext
from google.cloud import storage
from google.genai import types as gtypes

load_dotenv()

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "")


def upload_to_gcs(image_bytes: bytes, blob_name: str) -> str:
    """Upload image bytes to GCS; returns gs:// URI."""
    client = storage.Client()
    client.bucket(GCS_BUCKET).blob(blob_name).upload_from_string(
        image_bytes, content_type="image/png"
    )
    return f"gs://{GCS_BUCKET}/{blob_name}"


async def save_and_upload(
    tool_context: ToolContext,
    image_bytes: bytes,
    artifact_name: str,
    blob_name: str,
) -> str:
    """Save image as ADK artifact and upload to GCS.

    Returns the GCS URI, or the artifact_name as fallback if the upload fails.
    """
    await tool_context.save_artifact(
        artifact_name,
        gtypes.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    )
    try:
        gcs_uri = upload_to_gcs(image_bytes, blob_name)
        logging.info("Uploaded %s → %s", artifact_name, gcs_uri)
        return gcs_uri
    except Exception as e:
        logging.error("GCS upload failed for %s: %s", artifact_name, e)
        return artifact_name  # fallback so callers always get a usable reference
