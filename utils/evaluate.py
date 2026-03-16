"""
Image quality evaluation — adapted from content_gen_agent/utils/evaluate_media.py.

Generates N images, evaluates each with a structured EvalResult, picks the best.
"""

import logging
from typing import Literal

from google import genai
from google.api_core import exceptions as api_exceptions
from google.genai import types
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EVALUATION_MODEL = "gemini-2.5-flash"

IMAGE_EVALUATION_PROMPT = """
You are a strict Quality Assurance specialist for a film production pipeline.
Evaluate the provided image against this generation prompt:

PROMPT: "{prompt}"

Score each dimension as Pass or Fail:
- subject_adherence:     Does the image contain EXACTLY the subjects described — correct
                         count, correct identities, no extra or missing characters/objects?
- attribute_matching:    Do ALL visual attributes match exactly — colours, clothing, hair,
                         expressions, props, and distinguishing features as specified?
- spatial_accuracy:      Are positions, poses, and spatial relationships correct?
- style_fidelity:        Does the visual style match the requested art style precisely?
- quality_and_coherence: Is the image sharp, well-composed, and free of artefacts or
                         distorted faces?
- no_storyboard:         Does the image look like a finished scene (not a sketch/storyboard)?

Rules:
- subject_adherence FAILS if the wrong number of characters appears, even by one.
- attribute_matching FAILS if any named character's outfit, hair colour, or key feature
  differs from the prompt.
- Set the top-level "decision" to "Pass" ONLY if every dimension above passes.
- Provide a concise "reason" listing every specific failure.
"""


class EvalResult(BaseModel):
    decision: Literal["Pass", "Fail"]
    reason: str
    subject_adherence: Literal["Pass", "Fail"]
    attribute_matching: Literal["Pass", "Fail"]
    spatial_accuracy: Literal["Pass", "Fail"]
    style_fidelity: Literal["Pass", "Fail"]
    quality_and_coherence: Literal["Pass", "Fail"]
    no_storyboard: Literal["Pass", "Fail"]


async def evaluate_image(image_bytes: bytes, prompt: str) -> EvalResult | None:
    """Evaluate a generated image against its prompt using Gemini."""
    print(f"  [evaluate] evaluating image ({len(image_bytes)} bytes) against prompt: {prompt[:60]}...")
    try:
        client = genai.Client()
        response = await client.aio.models.generate_content(
            model=EVALUATION_MODEL,
            contents=[
                IMAGE_EVALUATION_PROMPT.format(prompt=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvalResult,
                # Thinking budget gives the evaluator time to examine each
                # attribute carefully before committing to Pass/Fail.
                thinking_config=types.ThinkingConfig(thinking_budget=512),
            ),
        )
        result = response.parsed
        print(f"  [evaluate] decision={result.decision}  score={score(result)}/22  reason={result.reason}")
        logging.info("Image evaluation: %s — %s", result.decision, result.reason)
        return result
    except (api_exceptions.GoogleAPICallError, ValueError) as e:
        print(f"  [evaluate] FAILED: {type(e).__name__}: {e}")
        logging.error("Image evaluation failed: %s", e)
        return None


def score(eval_result: EvalResult | None) -> int:
    """Convert EvalResult to a numeric score (0–24). Higher = better.

    Weights reflect which failures hurt film continuity most:
      decision             10  — top-level gate
      subject_adherence     3  — wrong character count destroys continuity
      attribute_matching    3  — wrong costume/colour breaks character consistency
      spatial_accuracy      2
      style_fidelity        2
      quality_and_coherence 2
      no_storyboard         2
                           ──
                           24  max
    """
    if not eval_result:
        return 0
    weights = {
        "decision":             10,
        "subject_adherence":     3,
        "attribute_matching":    3,
        "spatial_accuracy":      2,
        "style_fidelity":        2,
        "quality_and_coherence": 2,
        "no_storyboard":         2,
    }
    return sum(w for f, w in weights.items() if getattr(eval_result, f) == "Pass")
