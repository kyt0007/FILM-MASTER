"""
Gemini image generation helper — adapted from content_gen_agent/utils/gemini_utils.py.

Generates an image and evaluates it in one call. Used by both character portrait
generation (as a quality gate) and scene image generation.
"""

import asyncio
import logging
from typing import TypedDict

from google import genai, auth
from google.api_core import exceptions as api_exceptions
from google.genai import types
from google.genai.types import HarmBlockThreshold, HarmCategory, Modality

from film_generator_agent.utils.evaluate import EvalResult, evaluate_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IMAGE_MIME_TYPE = "image/png"
# Must be a Gemini model that supports image output via generate_content().
# Imagen models (imagen-3.0-*) use generate_images() and cannot be called
# with response_modalities — using them here would return no image data.
GEMINI_IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"

SAFETY_SETTINGS = [
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,        threshold=HarmBlockThreshold.OFF),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,   threshold=HarmBlockThreshold.OFF),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,   threshold=HarmBlockThreshold.OFF),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,          threshold=HarmBlockThreshold.OFF),
]


class ImageResult(TypedDict):
    image_bytes: bytes
    evaluation: EvalResult | None
    mime_type: str


def get_client() -> genai.Client | None:
    try:
        return genai.Client()
    except (auth.exceptions.DefaultCredentialsError, ValueError) as e:
        logging.error("Failed to init Gemini client: %s", e)
        return None


async def generate_and_evaluate_image(
    client: genai.Client,
    contents: list,
    prompt: str,
) -> ImageResult | None:
    """
    Generate one image with Gemini and evaluate it.

    Args:
        client: initialised genai.Client
        contents: list of str / types.Part for the generation call
        prompt: the original text prompt (used for evaluation)
    """
    print(f"  [gemini] generate_and_evaluate_image — prompt: {prompt[:100]}...")
    try:
        # response_modalities MUST use the Modality enum — plain strings are
        # not accepted by gemini-2.0-flash-preview-image-generation.
        # Note: types.ImageConfig does not exist in google-genai 1.x — aspect
        # ratio is controlled by the prompt text instead.
        config = types.GenerateContentConfig(
            response_modalities=[Modality.TEXT, Modality.IMAGE],
            safety_settings=SAFETY_SETTINGS,
            # Lower temperature → more literal prompt adherence, less random variation.
            # 0.7 balances consistency with creative rendering quality.
            temperature=0.7,
            # top_p / top_k tighten the token distribution further.
            top_p=0.9,
            top_k=32,
        )
        print(f"  [gemini] calling {GEMINI_IMAGE_MODEL} with {len(contents)} content part(s)...")
        response = await client.aio.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=contents,
            config=config,
        )
        candidates = response.candidates or []
        print(f"  [gemini] response received — candidates={len(candidates)}")
        if candidates and candidates[0].content and candidates[0].content.parts:
            parts = candidates[0].content.parts
            print(f"  [gemini] parts in response: {len(parts)} — types: {[type(p).__name__ for p in parts]}")
            for i, part in enumerate(parts):
                has_data = bool(part.inline_data and part.inline_data.data)
                print(f"  [gemini] part[{i}]: inline_data={has_data}  mime={getattr(part.inline_data, 'mime_type', None) if part.inline_data else None}")
                if has_data:
                    image_bytes = part.inline_data.data
                    print(f"  [gemini] image extracted: {len(image_bytes)} bytes")
                    evaluation = await evaluate_image(image_bytes, prompt)
                    return {
                        "image_bytes": image_bytes,
                        "evaluation": evaluation,
                        "mime_type": IMAGE_MIME_TYPE,
                    }
        print(f"  [gemini] WARNING: no image parts found in response")
        if candidates and candidates[0].finish_reason:
            print(f"  [gemini] finish_reason={candidates[0].finish_reason}")
        logging.warning("Gemini returned no image parts for prompt: %s", prompt[:80])
    except (
        api_exceptions.GoogleAPICallError,
        api_exceptions.ResourceExhausted,
        ValueError,
        Exception,  # catch-all so exceptions are logged, not silently dropped
    ) as e:
        print(f"  [gemini] EXCEPTION: {type(e).__name__}: {e}")
        logging.error("Gemini image generation failed: %s: %s", type(e).__name__, e)
    return None


async def generate_best_image(
    contents: list,
    prompt: str,
    num_attempts: int = 4,
    min_score: int = 18,
    max_rounds: int = 3,
) -> ImageResult | None:
    """
    Generate `num_attempts` images in parallel and return the highest-scoring one.

    If the best result scores below `min_score`, runs another round of `num_attempts`
    until a satisfactory result is found or `max_rounds` rounds are exhausted.

    Args:
        num_attempts: Parallel attempts per round. Default 4 gives more candidates.
        min_score:    Minimum acceptable score (0–24). Default 18 = decision Pass (10)
                      + subject_adherence (3) + attribute_matching (3) + style (2).
                      Set to 0 to disable threshold retry.
        max_rounds:   Maximum rounds before accepting best seen. Default 3 = up to 12 calls.
    """
    from film_generator_agent.utils.evaluate import score  # lazy to avoid circular import

    client = get_client()
    if not client:
        return None

    best_overall: ImageResult | None = None

    for round_num in range(1, max_rounds + 1):
        logging.info("Image generation round %d/%d — %d attempt(s)", round_num, max_rounds, num_attempts)
        results = await asyncio.gather(*[
            generate_and_evaluate_image(client, contents, prompt)
            for _ in range(num_attempts)
        ], return_exceptions=True)

        valid = [r for r in results if isinstance(r, dict) and r]
        if not valid:
            logging.warning("Round %d: all %d attempts failed.", round_num, num_attempts)
            continue

        best = max(valid, key=lambda r: score(r.get("evaluation")))
        best_score = score(best.get("evaluation"))
        best_decision = best["evaluation"].decision if best.get("evaluation") else "N/A"
        logging.info("Round %d best: %d/22  decision: %s", round_num, best_score, best_decision)

        if best_overall is None or best_score > score(best_overall.get("evaluation")):
            best_overall = best

        if best_score >= min_score:
            return best_overall  # good enough — stop early

    if best_overall is None:
        logging.error("All %d round(s) failed to produce any image.", max_rounds)
    return best_overall
