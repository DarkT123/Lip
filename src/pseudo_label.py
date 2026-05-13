"""Bootstrap pseudo-labeling via a multimodal LLM.

This is NOT the recognition model. It produces noisy initial transcripts
that the user reviews and corrects. The corrected text is the real label.

Currently uses Gemini 2.5 Pro (video-native). Replace or augment with other
providers (Grok, MiniMax) by adding sibling functions and switching on
`PSEUDO_LABEL_BACKEND`.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai

GEMINI_MODEL = "gemini-2.5-pro"
PROMPT_VERSION = "v1"

PROMPTS = {
    "v1": (
        "This is a short video of someone silently mouthing a phrase. "
        "Read their lips and return only the phrase as a single well-formed, "
        "grammatically polished English sentence with correct capitalization "
        "and punctuation. If you cannot make out the lips, return the single "
        "word UNCLEAR. Do not add any commentary."
    ),
}


@dataclass
class PseudoLabel:
    text: str
    model: str
    prompt_version: str


def label(video_path: Path) -> PseudoLabel:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Put it in .env.")

    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(video_path))
    while uploaded.state.name == "PROCESSING":
        time.sleep(1)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Video upload failed: state={uploaded.state.name}")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[uploaded, PROMPTS[PROMPT_VERSION]],
    )
    return PseudoLabel(
        text=(response.text or "").strip(),
        model=GEMINI_MODEL,
        prompt_version=PROMPT_VERSION,
    )
