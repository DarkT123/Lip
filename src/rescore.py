"""Optional LLM rescoring / cleanup on top of AVSR output.

AVSR models output a raw transcript — sometimes with disfluencies, missing
punctuation, or low-confidence n-best hypotheses. An LLM can:
  - Polish punctuation and capitalization
  - Pick the best of several n-best hypotheses (rescoring)
  - Apply domain-specific cleanup (e.g. user's vocabulary)

This is OPTIONAL. The AVSR model is the recognition system; the LLM is a
post-processor. Never let the LLM hallucinate content not in the n-best list
for the rescoring use case — constrain it.

Backends are intentionally swappable: Gemini today, Grok / MiniMax / a
fine-tuned model later. Only consider fine-tuning a provider as a *text*
post-processor; do not fine-tune them as the core recognition model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from google import genai

GEMINI_MODEL = "gemini-2.5-pro"


@dataclass
class RescoreResult:
    text: str
    backend: str


def polish(raw: str) -> RescoreResult:
    """Clean up punctuation/capitalization without changing the words."""
    prompt = (
        "Polish this transcript for grammar, capitalization, and punctuation. "
        "Do not change, add, or remove words. Return only the polished text.\n\n"
        f"TRANSCRIPT: {raw}"
    )
    text = _gemini_text(prompt)
    return RescoreResult(text=text, backend=GEMINI_MODEL)


def rescore_nbest(hypotheses: list[str]) -> RescoreResult:
    """Pick the most plausible hypothesis from an n-best list."""
    options = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(hypotheses))
    prompt = (
        "Below are n-best transcription hypotheses for a short utterance. "
        "Return the single most plausible English sentence verbatim — pick one "
        "of the listed options, do not invent new wording.\n\n"
        f"{options}"
    )
    text = _gemini_text(prompt)
    return RescoreResult(text=text, backend=GEMINI_MODEL)


def _gemini_text(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return (response.text or "").strip()
