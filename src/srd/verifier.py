"""
src/srd/verifier.py
────────────────────
VLM-based attribute verifier for the SRD module (Section 3.3).

For each generated image I_t, checks each attribute a_i ∈ A_ret
and returns:

    Binary vector   b_t ∈ {0,1}^N   (Eq. 4)
        b_{t,i} = 1  if attribute a_i is localised in I_t
                  0  otherwise

    Global Stability Index            (Eq. 5)
        GSI_t = (1/N) Σ b_{t,i}

Uses GPT-4o vision to perform all N binary checks in a single call.
"""

import base64
import io
import json
import logging
from dataclasses import dataclass

from PIL import Image
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


# ── Verification result ───────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """
    Output of a single VLM verification pass.

    binary_vector : {attribute → bool}   the b_t vector (Eq. 4)
    gsi           : float                GSI_t (Eq. 5)
    present       : attributes found in image
    missing       : A_miss — attributes NOT found (fed into refinement)
    """
    binary_vector: dict[str, bool]
    gsi:           float
    present:       list[str]
    missing:       list[str]

    @classmethod
    def from_checks(cls, checks: dict[str, bool]) -> "VerificationResult":
        present = [a for a, v in checks.items() if v]
        missing = [a for a, v in checks.items() if not v]
        # Eq. 5: GSI = (1/N) Σ b_{t,i}
        gsi = len(present) / len(checks) if checks else 0.0
        return cls(
            binary_vector=checks,
            gsi=gsi,
            present=present,
            missing=missing,
        )

    @classmethod
    def empty(cls) -> "VerificationResult":
        """Used when no attributes to verify — GSI = 1.0 (no corrections needed)."""
        return cls(binary_vector={}, gsi=1.0, present=[], missing=[])


# ── Verifier ──────────────────────────────────────────────────────────────────

class AttributeVerifier:
    """
    Verifies whether each attribute in A_ret is present in a generated image.

    Sends the image + attribute list to GPT-4o vision in a single call.
    Returns a VerificationResult containing b_t and GSI_t.

    Args:
        model : vision-capable OpenAI model (default gpt-4o)
    """

    _SYSTEM = """\
You are a precise visual attribute verifier for rare concept image generation.

Given an image and a list of expected visual attributes, determine whether
each attribute is clearly visible or represented in the image.

For each attribute respond with:
    true  — the attribute is clearly and unambiguously present
    false — the attribute is absent, unclear, or only partially present

Be strict — only mark true if the attribute is unambiguously present.
Partial presence counts as false.

Return ONLY a valid JSON object mapping each attribute string to a boolean.
No markdown, no explanation, just the JSON object.

Example:
{
  "dark complexion resembling storm clouds": true,
  "protruding fangs": false,
  "garland of flames": true
}"""

    _USER_TEMPLATE = """\
Check whether each of these {n} attributes is clearly present in the image:

{attr_list}

Return the JSON object now."""

    def __init__(self, model: str = "gpt-4o"):
        self.model  = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(self, image: Image.Image, attributes: list[str]) -> VerificationResult:
        """
        Run attribute verification on a PIL image.

        Implements AnalyzeImage(I_{k-1}, K) from Algorithm 1.

        Args:
            image      : generated PIL image I_t
            attributes : A_ret — list of attribute strings to check

        Returns:
            VerificationResult with binary_vector b_t and GSI_t
        """
        if not attributes:
            logger.warning("  Verifier: no attributes to check — returning GSI=1.0")
            return VerificationResult.empty()

        image_b64 = self._encode_image(image)
        attr_list  = "\n".join(f"- {a}" for a in attributes)

        user_msg = self._USER_TEMPLATE.format(
            n=len(attributes),
            attr_list=attr_list,
        )

        raw = self._call_vision(image_b64, user_msg)
        checks = self._parse_checks(raw, attributes)
        result = VerificationResult.from_checks(checks)

        logger.info(
            f"  Verifier: GSI={result.gsi:.3f} "
            f"({len(result.present)}/{len(attributes)} present)"
        )
        if result.missing:
            logger.debug(f"  Missing attrs: {result.missing}")

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _call_vision(self, image_b64: str, user_msg: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":    f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": user_msg},
                    ],
                },
            ],
            temperature=0,
            max_tokens=512,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _encode_image(image: Image.Image, max_size: int = 1024) -> str:
        """Resize if needed and encode to base64 PNG."""
        if max(image.size) > max_size:
            image = image.copy()
            image.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _parse_checks(raw: str, attributes: list[str]) -> dict[str, bool]:
        """
        Parse VLM JSON response into {attribute → bool}.
        Falls back to False for any attribute that can't be parsed.
        """
        import re
        # Strip any accidental markdown fences
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            # Match keys case-insensitively in case VLM slightly rewrites them
            data_lower = {k.lower(): v for k, v in data.items()}
            return {
                attr: bool(data_lower.get(attr.lower(), False))
                for attr in attributes
            }
        except json.JSONDecodeError:
            logger.warning(f"  Verifier: JSON parse failed — defaulting all to False")
            return {attr: False for attr in attributes}
