"""
src/kg/entity_generator.py
───────────────────────────
Given a domain and n, uses GPT-4o to generate a list of rare,
underrepresented entities suitable for knowledge graph construction.

The LLM is instructed to:
    - Think about what is visually rare and hard for T2I models to generate
    - Prioritise entities with unique morphological/visual features
    - Avoid well-known, common entities that are well-represented in training data
    - Generate an optimised Wikipedia search query per entity

Optionally accepts custom source URLs — if provided, the LLM biases
entity selection toward what those sources cover. Wikipedia is always
used as the primary scrape source regardless.

Output: list of entity dicts compatible with extractor.py
    [
        {
            "name": "Bleeding Tooth Fungus",
            "wiki_search": "Hydnellum peckii",
            "source_urls": []
        },
        ...
    ]
"""

import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


# ── Generation prompt ─────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert curator of rare, visually distinctive, and culturally
underrepresented concepts for a knowledge graph used to improve text-to-image
generation of long-tail concepts.

Your task: given a domain, generate a list of rare entities that:
1. Are genuinely rare or underrepresented in standard image datasets
2. Have unique, distinctive visual features that a diffusion model would
   typically get WRONG (defaulting to a generic prior)
3. Have enough documentation on Wikipedia or authoritative sources to
   allow structured attribute extraction
4. Are diverse — span different sub-categories within the domain
5. Are NOT well-known household names (avoid Zeus, Hanuman, Rose, Dog etc.)

For each entity also provide the best Wikipedia search query — usually the
scientific name, specific cultural term, or unambiguous title of the article.

Return ONLY a valid JSON array. No markdown, no explanation.

Format:
[
  {
    "name": "<entity common name>",
    "wiki_search": "<optimized Wikipedia search query>",
    "rarity_reason": "<one sentence: why this is rare/hard for T2I models>"
  }
]"""

_USER_TEMPLATE = """\
Domain: {domain}
Number of entities to generate: {n}
{source_context}

Generate {n} rare, visually distinctive entities from this domain.
Prioritise maximum visual rarity and uniqueness.
Each entity must be different enough from the others to add value to the graph."""


# ── Generator ─────────────────────────────────────────────────────────────────

class EntityGenerator:
    """
    Uses GPT-4o to generate rare entity lists for any domain.

    Usage:
        gen = EntityGenerator()
        entities = gen.generate(
            domain="biology",
            n=20,
            source_urls=["https://iucnredlist.org"]
        )
        # entities → [{"name": ..., "wiki_search": ..., "source_urls": [...]}]
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        domain:      str,
        n:           int,
        source_urls: list[str] | None = None,
    ) -> list[dict]:
        """
        Generate n rare entity dicts for the given domain.

        Args:
            domain      : domain string (e.g. "biology", "indian_mythology")
            n           : number of entities to generate
            source_urls : optional list of authoritative URLs to bias toward

        Returns:
            List of entity dicts with keys: name, wiki_search, source_urls
        """
        source_context = self._build_source_context(source_urls)

        user_msg = _USER_TEMPLATE.format(
            domain=domain,
            n=n,
            source_context=source_context,
        )

        logger.info(f"Generating {n} rare entities for domain='{domain}'...")

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.7,    # some creativity for diverse entity selection
            max_tokens=4096,
        )

        raw = (response.choices[0].message.content or "").strip()
        entities = self._parse_response(raw, domain, source_urls or [])

        logger.info(f"  Generated {len(entities)} entities.")
        for e in entities:
            logger.debug(f"    {e['name']} (wiki: {e['wiki_search']})")

        return entities

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_source_context(self, source_urls: list[str] | None) -> str:
        if not source_urls:
            return "Sources: Wikipedia (default)"
        url_list = "\n".join(f"  - {u}" for u in source_urls)
        return (
            f"Additional authoritative sources to draw entities from:\n{url_list}\n"
            "Bias entity selection toward what these sources cover, but Wikipedia "
            "will be used as the primary scrape source."
        )

    def _parse_response(
        self,
        raw:         str,
        domain:      str,
        source_urls: list[str],
    ) -> list[dict]:
        """
        Parse LLM JSON response into entity dicts.
        Adds source_urls field and removes rarity_reason (internal use only).
        """
        import re

        # Strip markdown fences if present
        cleaned = re.sub(r"```json|```", "", raw).strip()

        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(f"  JSON parse failed: {exc}")
            logger.debug(f"  Raw response: {raw[:500]}")
            return []

        entities = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name        = item.get("name", "").strip()
            wiki_search = item.get("wiki_search", name).strip()
            if not name:
                continue
            entities.append({
                "name":        name,
                "wiki_search": wiki_search,
                "source_urls": source_urls,
            })

        return entities
