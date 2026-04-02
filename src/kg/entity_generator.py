"""
src/kg/entity_generator.py
───────────────────────────
Given a domain and n, uses GPT-4o to generate a list of rare,
underrepresented entities suitable for knowledge graph construction.
"""

import json
import logging
import math
import os
import re

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

For each entity also provide the best Wikipedia search query.

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

        # Single batch (up to ~30)
        entities = gen.generate(domain="biology", n=20)

        # Large batch with automatic chunking (recommended for n > 30)
        entities = gen.generate_large(domain="biology", n=100)
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        domain:      str,
        n:           int,
        source_urls: list[str] | None = None,
    ) -> list[dict]:
        """Generate n rare entity dicts for the given domain in one API call."""
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
            temperature=0.7,
            max_tokens=16000,
        )

        raw      = (response.choices[0].message.content or "").strip()
        entities = self._parse_response(raw, domain, source_urls or [])

        logger.info(f"  Generated {len(entities)} entities.")
        return entities

    def generate_large(
        self,
        domain:      str,
        n:           int,
        source_urls: list[str] | None = None,
        batch_size:  int = 25,
    ) -> list[dict]:
        """
        Generate n entities in batches to avoid token limits.
        Deduplicates across batches by name.
        Use this instead of generate() when n > 30.
        """
        all_entities: list[dict] = []
        seen_names:   set[str]   = set()
        n_batches = math.ceil(n / batch_size)

        logger.info(f"Generating {n} entities for '{domain}' in {n_batches} batches of {batch_size}...")

        for i in range(n_batches):
            remaining  = n - len(all_entities)
            this_batch = min(batch_size, remaining)
            if this_batch <= 0:
                break

            logger.info(f"  Batch {i+1}/{n_batches} ({this_batch} entities)...")
            batch = self.generate(domain=domain, n=this_batch, source_urls=source_urls)

            for entity in batch:
                name = entity.get("name", "").strip()
                if name and name not in seen_names:
                    all_entities.append(entity)
                    seen_names.add(name)

        logger.info(f"  Total unique entities generated: {len(all_entities)}")
        return all_entities

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
        """Parse LLM JSON response into entity dicts."""
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


# ── Domain source URLs ────────────────────────────────────────────────────────

DOMAIN_SOURCES: dict[str, list[str]] = {
    "indian_mythology": [
        "https://www.wisdomlib.org",
        "https://www.sacred-texts.com/hin/index.htm",
        "https://hinduism.stackexchange.com",
        "https://www.hindupedia.com",
    ],
    "greek_mythology": [
        "https://www.theoi.com",
        "https://www.perseus.tufts.edu",
        "https://www.sacred-texts.com/cla/index.htm",
        "https://mythopedia.com/topics/greek-mythology",
    ],
    "chinese_mythology": [
        "https://www.sacred-texts.com/cfu/index.htm",
        "https://mythology.net/chinese",
        "https://www.chinaknowledge.de/Literature/Mythology/mythology.html",
    ],
    "literary": [
        "https://www.gutenberg.org",
        "https://www.sacred-texts.com",
        "https://en.wikisource.org",
        "https://www.poetryfoundation.org",
    ],
    "biology": [
        "https://www.iucnredlist.org",
        "https://eol.org",
        "https://www.gbif.org",
        "https://animaldiversity.org",
        "https://www.inaturalist.org",
    ],
    "natural_phenomena": [
        "https://www.usgs.gov",
        "https://www.noaa.gov",
        "https://earthobservatory.nasa.gov",
        "https://www.nationalgeographic.com/science",
    ],
    "cultural_artifact": [
        "https://www.metmuseum.org/art/collection",
        "https://www.britishmuseum.org/collection",
        "https://www.clevelandart.org/art/collection",
        "https://artsandculture.google.com",
        "https://www.si.edu/collections",
    ],
}


def get_domain_sources(domain: str) -> list[str]:
    """Return curated free public source URLs for a domain."""
    if domain in DOMAIN_SOURCES:
        return DOMAIN_SOURCES[domain]
    for key in DOMAIN_SOURCES:
        if key.split("_")[0] in domain or domain.split("_")[0] in key:
            return DOMAIN_SOURCES[key]
    return []
