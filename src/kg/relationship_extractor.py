"""
src/kg/relationship_extractor.py
──────────────────────────────────
Cross-entity relationship extraction pass.

After all entities in a domain are individually extracted, this module:
    1. Loads all extracted entity JSONs for the domain
    2. Sends the full entity name list to GPT-4o
    3. LLM identifies all meaningful typed relationships between them
    4. Creates/updates the relationship edges in Neo4j

This is a second pass — it runs AFTER individual extraction and loading,
so all nodes already exist. It enriches the graph with cross-entity edges
that individual extraction missed (since each entity is extracted in isolation).

Example output edges:
    (Yama) -[RULES_OVER]-> (Naraka)
    (Yama) -[HAS_ASSISTANT]-> (Chitragupta)
    (Chitragupta) -[SERVES]-> (Yama)
    (Rama) -[HAS_SPOUSE]-> (Sita)
    (Sita) -[HAS_SPOUSE]-> (Rama)
"""

import os
import json
import logging
from pathlib import Path

from openai import OpenAI
from src.kg.neo4j_client import Neo4jClient
from src.kg.loader import _merge_edge, _sanitize_rel_type, _infer_inverse

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "output"

_SYSTEM = """\
You are an expert knowledge graph builder. Given a list of named entities from
a specific domain, identify ALL meaningful typed relationships between them.

Rules:
- Only create relationships between entities that appear in the provided list
- Use specific, semantically meaningful relationship types in UPPERCASE_SNAKE_CASE
- Include both directions where appropriate (spouse, sibling, enemy)
- Relationship types should be domain-appropriate
- Include relationships that are culturally/scientifically well-established
- Each relationship must have a clear factual basis

Return ONLY a valid JSON array. No markdown, no explanation.

Format:
[
  {
    "from": "<entity name exactly as given>",
    "type": "RELATIONSHIP_TYPE",
    "to": "<entity name exactly as given>",
    "description": "<one line factual basis>"
  }
]"""


class RelationshipExtractor:
    """
    Extracts cross-entity relationships for all entities in a domain.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract_domain_relationships(
        self,
        domain: str,
        entity_names: list[str],
    ) -> list[dict]:
        """
        Ask GPT-4o to identify all relationships between a list of entities.

        Args:
            domain       : domain string for context
            entity_names : list of entity names already in the KG

        Returns:
            List of relationship dicts: [{from, type, to, description}]
        """
        if len(entity_names) < 2:
            return []

        entity_list = "\n".join(f"  - {name}" for name in entity_names)
        user_msg = (
            f"Domain: {domain}\n\n"
            f"Entities:\n{entity_list}\n\n"
            f"Identify all meaningful typed relationships between these entities."
        )

        logger.info(f"  Extracting cross-entity relationships for {len(entity_names)} entities...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                max_tokens=4096,
            )
            raw = (response.choices[0].message.content or "").strip()

            import re
            cleaned = re.sub(r"```json|```", "", raw).strip()
            relationships = json.loads(cleaned)
            logger.info(f"  Found {len(relationships)} cross-entity relationships")
            return relationships

        except Exception as e:
            logger.error(f"  Relationship extraction failed: {e}")
            return []

    def load_relationships(
        self,
        client: Neo4jClient,
        domain: str,
        relationships: list[dict],
        entity_names_set: set[str],
    ) -> None:
        """
        Load extracted relationships into Neo4j.
        Only creates edges between known entities — no stub nodes.
        """
        loaded = 0
        for rel in relationships:
            from_name = rel.get("from", "").strip()
            to_name   = rel.get("to",   "").strip()
            rel_type  = rel.get("type", "").strip()

            if not from_name or not to_name or not rel_type:
                continue

            # Only create edges between entities that exist in our KG
            if from_name not in entity_names_set or to_name not in entity_names_set:
                logger.debug(f"  Skipping edge — entity not in KG: {from_name} -> {to_name}")
                continue

            rel_type = _sanitize_rel_type(rel_type)
            inv_type = _infer_inverse(rel_type)

            _merge_edge(client, from_name, rel_type, to_name)
            _merge_edge(client, to_name, inv_type, from_name)
            loaded += 1

        logger.info(f"  Loaded {loaded} cross-entity relationship pairs")

    def run(self, client: Neo4jClient, domain: str) -> None:
        """
        Full pipeline: load entity names → extract relationships → load edges.

        Args:
            client : active Neo4jClient
            domain : domain to process
        """
        # Get all enriched entity names for this domain from Neo4j
        results = client.run(
            "MATCH (e:Entity) WHERE e.domain = $domain AND e.morphology IS NOT NULL "
            "RETURN e.name AS name",
            {"domain": domain}
        )
        entity_names = [r["name"] for r in results if r.get("name")]

        if len(entity_names) < 2:
            logger.warning(f"  Only {len(entity_names)} entities in '{domain}' — skipping relationship extraction")
            return

        logger.info(f"Running cross-entity relationship extraction for '{domain}' ({len(entity_names)} entities)")

        # Extract relationships via LLM
        relationships = self.extract_domain_relationships(domain, entity_names)

        if not relationships:
            logger.warning(f"  No relationships found for '{domain}'")
            return

        # Load into Neo4j
        entity_names_set = set(entity_names)
        self.load_relationships(client, domain, relationships, entity_names_set)

        logger.info(f"  Relationship pass complete for '{domain}'")
