"""
src/kg/retriever.py
────────────────────
Retrieves a context-rich subgraph from Neo4j for a given user prompt.

Matching strategy (three-tier):
    Tier 1 — Exact / case-insensitive / alternative_names match in Neo4j
    Tier 2 — Token overlap (word-level)
    Tier 3 — LLM semantic resolution against full KG name list

Relational query handling:
    "Ram's wife"        → anchor=Rama, traverse HAS_SPOUSE → fetch Sita
    "Yama's mount"      → anchor=Yama, traverse RIDES → fetch Nandi/etc
    "Rama and Sita"     → fetch both nodes directly
    "show me two headed bird" → semantic resolve → Ganda Bherunda
"""

import os
import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI
from src.kg.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# ── Context Packet ────────────────────────────────────────────────────────────

@dataclass
class ContextPacket:
    query:                   str
    domain:                  str
    primary_entities:        list[dict] = field(default_factory=list)
    neighbour_entities:      list[dict] = field(default_factory=list)
    relationships:           list[dict] = field(default_factory=list)
    retrieved_attributes:    list[str]  = field(default_factory=list)
    contrastive_constraints: list[str]  = field(default_factory=list)

    @property
    def all_entities(self) -> list[dict]:
        seen, result = set(), []
        for e in self.primary_entities + self.neighbour_entities:
            name = e.get("name", "")
            if name and name not in seen:
                seen.add(name)
                result.append(e)
        return result

    def is_empty(self) -> bool:
        return len(self.primary_entities) == 0


# ── Entity Extractor ──────────────────────────────────────────────────────────

class EntityExtractor:
    """
    GPT-4o extracts entity names AND relational intent from any prompt.

    Handles:
        - Any case, phrasing, possessives, partial names
        - Descriptive references ("the Hindu god of death" → "Yama")
        - Relational queries ("Ram's wife" → anchor=Rama, relation=HAS_SPOUSE)
        - Multi-entity ("Rama and Sita" → ["Rama", "Sita"])
    """

    _SYSTEM = """\
You extract entity names and optional relational intent from a text-to-image prompt.
The entity could be a mythological figure, rare animal, plant, artifact, or phenomenon.

Rules:
- Extract the most specific proper name(s) mentioned
- Handle any capitalization (YAMA, yama, Yama → "Yama")
- For descriptive references ("the Hindu god of death") resolve to the proper name
- For relational queries ("Ram's wife", "Yama's mount", "Krishna's enemy")
  extract BOTH the anchor entity AND a relationship type in UPPERCASE_SNAKE_CASE
- If the prompt is very long, focus only on the first sentence

Return ONLY valid JSON:
{
  "entities": ["EntityName1", "EntityName2"],
  "relational_intent": {
    "anchor": "AnchorEntityName",
    "relation": "RELATIONSHIP_TYPE"
  }
}
Set relational_intent to null if no relational query detected.
Relationship type examples: HAS_SPOUSE, RIDES, HAS_CHILD, HAS_PARENT,
HAS_SIBLING, ENEMY_OF, ALLY_OF, WIELDS, RULES_OVER, TEACHER_OF, STUDENT_OF
No explanation, no markdown, just the JSON."""

    def __init__(self, client: OpenAI):
        self.client = client

    def extract(self, prompt: str) -> tuple[list[str], dict | None]:
        """
        Returns (entity_names, relational_intent).
        relational_intent = {"anchor": str, "relation": str} or None
        """
        extract_from = prompt[:200] if len(prompt) > 200 else prompt

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user",   "content": extract_from},
            ],
            temperature=0,
            max_tokens=150,
        )
        raw = (response.choices[0].message.content or "{}").strip()

        # Handle both old format (array) and new format (object)
        import re
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                # Old format fallback
                return [n.strip() for n in result if isinstance(n, str)], None
            entities = [n.strip() for n in result.get("entities", [])
                       if isinstance(n, str) and n.strip()]
            relational = result.get("relational_intent")
            if relational and (not relational.get("anchor") or not relational.get("relation")):
                relational = None
            return entities, relational
        except json.JSONDecodeError:
            logger.warning(f"  Extractor returned non-JSON: {raw}")
            return [], None


# ── LLM Semantic Resolver ─────────────────────────────────────────────────────

class SemanticResolver:
    """
    When string matching fails, uses GPT-4o to pick the best KG node.
    """

    _SYSTEM = """\
You match a user-provided entity name to the best entry in a knowledge graph.
Given a name extracted from a user prompt and a list of KG entity names,
return the single best matching KG entity name, or "NONE" if nothing fits.
Return ONLY the entity name string — no explanation, no JSON wrapper."""

    def __init__(self, client: OpenAI):
        self.client = client

    def resolve(self, extracted_name: str, kg_names: list[str]) -> str | None:
        if not kg_names:
            return None

        kg_list  = "\n".join(f"  - {n}" for n in kg_names)
        user_msg = (
            f"Extracted name: \"{extracted_name}\"\n\n"
            f"Knowledge graph entities:\n{kg_list}\n\n"
            f"Best match (or NONE):"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_tokens=30,
        )
        result = (response.choices[0].message.content or "").strip()
        if result == "NONE" or not result:
            return None
        result_lower = result.lower()
        for name in kg_names:
            if name.lower() == result_lower:
                return name
        for name in kg_names:
            if result_lower in name.lower() or name.lower() in result_lower:
                return name
        return None


# ── Retriever ─────────────────────────────────────────────────────────────────

class KGRetriever:
    """
    Three-tier entity matching + relational graph traversal.

    When a relational query is detected (e.g. "Ram's wife"):
        1. Match the anchor entity (Rama)
        2. Traverse the KG via the relationship type (HAS_SPOUSE)
        3. Fetch the target node (Sita) and return both
    """

    def __init__(self, client: Neo4jClient, k: int = 1, max_neighbours: int = 10):
        self.client         = client
        self.k              = k
        self.max_neighbours = max_neighbours

        llm_client           = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._llm_client     = llm_client  # shared for traversal LLM calls
        self.extractor       = EntityExtractor(llm_client)
        self.resolver        = SemanticResolver(llm_client)
        self._kg_names: list[str] = self._load_all_kg_names()
        logger.info(f"  KG name cache: {len(self._kg_names)} enriched entities loaded")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, prompt: str) -> ContextPacket:
        logger.info(f"Retrieving context for: '{prompt}'")

        # Step 1 — extract entity names + relational intent
        entity_names, relational_intent = self.extractor.extract(prompt)
        logger.info(f"  Extracted: {entity_names}, relational: {relational_intent}")

        if not entity_names and not relational_intent:
            logger.warning("  No entities extracted.")
            return ContextPacket(query=prompt, domain="")

        # Step 2 — match nodes (with optional relational traversal)
        primary = self._match_with_relations(entity_names, relational_intent)

        if not primary:
            logger.warning(f"  No KG nodes matched.")
            return ContextPacket(query=prompt, domain="")

        # Step 3 — k-hop neighbours
        neighbours = self._expand_khop(primary)

        # Step 4 — relationships between primary nodes
        relationships = self._fetch_relationships(primary)

        # Step 5 — attributes + contrastive
        attrs       = self._build_attribute_list(primary)
        contrastive = self._build_contrastive(primary)
        domain      = primary[0].get("domain", "")

        ctx = ContextPacket(
            query=prompt,
            domain=domain,
            primary_entities=primary,
            neighbour_entities=neighbours,
            relationships=relationships,
            retrieved_attributes=attrs,
            contrastive_constraints=contrastive,
        )

        logger.info(
            f"  ContextPacket: {len(primary)} primary, "
            f"{len(neighbours)} neighbours, "
            f"{len(attrs)} attributes, "
            f"{len(relationships)} relationships"
        )
        return ctx

    # ── Relational matching ───────────────────────────────────────────────────

    def _match_with_relations(
        self,
        entity_names: list[str],
        relational_intent: dict | None,
    ) -> list[dict]:
        """
        Match entities, optionally traversing relational edges.

        If relational_intent is given (e.g. anchor=Rama, relation=HAS_SPOUSE):
            1. Match the anchor
            2. Traverse the edge to find the target
            3. Return both anchor + target nodes
        """
        matched, seen = [], set()

        # Direct entity matches
        for name in entity_names:
            node = self._resolve_node(name)
            if node:
                node_name = node.get("name", "")
                if node_name not in seen:
                    matched.append(node)
                    seen.add(node_name)
                    logger.info(f"  Matched '{name}' → '{node_name}'")
            else:
                logger.warning(f"  No match found for '{name}'")

        # Relational traversal
        if relational_intent:
            anchor_name = relational_intent.get("anchor", "")
            relation    = relational_intent.get("relation", "")

            if anchor_name and relation:
                # Resolve anchor
                anchor_node = self._resolve_node(anchor_name)
                if anchor_node:
                    anchor_kgname = anchor_node.get("name", "")
                    if anchor_kgname not in seen:
                        matched.append(anchor_node)
                        seen.add(anchor_kgname)
                        logger.info(f"  Matched anchor '{anchor_name}' → '{anchor_kgname}'")

                    # Traverse the relationship edge
                    targets = self._traverse_relation(anchor_kgname, relation)
                    for target in targets:
                        t_name = target.get("name", "")
                        if t_name and t_name not in seen:
                            matched.append(target)
                            seen.add(t_name)
                            logger.info(f"  Traversed {relation} → '{t_name}'")

        return matched

    def _traverse_relation(self, anchor_name: str, relation_type: str) -> list[dict]:
        """
        Traverse a typed relationship edge from anchor node.

        Strategy:
            1. Get all actual edges for this entity from Neo4j
            2. Use LLM to pick the best matching edge type for the intent
            3. Traverse that edge
        This handles any edge type naming without hardcoded variants.
        """
        # Get all outgoing and incoming edge types + targets for this entity
        # Return individual properties — returning whole node (b AS node)
        # causes serialization issues with the Neo4j driver
        cypher = """
        MATCH (a:Entity {name: $name})-[r]->(b:Entity)
        RETURN type(r) AS rel_type, b.name AS target,
               b.domain AS domain, b.morphology AS morphology,
               b.entity_type AS entity_type,
               b.distinctive_features AS distinctive_features,
               b.color_palette AS color_palette,
               b.contrastive_constraints AS contrastive_constraints
        LIMIT 30
        """
        out_results = self.client.run(cypher, {"name": anchor_name})

        cypher = """
        MATCH (b:Entity)-[r]->(a:Entity {name: $name})
        RETURN type(r) AS rel_type, b.name AS target,
               b.domain AS domain, b.morphology AS morphology,
               b.entity_type AS entity_type,
               b.distinctive_features AS distinctive_features,
               b.color_palette AS color_palette,
               b.contrastive_constraints AS contrastive_constraints
        LIMIT 30
        """
        in_results = self.client.run(cypher, {"name": anchor_name})

        all_edges = []
        node_map  = {}
        for row in out_results + in_results:
            rel = row.get("rel_type", "")
            tgt = row.get("target", "")
            if rel and tgt:
                all_edges.append((rel, tgt))
                # Build a node dict from the returned properties
                node_map[tgt] = {
                    "name":                   tgt,
                    "domain":                 row.get("domain", ""),
                    "entity_type":            row.get("entity_type", ""),
                    "morphology":             row.get("morphology", ""),
                    "distinctive_features":   row.get("distinctive_features", []),
                    "color_palette":          row.get("color_palette", []),
                    "contrastive_constraints":row.get("contrastive_constraints", []),
                }

        if not all_edges:
            logger.warning(f"  No edges found for '{anchor_name}'")
            return []

        logger.info(f"  All edges for '{anchor_name}': {all_edges[:10]}")

        # Ask LLM to pick the best matching edge for the intent
        edge_list = "\n".join(f"  [{r}] -> {t}" for r, t in all_edges)
        prompt = (
            f"Given these graph edges for entity '{anchor_name}':\n{edge_list}\n\n"
            f"Which edge best matches the relationship intent: '{relation_type}'?\n"
            f"Return ONLY the target entity name (exactly as shown), or NONE."
        )
        response = self._llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You select the best matching graph edge for a relationship intent. Return only the target entity name or NONE."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=30,
        )
        best_target = (response.choices[0].message.content or "").strip()

        if best_target == "NONE" or not best_target:
            logger.warning(f"  LLM found no matching edge for '{relation_type}' from '{anchor_name}'")
            return []

        # Match against known targets (handle minor formatting differences)
        for tgt_name, node in node_map.items():
            if tgt_name.lower() == best_target.lower() or best_target.lower() in tgt_name.lower():
                logger.info(f"    LLM matched '{relation_type}' → '{tgt_name}'")
                return [node]

        logger.warning(f"  LLM returned '{best_target}' but not found in node map")
        return []

    # ── Three-tier node matching ───────────────────────────────────────────────

    def _resolve_node(self, name: str) -> dict | None:
        """Tier 1 → Tier 2 → Tier 3."""
        node = self._neo4j_match(name)
        if node:
            return node

        node = self._token_overlap_match(name)
        if node:
            logger.info(f"    Tier 2 (token overlap) matched '{name}'")
            return node

        best_name = self.resolver.resolve(name, self._kg_names)
        if best_name:
            logger.info(f"    Tier 3 (LLM semantic) matched '{name}' → '{best_name}'")
            return self._neo4j_match(best_name)

        return None

    def _neo4j_match(self, name: str) -> dict | None:
        cypher = """
        MATCH (e:Entity)
        WHERE e.domain IS NOT NULL
          AND (
            toLower(e.name) = toLower($name)
            OR any(a IN e.alternative_names WHERE toLower(a) = toLower($name))
          )
        RETURN e
        LIMIT 1
        """
        results = self.client.run(cypher, {"name": name})
        if results:
            return results[0].get("e", {})
        return None

    def _token_overlap_match(self, name: str) -> dict | None:
        name_tokens = set(name.lower().split())
        best_node, best_score = None, 0

        cypher = "MATCH (e:Entity) WHERE e.domain IS NOT NULL RETURN e.name AS name LIMIT 500"
        results = self.client.run(cypher)

        for row in results:
            kg_name = row.get("name", "")
            kg_tokens = set(kg_name.lower().split())
            overlap = len(name_tokens & kg_tokens)
            if overlap > best_score:
                best_score = overlap
                best_node  = kg_name

        if best_score > 0 and best_node:
            return self._neo4j_match(best_node)
        return None

    # ── KG name cache ─────────────────────────────────────────────────────────

    def _load_all_kg_names(self) -> list[str]:
        try:
            results = self.client.run(
                "MATCH (e:Entity) WHERE e.domain IS NOT NULL RETURN e.name AS name"
            )
            return [r["name"] for r in results if r.get("name")]
        except Exception as exc:
            logger.warning(f"  Could not load KG names: {exc}")
            return []

    # ── k-hop expansion ───────────────────────────────────────────────────────

    def _expand_khop(self, primary_nodes: list[dict]) -> list[dict]:
        neighbours    = []
        primary_names = {n.get("name") for n in primary_nodes}

        for node in primary_nodes:
            cypher = f"""
            MATCH (e:Entity {{name: $name}})-[*1..{self.k}]-(nb:Entity)
            WHERE nb.domain IS NOT NULL AND nb.name <> $name
            RETURN DISTINCT nb
            LIMIT {self.max_neighbours}
            """
            results = self.client.run(cypher, {"name": node.get("name")})
            for row in results:
                nb = row.get("nb", {})
                nb_name = nb.get("name", "")
                if nb_name and nb_name not in primary_names:
                    neighbours.append(nb)
                    primary_names.add(nb_name)

        return neighbours

    # ── Relationships ─────────────────────────────────────────────────────────

    def _fetch_relationships(self, primary_nodes: list[dict]) -> list[dict]:
        if not primary_nodes:
            return []
        names = [n.get("name") for n in primary_nodes if n.get("name")]
        cypher = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.name IN $names OR b.name IN $names
        RETURN a.name AS from_node, type(r) AS rel_type, b.name AS to_node
        LIMIT 50
        """
        results = self.client.run(cypher, {"names": names})
        return [{"from": r["from_node"], "type": r["rel_type"], "to": r["to_node"]}
                for r in results]

    # ── Attribute helpers ─────────────────────────────────────────────────────

    def _build_attribute_list(self, nodes: list[dict]) -> list[str]:
        attrs, seen = [], set()
        for node in nodes:
            candidates = (
                [node.get("morphology", "")]
                + (node.get("distinctive_features", []) or [])
                + (node.get("color_palette", []) or [])
                + [node.get("texture", ""), node.get("size_and_scale", "")]
            )
            for a in candidates:
                if a and a not in seen:
                    seen.add(a)
                    attrs.append(a)
        return attrs

    def _build_contrastive(self, nodes: list[dict]) -> list[str]:
        constraints = []
        for node in nodes:
            constraints.extend(node.get("contrastive_constraints", []) or [])
        return constraints
