"""
src/kg/loader.py
─────────────────
Loads extracted entity JSONs into Neo4j.

For each entity JSON:
    1. MERGE the Entity node with all extracted properties
    2. MERGE bidirectional edges from relational_attributes.relationships

Bidirectionality (Eq. 2 in paper):
    ∀(vi, r, vj) ∈ R, ∃(vj, r', vi) ∈ R

The inverse relationship type is inferred automatically by the LLM
during extraction. If no inverse is stored, we create a generic
INVERSE_<TYPE> edge so the graph stays traversable in both directions.
"""

import json
import logging
from pathlib import Path

from src.kg.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# ── Output root (where extractor saves JSONs) ─────────────────────────────────
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "output"


# ── Node loading ──────────────────────────────────────────────────────────────

def load_entity_node(client: Neo4jClient, data: dict) -> None:
    """
    MERGE an Entity node into Neo4j from an extracted entity dict.

    Uses MERGE so re-running is safe — existing nodes are updated,
    not duplicated.

    Args:
        client : active Neo4jClient
        data   : parsed entity JSON dict
    """
    name = data.get("name", "").strip()
    if not name:
        logger.warning("  Skipping entity with no name.")
        return

    # Flatten nested attributes into top-level node properties
    visual      = data.get("visual_attributes", {})
    functional  = data.get("functional_attributes", {})
    contextual  = data.get("contextual_attributes", {})

    cypher = """
    MERGE (e:Entity {name: $name})
    SET
        e.domain                = $domain,
        e.entity_type           = $entity_type,
        e.alternative_names     = $alternative_names,
        e.primary_sources       = $primary_sources,

        e.morphology            = $morphology,
        e.distinctive_features  = $distinctive_features,
        e.color_palette         = $color_palette,
        e.texture               = $texture,
        e.size_and_scale        = $size_and_scale,
        e.structural_arrangement= $structural_arrangement,

        e.primary_function      = $primary_function,
        e.secondary_functions   = $secondary_functions,

        e.origin                = $origin,
        e.historical_period     = $historical_period,
        e.geographic_range      = $geographic_range,
        e.cultural_significance = $cultural_significance,

        e.contrastive_constraints = $contrastive_constraints
    """

    params = {
        "name":                   name,
        "domain":                 data.get("domain", ""),
        "entity_type":            data.get("entity_type", ""),
        "alternative_names":      data.get("alternative_names", []),
        "primary_sources":        data.get("primary_sources", []),

        "morphology":             visual.get("morphology", ""),
        "distinctive_features":   visual.get("distinctive_features", []),
        "color_palette":          visual.get("color_palette", []),
        "texture":                visual.get("texture", ""),
        "size_and_scale":         visual.get("size_and_scale", ""),
        "structural_arrangement": visual.get("structural_arrangement", ""),

        "primary_function":       functional.get("primary_function", ""),
        "secondary_functions":    functional.get("secondary_functions", []),

        "origin":                 contextual.get("origin", ""),
        "historical_period":      contextual.get("historical_period", ""),
        "geographic_range":       contextual.get("geographic_range", ""),
        "cultural_significance":  contextual.get("cultural_significance", ""),

        "contrastive_constraints": data.get("contrastive_constraints", []),
    }

    client.run(cypher, params)
    logger.info(f"  MERGE entity node: '{name}'")

    # Store domain_specific fields as a separate node property (JSON string)
    # so we don't lose any LLM-inferred domain attributes
    _store_domain_specific(client, name, visual, functional)


def _store_domain_specific(
    client: Neo4jClient,
    name: str,
    visual: dict,
    functional: dict,
) -> None:
    """
    Store domain_specific_visual and domain_specific_functional
    as JSON strings on the node so no LLM-inferred data is lost.
    """
    import json as _json

    ds_visual      = visual.get("domain_specific_visual", {})
    ds_functional  = functional.get("domain_specific_functional", {})

    if not ds_visual and not ds_functional:
        return

    cypher = """
    MATCH (e:Entity {name: $name})
    SET e.domain_specific_visual      = $ds_visual,
        e.domain_specific_functional  = $ds_functional
    """
    client.run(cypher, {
        "name":          name,
        "ds_visual":     _json.dumps(ds_visual,     ensure_ascii=False),
        "ds_functional": _json.dumps(ds_functional, ensure_ascii=False),
    })


# ── Edge loading ──────────────────────────────────────────────────────────────

def load_entity_edges(client: Neo4jClient, data: dict) -> None:
    """
    Create bidirectional typed edges from relational_attributes.relationships.

    For each relationship {type, target}:
        (entity) -[TYPE]-> (target)
        (target) -[INVERSE_TYPE]-> (entity)

    Both nodes are MERGEd so stub nodes are created for targets
    not yet loaded — they will be enriched when their own JSON is loaded.

    Args:
        client : active Neo4jClient
        data   : parsed entity JSON dict
    """
    name          = data.get("name", "").strip()
    relationships = data.get("relational_attributes", {}).get("relationships", [])

    if not name or not relationships:
        return

    for rel in relationships:
        rel_type = rel.get("type", "").strip().upper().replace(" ", "_")
        target   = rel.get("target", "").strip()

        if not rel_type or not target:
            continue

        # Sanitize relationship type — Neo4j requires valid identifier
        rel_type = _sanitize_rel_type(rel_type)
        inv_type = _infer_inverse(rel_type)

        # Forward edge
        _merge_edge(client, name, rel_type, target)

        # Inverse edge
        _merge_edge(client, target, inv_type, name)


def _merge_edge(client: Neo4jClient, from_name: str, rel_type: str, to_name: str) -> None:
    """MERGE a single directed edge, creating stub nodes if needed."""
    cypher = f"""
    MERGE (a:Entity {{name: $from_name}})
    MERGE (b:Entity {{name: $to_name}})
    MERGE (a)-[:{rel_type}]->(b)
    """
    client.run(cypher, {"from_name": from_name, "to_name": to_name})
    logger.debug(f"    MERGE ({from_name})-[{rel_type}]->({to_name})")


def _sanitize_rel_type(rel_type: str) -> str:
    """
    Ensure relationship type is a valid Neo4j identifier.
    Replaces spaces and hyphens with underscores, strips special chars.
    """
    import re
    sanitized = re.sub(r"[^A-Z0-9_]", "_", rel_type.upper())
    # Must not start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "REL_" + sanitized
    return sanitized


def _infer_inverse(rel_type: str) -> str:
    """
    Return the inverse relationship type.

    Known pairs are mapped explicitly.
    Unknown types get a generic INVERSE_ prefix so the graph
    stays bidirectionally traversable.
    """
    known_inverses = {
        "HAS_PARENT":       "HAS_CHILD",
        "HAS_CHILD":        "HAS_PARENT",
        "HAS_SPOUSE":       "HAS_SPOUSE",
        "HAS_SIBLING":      "HAS_SIBLING",
        "AVATAR_OF":        "HAS_AVATAR",
        "HAS_AVATAR":       "AVATAR_OF",
        "RIDES":            "IS_RIDDEN_BY",
        "IS_RIDDEN_BY":     "RIDES",
        "WIELDS":           "IS_WIELDED_BY",
        "IS_WIELDED_BY":    "WIELDS",
        "ENEMY_OF":         "ENEMY_OF",
        "ALLY_OF":          "ALLY_OF",
        "TEACHER_OF":       "STUDENT_OF",
        "STUDENT_OF":       "TEACHER_OF",
        "POLLINATED_BY":    "POLLINATES",
        "POLLINATES":       "POLLINATED_BY",
        "PREY_OF":          "PREYS_ON",
        "PREYS_ON":         "PREY_OF",
        "SYMBIOTIC_WITH":   "SYMBIOTIC_WITH",
        "FOUND_IN_HABITAT": "CONTAINS_SPECIES",
        "CONTAINS_SPECIES": "FOUND_IN_HABITAT",
        "ENDEMIC_TO":       "HAS_ENDEMIC_SPECIES",
        "WORN_WITH":        "WORN_WITH",
        "USED_IN":          "FEATURES",
        "FEATURES":         "USED_IN",
        "CREATED_BY":       "CREATED",
        "CREATED":          "CREATED_BY",
        "INFLUENCED_BY":    "INFLUENCED",
        "INFLUENCED":       "INFLUENCED_BY",
        "PART_OF":          "HAS_PART",
        "HAS_PART":         "PART_OF",
        "ASSOCIATED_WITH":  "ASSOCIATED_WITH",
        "RULES_OVER":       "RULED_BY",
        "RULED_BY":         "RULES_OVER",
        "APPEARS_IN":       "FEATURES_CHARACTER",
        "FEATURES_CHARACTER": "APPEARS_IN",
        "LIVES_IN":         "INHABITED_BY",
        "INHABITED_BY":     "LIVES_IN",
    }
    return known_inverses.get(rel_type, f"INVERSE_{rel_type}")


# ── Domain-level loader ───────────────────────────────────────────────────────

def load_domain(client: Neo4jClient, domain: str) -> None:
    """
    Load all extracted entity JSONs for a domain into Neo4j.

    Runs in two passes:
        Pass 1 — load all entity nodes
        Pass 2 — load all edges (ensures target nodes exist before edges)

    Args:
        client : active Neo4jClient
        domain : domain string (must match output subdirectory name)
    """
    domain_dir = OUTPUT_ROOT / domain

    if not domain_dir.exists():
        logger.error(f"Output directory not found: {domain_dir}")
        logger.error("Run the extractor first: python scripts/build_kg.py --domain {domain}")
        return

    json_files = sorted(domain_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {domain_dir}")
        return

    logger.info(f"Loading domain '{domain}': {len(json_files)} entities")

    # Load all JSON files once
    entities = []
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            try:
                entities.append(json.load(f))
            except json.JSONDecodeError as e:
                logger.warning(f"  Skipping invalid JSON {path.name}: {e}")

    # Pass 1 — nodes
    logger.info("  Pass 1: loading entity nodes...")
    for data in entities:
        load_entity_node(client, data)

    # Pass 2 — edges
    logger.info("  Pass 2: loading edges...")
    for data in entities:
        load_entity_edges(client, data)

    logger.info(f"  Done: {len(entities)} entities loaded for domain '{domain}'")
