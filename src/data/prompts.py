"""
src/data/prompts.py
────────────────────
Universal knowledge graph extraction prompt.
Domain is passed as a variable — the LLM uses Chain-of-Thought
reasoning to infer all domain-appropriate attributes, relationship
types, and contrastive constraints on its own.

No hardcoded domain descriptions or relationship type lists.
Adding a new domain requires zero changes to this file.
"""


UNIVERSAL_EXTRACTION_PROMPT = """\
### Instruction

You are an expert knowledge graph builder specializing in rare and
underrepresented concepts across diverse domains.

Your task is to analyze the provided text about "{entity_name}" from the
"{domain}" domain and extract structured attributes that will be used to
generate accurate, detailed images of this entity.

=== CHAIN-OF-THOUGHT REASONING ===

Before producing the final JSON, reason through the following steps:

Step 1 — Domain Understanding:
    What kind of domain is "{domain}"?
    What types of entities exist in this domain?
    What attributes are most important for visually representing
    entities from this domain accurately?

Step 2 — Entity Classification:
    What specific type of entity is "{entity_name}" within this domain?
    What makes it rare or visually distinctive compared to common
    entities in the same category?

Step 3 — Visual Attribute Extraction:
    What are ALL the visual features needed for accurate image generation?
    Think about: shape, size, color, texture, structure, material,
    unique anatomical or decorative features.
    What would a text-to-image model get WRONG if not explicitly told?
    What common generic entity would it default to instead?

Step 4 — Functional and Contextual Grounding:
    What does this entity do, represent, or symbolize?
    Where does it come from geographically, culturally, or historically?
    What is its ecological, narrative, ceremonial, or social role?

Step 5 — Relational Mapping:
    What other named entities is it directly related to?
    Based on the domain "{domain}", infer the most appropriate
    relationship TYPE for each connection using UPPERCASE_SNAKE_CASE
    (e.g. HAS_PARENT, POLLINATED_BY, WORN_WITH, ENEMY_OF, ENDEMIC_TO).
    Choose relationship types that are specific and meaningful for
    this domain.

Step 6 — Contrastive Analysis:
    What is the most common generic entity this would be confused with?
    Write 2-3 explicit "NOT X" constraints that prevent a diffusion
    model from defaulting to that generic prior.

=== OUTPUT FORMAT ===

Return ONLY a valid JSON object.
No markdown fences, no preamble, no explanation outside the JSON.

{{
  "name": "<entity name>",
  "domain": "{domain}",
  "entity_type": "<specific type within this domain>",
  "alternative_names": ["<other names, titles, or scientific names>"],
  "primary_sources": ["<texts, databases, or references documenting this entity>"],

  "visual_attributes": {{
    "morphology": "<overall form, shape, body structure — be specific>",
    "distinctive_features": [
      "<most visually important unique feature>",
      "<second distinctive feature>",
      "<add as many as are relevant>"
    ],
    "color_palette": ["<primary color>", "<secondary color>", "<accent color>"],
    "texture": "<surface texture — e.g. velvety, scaly, smooth, waxy>",
    "size_and_scale": "<size with units where possible>",
    "structural_arrangement": "<how the parts are spatially laid out>",
    "domain_specific_visual": {{
      "<infer key from domain>": "<value>",
      "<infer key from domain>": "<value>"
    }}
  }},

  "functional_attributes": {{
    "primary_function": "<main role, purpose, power, or ecological function>",
    "secondary_functions": ["<secondary role 1>", "<secondary role 2>"],
    "domain_specific_functional": {{
      "<infer key from domain>": "<value>",
      "<infer key from domain>": "<value>"
    }}
  }},

  "relational_attributes": {{
    "relationships": [
      {{
        "type": "<INFERRED_RELATIONSHIP_TYPE>",
        "target": "<name of related entity>",
        "description": "<one line explaining this relationship>"
      }}
    ],
    "associated_entities": ["<entity name 1>", "<entity name 2>"],
    "symbolic_items": ["<item 1>", "<item 2>"]
  }},

  "contextual_attributes": {{
    "origin": "<geographic, cultural, or mythological origin>",
    "historical_period": "<time period or era if applicable>",
    "geographic_range": "<regions or locations>",
    "cultural_significance": "<what this entity means to its culture or ecosystem>",
    "primary_sources": ["<source 1>", "<source 2>"]
  }},

  "contrastive_constraints": [
    "NOT <generic entity it would be confused with>",
    "NOT <wrong visual feature to avoid>",
    "NOT <wrong cultural or contextual framing>"
  ]
}}

=== SOURCE TEXT ===
{source_text}
"""


def build_extraction_prompt(entity_name: str, domain: str, source_text: str) -> str:
    """
    Build the extraction prompt for any entity and domain.

    Args:
        entity_name : name of the entity to extract
        domain      : domain string — passed directly to the LLM,
                      no registration or config needed
        source_text : raw scraped text (Wikipedia / Gutenberg / other)

    Returns:
        Formatted prompt string ready to send to GPT-4o
    """
    return UNIVERSAL_EXTRACTION_PROMPT.format(
        entity_name=entity_name,
        domain=domain,
        source_text=source_text[:8000],
    )
