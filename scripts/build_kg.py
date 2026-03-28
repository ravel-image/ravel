"""
scripts/build_kg.py
────────────────────
CLI entry point for KG construction.

Two modes:

    1. Manual entity list (existing behaviour):
        python scripts/build_kg.py --domain biology
        python scripts/build_kg.py --domain biology --entities path/to/list.json

    2. Auto-generate entities (new):
        python scripts/build_kg.py --domain biology --auto-generate 20
        python scripts/build_kg.py --domain indian_mythology --auto-generate 15 \
            --sources https://www.wisdomlib.org https://www.sacred-texts.com

In both modes the pipeline is:
    extract (scrape + LLM) → save JSONs → load into Neo4j

Pipeline control:
    --extract-only   run extraction only, skip Neo4j loading
    --load-only      load existing JSONs only, skip extraction
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.kg.entity_generator import EntityGenerator
from src.kg.extractor import extract_domain
from src.kg.loader import load_domain
from src.kg.neo4j_client import Neo4jClient

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

SAMPLE_ENTITIES_DIR    = Path(__file__).resolve().parents[1] / "src" / "data" / "sample_entities"
EXTRACTED_ENTITIES_DIR = Path(__file__).resolve().parents[1] / "data" / "extracted_entities"

ALL_DOMAINS = [
    "indian_mythology",
    "greek_mythology",
    "chinese_mythology",
    "literary",
    "biology",
    "natural_phenomena",
    "cultural_artifact",
]


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the RAVEL knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Domain
    domain_group = parser.add_mutually_exclusive_group(required=True)
    domain_group.add_argument(
        "--domain", nargs="+", metavar="DOMAIN",
        help=f"Domain(s) to process. Known: {ALL_DOMAINS}",
    )
    domain_group.add_argument(
        "--all", action="store_true",
        help="Process all known domains.",
    )

    # Entity source — manual list OR auto-generate
    entity_group = parser.add_mutually_exclusive_group()
    entity_group.add_argument(
        "--entities", type=Path, default=None,
        help="Path to a custom entity list JSON (only with single --domain).",
    )
    entity_group.add_argument(
        "--auto-generate", type=int, metavar="N", default=None,
        help="Auto-generate N rare entities via LLM instead of using a fixed list.",
    )

    # Auto-generate options
    parser.add_argument(
        "--sources", nargs="*", default=None,
        metavar="URL",
        help="Optional authoritative source URLs to bias entity generation toward. "
             "Wikipedia is always used regardless.",
    )

    # Pipeline control
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Only scrape + extract. Skip Neo4j loading.",
    )
    parser.add_argument(
        "--load-only", action="store_true",
        help="Only load existing JSONs into Neo4j. Skip extraction.",
    )

    # Extraction options
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI model for extraction (default: gpt-4o).",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Seconds between API calls (default: 1.0).",
    )

    return parser.parse_args()


# ── Auto-generate entity list ─────────────────────────────────────────────────

def generate_entity_list(domain: str, n: int, source_urls: list[str]) -> Path:
    """
    Use LLM to generate n rare entities, save to extracted_entities/<domain>.json
    and return the path so the extractor can use it.
    """
    generator = EntityGenerator()
    entities  = generator.generate(domain=domain, n=n, source_urls=source_urls)

    if not entities:
        logger.error(f"Entity generation returned 0 entities for '{domain}'")
        sys.exit(1)

    # Save generated list for traceability
    EXTRACTED_ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXTRACTED_ENTITIES_DIR / f"{domain}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)

    logger.info(f"  Saved generated entity list: {output_path}")
    return output_path


# ── Pipeline runners ──────────────────────────────────────────────────────────

def run_extraction(
    domain:           str,
    entity_list_path: Path,
    model:            str,
    sleep:            float,
) -> None:
    logger.info(f"{'='*50}")
    logger.info(f"EXTRACTION: {domain}")
    logger.info(f"{'='*50}")
    extract_domain(
        domain=domain,
        entity_list_path=entity_list_path,
        model=model,
        sleep_between=sleep,
    )


def run_loading(domain: str, client: Neo4jClient) -> None:
    logger.info(f"{'='*50}")
    logger.info(f"LOADING: {domain}")
    logger.info(f"{'='*50}")
    load_domain(client, domain)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve domains
    domains = ALL_DOMAINS if args.all else args.domain

    # Validate --entities only with single domain
    if args.entities and len(domains) > 1:
        logger.error("--entities can only be used with a single --domain.")
        sys.exit(1)

    # Validate --sources only with --auto-generate
    if args.sources and args.auto_generate is None:
        logger.error("--sources requires --auto-generate.")
        sys.exit(1)

    source_urls = args.sources or []

    # ── Resolve entity list paths per domain ──────────────────────────────────
    entity_list_paths: dict[str, Path] = {}

    for domain in domains:

        if args.load_only:
            # Load-only: no entity list needed
            entity_list_paths[domain] = None

        elif args.auto_generate:
            # Auto-generate: LLM generates entity list, save to extracted_entities/
            logger.info(
                f"Auto-generating {args.auto_generate} entities "
                f"for '{domain}'..."
            )
            path = generate_entity_list(
                domain=domain,
                n=args.auto_generate,
                source_urls=source_urls,
            )
            entity_list_paths[domain] = path

        elif args.entities:
            # Custom entity list provided
            if not args.entities.exists():
                logger.error(f"Entity list not found: {args.entities}")
                sys.exit(1)
            entity_list_paths[domain] = args.entities

        else:
            # Default: use pre-made sample_entities/<domain>.json
            path = SAMPLE_ENTITIES_DIR / f"{domain}.json"
            if not path.exists():
                logger.error(
                    f"No entity list found for '{domain}' at {path}\n"
                    f"Use --auto-generate N to generate one automatically."
                )
                sys.exit(1)
            entity_list_paths[domain] = path

    # ── Extraction pass ───────────────────────────────────────────────────────
    if not args.load_only:
        for domain in domains:
            run_extraction(
                domain=domain,
                entity_list_path=entity_list_paths[domain],
                model=args.model,
                sleep=args.sleep,
            )

    # ── Loading pass ──────────────────────────────────────────────────────────
    if not args.extract_only:
        try:
            with Neo4jClient() as client:
                client.ensure_constraints()
                for domain in domains:
                    run_loading(domain, client)
        except ConnectionError as e:
            logger.error(f"Neo4j connection failed: {e}")
            sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
