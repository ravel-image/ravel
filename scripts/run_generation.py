"""
scripts/run_generation.py
──────────────────────────
CLI for RAVEL image generation. Supports all 5 backbones.

Examples:
    # DALL-E 3 (no GPU needed)
    python scripts/run_generation.py \
        --prompt "Hindu god Yama seated on a water buffalo" \
        --backbone dalle3 --srd

    # Flux
    python scripts/run_generation.py \
        --prompt "Red Ginger plant blooming in tropical forest" \
        --backbone flux --guidance-scale 3.5 --steps 50 --srd

    # SDXL
    python scripts/run_generation.py \
        --prompt "Saola in dense Vietnamese forest" \
        --backbone sdxl --guidance-scale 7.5 --steps 50 --no-refiner

    # Janus-Pro (autoregressive)
    python scripts/run_generation.py \
        --prompt "Hindu god Yama seated on a water buffalo" \
        --backbone janus_pro --temperature 1.0 --cfg-weight 5.0

    # Infinity (autoregressive, needs INFINITY_REPO env var)
    python scripts/run_generation.py \
        --prompt "Hindu god Yama seated on a water buffalo" \
        --backbone infinity --model-size 2b --cfg-scale 3.0

    # With SRD + seed
    python scripts/run_generation.py \
        --prompt "Kapala skull bowl ritual object" \
        --backbone dalle3 --srd --seed 42 --output output/

    # Batch from file (one prompt per line)
    python scripts/run_generation.py \
        --prompts-file prompts.txt --backbone flux --srd
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from pipeline import RAVELPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAVEL rare concept image generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Prompt ────────────────────────────────────────────────────────────────
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt", type=str,
        help="Single text prompt."
    )
    prompt_group.add_argument(
        "--prompts-file", type=Path,
        help="Text file with one prompt per line."
    )

    # ── Backbone ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--backbone", type=str, default="dalle3",
        choices=["sdxl", "flux", "dalle3", "janus_pro", "glm_image"],
        help="T2I backbone (default: dalle3)."
    )

    # ── Diffusion backbone args ───────────────────────────────────────────────
    parser.add_argument("--guidance-scale", type=float, default=None,
        help="CFG guidance scale. Default: 7.5 (sdxl), 3.5 (flux).")
    parser.add_argument("--steps", type=int, default=None,
        help="Number of denoising steps (default: 50).")
    parser.add_argument("--image-size", type=int, default=1024,
        help="Output image size in pixels (default: 1024).")
    parser.add_argument("--no-refiner", action="store_true",
        help="Disable SDXL refiner stage.")
    parser.add_argument("--dalle-quality", type=str, default="hd",
        choices=["standard", "hd"],
        help="DALL-E 3 quality (default: hd).")

    # ── Janus-Pro args ────────────────────────────────────────────────────────
    parser.add_argument("--temperature", type=float, default=1.0,
        help="Janus-Pro sampling temperature (default: 1.0).")
    parser.add_argument("--cfg-weight", type=float, default=5.0,
        help="Janus-Pro classifier-free guidance weight (default: 5.0).")

    # ── Infinity args ─────────────────────────────────────────────────────────
    parser.add_argument("--model-size", type=str, default="2b",
        choices=["2b", "8b"],
        help="Infinity model size (default: 2b).")
    parser.add_argument("--cfg-scale", type=float, default=3.0,
        help="Infinity CFG scale (default: 3.0).")
    parser.add_argument("--vae-type", type=int, default=32,
        choices=[16, 24, 32, 64],
        help="Infinity VAE vocabulary bits (default: 32).")

    # ── SRD ───────────────────────────────────────────────────────────────────
    srd_group = parser.add_mutually_exclusive_group()
    srd_group.add_argument("--srd", action="store_true", default=False,
        help="Enable SRD iterative self-correction.")
    srd_group.add_argument("--no-srd", action="store_true",
        help="Disable SRD (default).")

    parser.add_argument("--tau", type=float, default=0.85,
        help="SRD GSI convergence threshold (default: 0.85).")
    parser.add_argument("--max-k", type=int, default=3,
        help="SRD max iterations (default: 3).")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=None,
        help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="output/",
        help="Output directory (default: output/).")

    return parser.parse_args()


# ── Build backbone kwargs ─────────────────────────────────────────────────────

def build_backbone_kwargs(args: argparse.Namespace) -> dict:
    """Extract backbone-specific kwargs from parsed args."""
    kwargs = {}

    if args.backbone == "sdxl":
        if args.guidance_scale is not None:
            kwargs["guidance_scale"] = args.guidance_scale
        else:
            kwargs["guidance_scale"] = 7.5
        if args.steps is not None:
            kwargs["steps"] = args.steps
        kwargs["image_size"]   = args.image_size
        kwargs["use_refiner"]  = not args.no_refiner

    elif args.backbone == "flux":
        if args.guidance_scale is not None:
            kwargs["guidance_scale"] = args.guidance_scale
        else:
            kwargs["guidance_scale"] = 3.5
        if args.steps is not None:
            kwargs["steps"] = args.steps
        kwargs["image_size"] = args.image_size

    elif args.backbone == "dalle3":
        kwargs["quality"] = args.dalle_quality
        size = f"{args.image_size}x{args.image_size}"
        # DALL-E 3 only supports specific sizes
        valid = {"1024x1024", "1792x1024", "1024x1792"}
        kwargs["size"] = size if size in valid else "1024x1024"

    elif args.backbone == "janus_pro":
        kwargs["temperature"] = args.temperature
        kwargs["cfg_weight"]  = args.cfg_weight

    elif args.backbone == "glm_image":
        if args.guidance_scale is not None:
            kwargs["guidance_scale"] = args.guidance_scale
        else:
            kwargs["guidance_scale"] = 5.0
        if args.steps is not None:
            kwargs["steps"] = args.steps
        kwargs["height"] = args.image_size
        kwargs["width"]  = args.image_size

    return kwargs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.prompts_file, "r") as f:
            prompts = [
                line.strip() for line in f
                if line.strip() and not line.startswith("#")
            ]
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    srd_enabled     = args.srd and not args.no_srd
    backbone_kwargs = build_backbone_kwargs(args)

    logger.info(f"Backbone : {args.backbone}")
    logger.info(f"SRD      : {srd_enabled}")
    logger.info(f"Kwargs   : {backbone_kwargs}")

    with RAVELPipeline(
        backbone_name=args.backbone,
        srd=srd_enabled,
        tau=args.tau,
        max_k=args.max_k,
        output_dir=args.output,
        **backbone_kwargs,
    ) as pipeline:
        if len(prompts) == 1:
            pipeline.run(prompts[0], seed=args.seed)
        else:
            pipeline.run_batch(prompts, seed=args.seed)

    logger.info("Done.")


if __name__ == "__main__":
    main()
