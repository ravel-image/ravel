"""
src/generation/backbone.py
───────────────────────────
Unified T2I backbone interface.

Usage via CLI:
    python scripts/run_generation.py --backbone dalle3   --prompt "..."
    python scripts/run_generation.py --backbone flux     --prompt "..."
    python scripts/run_generation.py --backbone sdxl     --prompt "..."
    python scripts/run_generation.py --backbone janus_pro --prompt "..."
    python scripts/run_generation.py --backbone infinity  --prompt "..."

Backbones:
    Diffusion  : sdxl, flux, dalle3
    Autoregressive : janus_pro, infinity

All share: generate(prompt, seed) -> PIL.Image

Setup per backbone:
    sdxl       : pip install diffusers transformers accelerate
    flux       : pip install diffusers transformers accelerate + HF_TOKEN in .env
    dalle3     : OPENAI_API_KEY in .env (no GPU needed)
    janus_pro  : pip install transformers accelerate + HF_TOKEN in .env
    infinity   : git clone https://github.com/FoundationVision/Infinity
                 cd Infinity && pip install -r requirements.txt
                 download checkpoints per README
                 export INFINITY_REPO=/path/to/Infinity
"""

import io
import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseBackbone(ABC):

    @abstractmethod
    def generate(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


# ── SDXL ──────────────────────────────────────────────────────────────────────

class SDXLBackbone(BaseBackbone):
    """
    Stable Diffusion XL + optional refiner stage.
    Hierarchical U-Net architecture.
    pip install diffusers transformers accelerate
    """

    def __init__(
        self,
        model_id:       str   = "stabilityai/stable-diffusion-xl-base-1.0",
        refiner_id:     str   = "stabilityai/stable-diffusion-xl-refiner-1.0",
        use_refiner:    bool  = True,
        guidance_scale: float = 7.5,
        steps:          int   = 50,
        image_size:     int   = 1024,
    ):
        import torch
        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
        )

        self.guidance_scale = guidance_scale
        self.steps          = steps
        self.size           = image_size
        self.use_refiner    = use_refiner
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        dtype               = torch.float16 if self.device == "cuda" else torch.float32

        logger.info(f"Loading SDXL on {self.device}...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        ).to(self.device)
        self.pipe.enable_attention_slicing()

        if use_refiner:
            logger.info("Loading SDXL refiner...")
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_id,
                torch_dtype=dtype,
                use_safetensors=True,
            ).to(self.device)
            self.refiner.enable_attention_slicing()

    def generate(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        import torch
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None

        image = self.pipe(
            prompt=prompt,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            width=self.size,
            height=self.size,
            generator=generator,
            output_type="latent" if self.use_refiner else "pil",
        ).images[0]

        if self.use_refiner:
            image = self.refiner(
                prompt=prompt,
                image=image[None, :],
                generator=generator,
            ).images[0]

        return image

    def name(self) -> str:
        return "sdxl"


# ── Flux ──────────────────────────────────────────────────────────────────────

class FluxBackbone(BaseBackbone):
    """
    FLUX.1-dev — Multimodal Diffusion Transformer (MM-DiT).
    pip install diffusers transformers accelerate
    Requires HF_TOKEN in .env for gated model access.
    """

    def __init__(
        self,
        model_id:       str   = "black-forest-labs/FLUX.1-dev",
        guidance_scale: float = 3.5,
        steps:          int   = 50,
        image_size:     int   = 1024,
    ):
        import torch
        from diffusers import FluxPipeline

        self.guidance_scale = guidance_scale
        self.steps          = steps
        self.size           = image_size
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        dtype               = torch.bfloat16 if self.device == "cuda" else torch.float32

        logger.info(f"Loading Flux on {self.device}...")
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=os.getenv("HF_TOKEN"),
        ).to(self.device)

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

    def generate(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        import torch
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed else None

        return self.pipe(
            prompt=prompt,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            width=self.size,
            height=self.size,
            generator=generator,
        ).images[0]

    def name(self) -> str:
        return "flux"


# ── DALL-E 3 ──────────────────────────────────────────────────────────────────

class DallE3Backbone(BaseBackbone):
    """
    DALL-E 3 via OpenAI API. No GPU needed.
    Requires OPENAI_API_KEY in .env.
    """

    def __init__(self, quality: str = "hd", size: str = "1024x1024"):
        from openai import OpenAI
        self.client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.quality = quality
        self.size    = size
        logger.info("DALL-E 3 backbone ready (API mode).")

    def generate(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt[:4000],
            size=self.size,
            quality=self.quality,
            n=1,
        )
        img_bytes = requests.get(response.data[0].url, timeout=30).content
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def name(self) -> str:
        return "dalle3"


# ── Janus-Pro ─────────────────────────────────────────────────────────────────

class JanusProBackbone(BaseBackbone):
    """
    Janus-Pro 7B — Unified Multimodal Autoregressive Model (DeepSeek, 2025).

    Uses the official deepseek-ai/Janus-Pro-7B with trust_remote_code=True
    and the custom CFG generation loop from the official repo.

    Setup:
        pip install git+https://github.com/deepseek-ai/Janus.git
        (installs janus.models: MultiModalityCausalLM, VLChatProcessor)

    Model: deepseek-ai/Janus-Pro-7B
    Output resolution: fixed 384x384 (architectural constraint)
    """

    def __init__(
        self,
        model_id:    str   = "deepseek-ai/Janus-Pro-7B",
        temperature: float = 1.0,
        cfg_weight:  float = 5.0,
        image_token_num: int = 576,   # 384/16 * 384/16 = 24*24
        img_size:    int   = 384,
        patch_size:  int   = 16,
    ):
        import sys
        import torch
        from transformers import AutoModelForCausalLM

        self.temperature     = temperature
        self.cfg_weight      = cfg_weight
        self.image_token_num = image_token_num
        self.img_size        = img_size
        self.patch_size      = patch_size

        # The pip janus package is broken on Python 3.12+ and newer transformers.
        # Must use the official repo directly.
        # Setup: git clone https://github.com/deepseek-ai/Janus ~/Janus
        # Do NOT pip install it — just add it to sys.path
        janus_repo = os.getenv("JANUS_REPO", os.path.expanduser("~/Janus"))
        if not os.path.isdir(janus_repo):
            raise EnvironmentError(
                "Janus repo not found. Setup:\n"
                "  git clone https://github.com/deepseek-ai/Janus ~/Janus\n"
                "  export JANUS_REPO=~/Janus\n"
                "Do NOT pip install janus — use the cloned repo directly."
            )
        # Insert at front so repo version takes priority over any pip install
        if janus_repo not in sys.path:
            sys.path.insert(0, janus_repo)

        # Unload any previously imported janus modules so the repo version is used
        for mod in list(sys.modules.keys()):
            if mod.startswith("janus"):
                del sys.modules[mod]

        from janus.models import MultiModalityCausalLM, VLChatProcessor

        logger.info(f"Loading Janus-Pro from {model_id} (repo: {janus_repo})...")

        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

        # Load without any device_map or low_cpu_mem_usage — janus siglip_vit
        # calls .item() during __init__ which fails on meta tensors
        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()

        logger.info("Janus-Pro loaded.")

    @staticmethod
    def _build_prompt(processor, prompt: str) -> str:
        """Format prompt using Janus SFT template + image start tag."""
        conversation = [
            {"role": "<|User|>",      "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        return sft + processor.image_start_tag

    def generate(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        import torch
        import numpy as np

        if seed is not None:
            torch.manual_seed(seed)

        formatted_prompt = self._build_prompt(self.processor, prompt)

        # Tokenize — duplicate for CFG (conditional + unconditional)
        input_ids = self.tokenizer.encode(formatted_prompt)
        input_ids = torch.LongTensor(input_ids)

        # parallel_size=1 → 2 rows: [conditional, unconditional]
        parallel_size = 1
        tokens = torch.zeros(
            (parallel_size * 2, len(input_ids)), dtype=torch.int
        ).cuda()
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                # unconditional: blank out everything except BOS/EOS
                tokens[i, 1:-1] = self.processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        # Autoregressive image token generation with CFG
        generated_tokens = torch.zeros(
            (parallel_size, self.image_token_num), dtype=torch.int
        ).cuda()
        past_key_values = None

        for i in range(self.image_token_num):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            hidden_states   = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logits_cond   = logits[0::2, :]
            logits_uncond = logits[1::2, :]

            # CFG: merge conditional and unconditional logits
            logits = logits_uncond + self.cfg_weight * (logits_cond - logits_uncond)
            probs  = torch.softmax(logits / self.temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            # Embed next token for both conditional + unconditional paths
            next_token_combined = next_token.repeat(2, 1)
            inputs_embeds = self.model.prepare_gen_img_embeds(next_token_combined)

        # Decode image tokens → pixel values via VQVAE
        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, self.img_size // self.patch_size,
                   self.img_size // self.patch_size],
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(dec[0])

    def name(self) -> str:
        return "janus_pro"


# ── Infinity ──────────────────────────────────────────────────────────────────

class GLMImageBackbone(BaseBackbone):
    """
    GLM-Image — Hybrid Autoregressive + Diffusion Decoder model (Zhipu AI, 2025).

    Architecture:
        - 9B AR generator (GLM-4-9B) generates semantic visual tokens
        - 7B single-stream DiT decoder reconstructs high-fidelity image
    Combines AR semantic understanding with diffusion visual fidelity.
    Particularly strong on knowledge-intensive and rare concept generation.

    Fully in diffusers — no repo clone needed:
        pip install diffusers transformers accelerate
        model: zai-org/GLM-Image (~16B total, needs 40GB VRAM or device_map)

    Paper: GLM-Image Technical Report, Zhipu AI 2025.
    HuggingFace: zai-org/GLM-Image
    """

    def __init__(
        self,
        model_id:       str   = "zai-org/GLM-Image",
        guidance_scale: float = 5.0,
        steps:          int   = 30,
        height:         int   = 1024,
        width:          int   = 1024,
    ):
        import torch
        from diffusers.pipelines.glm_image import GlmImagePipeline

        self.guidance_scale = guidance_scale
        self.steps          = steps
        self.height         = height
        self.width          = width

        logger.info(f"Loading GLM-Image from {model_id}...")

        self.pipe = GlmImagePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
        logger.info("GLM-Image loaded.")

    def generate(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        import torch

        generator = (
            torch.Generator(device="cuda").manual_seed(seed)
            if seed is not None else None
        )

        result = self.pipe(
            prompt=prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        return result.images[0]

    def name(self) -> str:
        return "glm_image"


# ── Factory ───────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[BaseBackbone]] = {
    "sdxl":      SDXLBackbone,
    "flux":      FluxBackbone,
    "dalle3":    DallE3Backbone,
    "janus_pro": JanusProBackbone,
    "glm_image": GLMImageBackbone,
}

_ALIASES: dict[str, str] = {
    "dalle":     "dalle3",
    "dall-e-3":  "dalle3",
    "janus":     "janus_pro",
    "januspro":  "janus_pro",
    "glm":       "glm_image",
}


def load_backbone(name: str, **kwargs) -> BaseBackbone:
    """
    Instantiate and return the requested T2I backbone.

    Args:
        name   : backbone identifier
        kwargs : passed directly to backbone constructor

    Backbones:
        "sdxl"      — Stable Diffusion XL (local GPU)
        "flux"      — FLUX.1-dev (local GPU, needs HF_TOKEN)
        "dalle3"    — DALL-E 3 (OpenAI API, no GPU)
        "janus_pro" — Janus-Pro 7B AR model (local GPU, needs HF_TOKEN)
        "glm_image" — GLM-Image hybrid AR+DiT (local GPU, 40GB VRAM)

    Examples:
        load_backbone("dalle3")
        load_backbone("flux", guidance_scale=3.5, steps=50)
        load_backbone("sdxl", use_refiner=False)
        load_backbone("janus_pro", temperature=1.0, cfg_weight=5.0)
        load_backbone("glm_image", guidance_scale=5.0, steps=30)
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)

    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'.\n"
            f"Available: {list(_REGISTRY.keys())}"
        )

    logger.info(f"Initialising backbone: {key}")
    return _REGISTRY[key](**kwargs)
