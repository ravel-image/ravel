# RAVEL: Rare Concept Generation and Editing via Graph-driven Relational Guidance

A training-free framework that grounds text-to-image synthesis in a structured
Knowledge Graph (KG) to enable high-fidelity generation of rare, culturally
nuanced, and long-tail concepts.

> **Python 3.12+ required.** Tested on CUDA GPU nodes.

---

## Quick Start

### Step 1 — Install dependencies

```bash
pip install openai neo4j wikipedia-api wikipedia beautifulsoup4 lxml \
            requests python-dotenv diffusers transformers accelerate \
            torch torchvision
```

### Step 2 — Configure environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=xxxxxxxx
NEO4J_PASSWORD=your_aura_password
HF_TOKEN=hf_...
```

**Neo4j:** Free AuraDB instance at console.neo4j.io. Pauses after 3 days — resume before running.
`NEO4J_USERNAME` is the instance ID from your downloaded credentials file (e.g. `0e6d187a`).

**HuggingFace token:** Required for Flux and Janus-Pro.

### Step 3 — Build the Knowledge Graph (one time only)

**Option A — Pre-made entity lists (recommended for reviewers)**

```bash
python scripts/build_kg.py --all
```

Processes `src/data/sample_entities/` — 7 domains, ~100 curated rare entities each.
Scrapes Wikipedia, extracts structured attributes via GPT-4o, loads into Neo4j
with typed relational edges. Takes ~30–60 minutes. Only run once.

**Option B — Auto-generate entity lists via LLM**

```bash
# Auto-generate 20 rare biology entities
python scripts/build_kg.py --domain biology --auto-generate 20

# With custom authoritative source URLs
python scripts/build_kg.py --domain cultural_artifact --auto-generate 15 \
    --sources https://www.metmuseum.org https://www.britishmuseum.org

# Any new domain — not limited to predefined ones
python scripts/build_kg.py --domain japanese_mythology --auto-generate 10
```

### Step 4 — Verify the graph

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.kg.neo4j_client import Neo4jClient
with Neo4jClient() as c:
    r = c.run('MATCH (e:Entity) WHERE e.domain IS NOT NULL RETURN e.domain, count(e) AS n ORDER BY n DESC')
    total = 0
    for x in r:
        print(f'  {x[\"e.domain\"]}: {x[\"n\"]}')
        total += x['n']
    print(f'  TOTAL: {total}')
    edges = c.run('MATCH ()-[r]->() RETURN count(r) AS n')
    print(f'  EDGES: {edges[0][\"n\"]}')
"
```

### Step 5 — Generate images

```bash
# Flux (GPU recommended)
python scripts/run_generation.py --prompt "Yama the Hindu god of death" --backbone flux --srd --output output/

# No GPU — use DALL-E 3
python scripts/run_generation.py --prompt "Yama the Hindu god of death" --backbone dalle3 --srd --output output/
```

---

## Output Structure

Each run creates a dedicated folder:

```
output/yama_flux/
├── 00_base.png              ← vanilla model, raw prompt, no RAVEL
├── 01_ravel.png             ← KG-enriched contrastive prompt, no SRD
├── 02_srd_r1_gsi0.22.png   ← after SRD round 1
├── 03_srd_r2_gsi1.00.png   ← after SRD round 2 (converged)
├── final.png                ← best image
└── run_info.json            ← full metadata, GSI trajectory, attributes used
```

Compare `00_base.png` → `01_ravel.png` → `final.png` to see the progressive
effect of KG-enriched contrastive prompting and iterative self-correction.

---

## Example Prompts — All 7 Domains

### Indian Mythology

```bash
python scripts/run_generation.py --prompt "Yama" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Ganda Bherunda" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Chhinnamasta" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Sharabha" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Tumburu" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Kamadhenu" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Hayagriva" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Makara" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Vetala" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Kinnara" --backbone flux --srd --output output/

# Relational — entity resolved from KG (no name given in prompt)
python scripts/run_generation.py --prompt "Yama's sister" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Shiva's wife" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Kartikeya's vehicle" --backbone flux --srd --output output/
```

### Greek Mythology

```bash
python scripts/run_generation.py --prompt "Cerberus" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Typhon" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Hecatoncheires" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Argus Panoptes" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Scylla" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Ichthyocentaur" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Empusa" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Graeae" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Ceryneian Hind" --backbone flux --srd --output output/

# Relational
python scripts/run_generation.py --prompt "Cerberus's parent" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Typhon's child" --backbone flux --srd --output output/
```

### Chinese Mythology

```bash
python scripts/run_generation.py --prompt "Fenghuang" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Fei Lian" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Pixiu" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Zhong Kui" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Taotie" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Zhulong" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Bai Ze" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Yinglong" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Jiangshi" --backbone flux --srd --output output/

# Relational
python scripts/run_generation.py --prompt "Fenghuang's paired creature" --backbone flux --srd --output output/
```

### Literary

```bash
python scripts/run_generation.py --prompt "Uncle Wiggily Longears" --backbone dalle3 --srd --output output/
python scripts/run_generation.py --prompt "Grendel" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Cthulhu" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "The Jabberwock" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Baba Yaga" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "The Wendigo" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "The Golem of Prague" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Roc Bird" --backbone flux --srd --output output/

# Relational
python scripts/run_generation.py --prompt "Grendel's enemy" --backbone flux --srd --output output/
```

### Biology

```bash
python scripts/run_generation.py --prompt "Saola" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Aye-aye" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Bleeding Tooth Fungus" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Saiga Antelope" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Red-lipped Batfish" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Dragon Blood Tree" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Tarsier" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Leafy Sea Dragon" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Shoebill" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Star-Nosed Mole" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Venezuelan Poodle Moth" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Glass Frog" --backbone flux --srd --output output/
```

### Natural Phenomena

```bash
python scripts/run_generation.py --prompt "Hair Ice" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Brocken Spectre" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Mammatus cloud" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Catatumbo Lightning" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Penitentes" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Brinicle" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Frost Flowers" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Morning Glory cloud" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Fire Rainbow" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Kelvin-Helmholtz instability" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Moonbow" --backbone flux --srd --output output/
```

### Cultural Artifacts

```bash
python scripts/run_generation.py --prompt "Kapala Bowl" --backbone glm_image --srd --output output/
python scripts/run_generation.py --prompt "Nkisi Nkondi" --backbone glm_image --srd --output output/
python scripts/run_generation.py --prompt "Moche Stirrup Spout Vessel" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Noh Theater Mask" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Olmec Colossal Head" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Paracas Textile" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Tibetan Thangka" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Cycladic Figurine" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Luristan Bronze" --backbone flux --srd --output output/
python scripts/run_generation.py --prompt "Benin Bronze Plaque" --backbone flux --srd --output output/
```

---

## Batch Run — Representative Cross-Domain Set

```bash
cat > prompts.txt << 'END'
Yama
Ganda Bherunda
Chhinnamasta
Cerberus
Typhon
Fenghuang
Zhong Kui
Saola
Aye-aye
Bleeding Tooth Fungus
Hair Ice
Brocken Spectre
Kapala Bowl
Nkisi Nkondi
Uncle Wiggily Longears
Grendel
Cthulhu
Shoebill
Dragon Blood Tree
Mammatus cloud
END

python scripts/run_generation.py \
    --prompts-file prompts.txt \
    --backbone flux \
    --srd \
    --output output/batch/
```

---

## Relational Query Demo

RAVEL resolves related entities from the KG — no names needed in the prompt:

| Prompt | KG Traversal | Entities Retrieved |
|---|---|---|
| `"Yama and his sister"` | `Yama -[HAS_SIBLING]-> Yami` | Yama + Yami |
| `"Shiva and his wife"` | `Shiva -[HAS_SPOUSE]-> Parvati` | Shiva + Parvati |
| `"Kartikeya on his vehicle"` | `Kartikeya -[RIDES]-> Peacock` | Kartikeya + Peacock |
| `"Cerberus and his parent"` | `Cerberus -[HAS_PARENT]-> Typhon` | Cerberus + Typhon |
| `"Fenghuang and its paired creature"` | `Fenghuang -[PAIRED_WITH]-> Dragon` | Fenghuang + Dragon |

Test entity resolution:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.kg.neo4j_client import Neo4jClient
from src.kg.retriever import KGRetriever

with Neo4jClient() as client:
    r = KGRetriever(client)
    tests = [
        'Yama',
        'Ganda Bherunda',
        'show me the two headed bird from indian mythology',
        'a saola standing in forest',
        'the bleeding tooth mushroom',
        'Yama and his sister',
        'Shiva and his wife',
        'Cerberus and his parent',
    ]
    for p in tests:
        ctx = r.retrieve(p)
        print(f'{p!r:50s} -> {[e[\"name\"] for e in ctx.primary_entities]}')
"
```

---

## Supported Backbones

| Backbone | Type | GPU | Extra Setup |
|---|---|---|---|
| `flux` | Diffusion MM-DiT | Yes | `HF_TOKEN` |
| `sdxl` | Diffusion U-Net | Yes | None |
| `dalle3` | Diffusion API | No | `OPENAI_API_KEY` |
| `janus_pro` | Autoregressive | Yes | `HF_TOKEN` + Janus repo |
| `glm_image` | AR + DiT hybrid | Yes (40GB+) | None |

**Janus-Pro** requires cloning the official repo (pip package is broken on Python 3.12):

```bash
git clone https://github.com/deepseek-ai/Janus ~/Janus
echo "JANUS_REPO=/home/$USER/Janus" >> .env
```

---

## CLI Reference

```bash
# Single prompt — flux (GPU)
python scripts/run_generation.py --prompt "Yama" --backbone flux --srd --output output/

# Single prompt — dalle3 (no GPU)
python scripts/run_generation.py --prompt "Yama" --backbone dalle3 --srd --output output/

# With SRD hyperparameters
python scripts/run_generation.py --prompt "Yama" --backbone flux --srd --tau 0.85 --max-k 3 --seed 42

# Batch from file
python scripts/run_generation.py --prompts-file prompts.txt --backbone flux --srd --output output/batch/

# No SRD — RAVEL enrichment only, no iterative correction
python scripts/run_generation.py --prompt "Yama" --backbone flux --output output/

# KG build flags
python scripts/build_kg.py --all                                # all domains
python scripts/build_kg.py --domain biology --extract-only     # extract only
python scripts/build_kg.py --domain biology --load-only        # load only
python scripts/build_kg.py --domain biology --auto-generate 20 # LLM-generated
```

---

## Key Hyperparameters (Paper Defaults)

| Parameter | Value | Description |
|---|---|---|
| `tau` | 0.85 | GSI convergence threshold |
| `max_k` | 3 | Max SRD iterations |
| `d0` | 0.9 | Initial decay |
| `k_hops` | 1 | KG retrieval hops |
| `guidance_scale` | 15–30 | Recommended T2I guidance range |

---

## Project Structure

```
ravel/
├── src/
│   ├── data/
│   │   ├── prompts.py                  # Universal LLM extraction prompt
│   │   └── sample_entities/            # Pre-made entity lists (~100 per domain)
│   └── kg/
│       ├── entity_generator.py         # LLM auto-generates rare entity names
│       ├── scraper.py                  # Wikipedia + Gutenberg scraping
│       ├── extractor.py                # LLM structured attribute extraction
│       ├── relationship_extractor.py   # Cross-entity relationship extraction
│       ├── loader.py                   # Neo4j node + edge loading
│       ├── neo4j_client.py             # Neo4j driver wrapper
│       └── retriever.py                # 3-tier matching + relational traversal
│   ├── generation/
│   │   ├── prompt_synth.py             # Contrastive CoT synthesis (Table 11)
│   │   └── backbone.py                 # All T2I backbones
│   └── srd/
│       ├── verifier.py                 # VLM attribute verification (GSI)
│       └── refiner.py                  # SRD iterative self-correction
├── pipeline.py                         # Top-level orchestrator
└── scripts/
    ├── build_kg.py                     # CLI: KG construction
    └── run_generation.py               # CLI: image generation
```

---

## Troubleshooting

**Neo4j: AuthError** — `NEO4J_USERNAME` is the instance ID (e.g. `0e6d187a`), not `neo4j`.
AuraDB free tier pauses after 3 days — resume at console.neo4j.io.

**OpenAI: RateLimitError during extraction**
```bash
python scripts/build_kg.py --domain biology --sleep 2.0
```

**Flux / SDXL: Out of memory** — add `self.pipe.enable_model_cpu_offload()` after
loading in `src/generation/backbone.py`.

**Janus-Pro: meta tensor / TypeError** — do not use `device_map="auto"`.
Use the cloned official repo, not the pip package.

**GLM-Image: device_map error** — use `device_map="balanced"` not `device_map="cuda"`.

**Entity not found in KG**
```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.kg.neo4j_client import Neo4jClient
with Neo4jClient() as c:
    r = c.run(\"MATCH (e:Entity) WHERE e.domain='indian_mythology' RETURN e.name\")
    for x in r: print(x['e.name'])
"
```
