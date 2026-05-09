# Project Plan: Circuit Fingerprinting of True/False Judgment in Gemma-2-2B
 
## Overview
 
This project investigates how a short LoRA fine-tuning run changes the internal computational circuits of Gemma-2-2B when learning to produce True/False judgments on triangle-inequality statements. The core experiment has three phases: (1) generate a parametrized triangle prompt dataset, (2) compute attribution graph statistics on the base model, (3) fine-tune with LoRA and recompute the same statistics, then compare pre/post fingerprints both structurally (Version A) and semantically via Neuronpedia feature categorization (Version B). The primary setup is triangle-only; broader-math fine-tuning is optional Stage 5 expansion.
 
---
 
## Critical Constants — Never Change These
 
```python
MODEL_NAME = "google/gemma-2-2b"          # base model, NOT instruction-tuned
TRANSCODER_NAME = "gemma"
TOKEN_TRUE = " True"                       # leading space is part of the token
TOKEN_FALSE = " False"                     # leading space is part of the token
VOCAB_ID_TRUE = 5569                       # verified against Gemma tokenizer
VOCAB_ID_FALSE = 7662                      # verified against Gemma tokenizer
NEURONPEDIA_MODEL = "gemma-2-2b"
NEURONPEDIA_SET = "gemmascope-transcoder-16k"
N_LAYERS = 26                              # Gemma-2-2B has 26 transformer layers
```
 
These constants must be used everywhere — dataset labeling, LoRA target token specification, attribution target specification, evaluation metric computation, and Neuronpedia URL construction. Never use `"True"` or `"False"` without the leading space.
 
---
 
## Repository Structure
 
```
my_work/
├── data/
│   ├── prompts_triangle.jsonl          # 300 prompt dataset (balanced)
│   └── splits/
│       ├── train_triangle.jsonl        # training split for LoRA
│       ├── eval.jsonl                  # held-out eval set (never used for training)
│       └── analysis.jsonl              # held-out analysis set for circuit tracing
├── cache/
│   ├── neuronpedia_cache.json          # persistent feature label + category cache
│   └── category_overrides.json         # manual category override map (optional)
├── checkpoints/
│   └── lora_triangle/                  # LoRA weights for triangle run
├── results/
│   ├── graphs_base/                    # attribution graph files, base model
│   ├── graphs_lora_triangle/           # attribution graph files, post-LoRA
│   └── statistics/
│       ├── stats_base.json
│       └── stats_lora_triangle.json
├── utils/
│   ├── prompt_generator.py             # dataset generation utilities
│   ├── graph_statistics.py             # graph-level statistics computation
│   ├── neuronpedia.py                  # Neuronpedia fetching + persistent cache
│   └── feature_categorizer.py          # feature label → category mapping
└── notebooks/
    ├── 01_dataset_generation.ipynb
    ├── 02_baseline_attribution.ipynb
    ├── 03_baseline_structural_analysis.ipynb
    ├── 04_lora_finetuning_triangle.ipynb
    ├── 05_postft_attribution_triangle.ipynb
    └── 06_comparison_analysis.ipynb
```
 
All notebooks use cells that import from `utils/`. The notebooks are numbered and meant to be run in order. All outputs should be written under `my_work/` using relative paths so the pipeline runs both locally and on remote pods.
 
---
 
## Phase 1: Dataset Generation (`01_dataset_generation.ipynb`)
 
### Goal
 
Generate exactly 300 parametrized True/False prompts about the triangle inequality, balanced 50/50 True/False, split into train/eval/analysis sets.
 
### Triangle Inequality Prompt Design
 
The triangle inequality states: for any triangle with sides a, b, c, the sum of any two sides must be greater than the third side. Three sub-conditions must all hold: a+b > c, a+c > b, b+c > a.
 
**True prompts** — valid triangles where the stated inequality holds:
- Statement claims the triangle inequality holds → True
- Use a variety of valid side length triples, e.g. (3,4,5), (5,12,13), (2,3,4), (7,8,10), (6,6,6), (1,2,2), (10,10,1)
 
**False prompts** — two types:
1. Valid triangle, but the statement claims the inequality does NOT hold → False (the inequality always holds for valid triangles, so negating it is always false)
2. Degenerate or invalid triples where the triangle inequality is violated, and the statement claims it holds → False (e.g. sides (1,2,3) where 1+2=3, not strictly greater; or (1,1,3) where 1+1 < 3)
 
Include exactly 10% degenerate or invalid triples. These are still labeled True or False — the label is always whether the stated claim is correct.
 
**Ground truth labeling:** Every prompt must have a programmatically verified label. Never assign True/False manually. Write a function `verify_triangle_claim(a, b, c, claim_type) -> bool` that computes the correct answer before generating the prompt string.
 
### Wording Templates
 
Generate at least 5 distinct wording templates. Examples:
 
```python
TEMPLATES = [
    # Template 1 — simple statement
    "Statement: For a triangle with sides {a}, {b}, and {c}, the sum of any two sides is {comparison} the third side. Answer:",
 
    # Template 2 — explicit instruction
    "Statement: For a triangle with sides {a}, {b}, and {c}, the sum of any two sides is {comparison} the third side. Answer with exactly one word, True or False:",
 
    # Template 3 — question form
    "Is it true that for a triangle with sides {a}, {b}, and {c}, the sum of any two sides is {comparison} the third side? Answer:",
 
    # Template 4 — formal math style
    "Mathematical claim: Given side lengths {a}, {b}, {c}, the triangle inequality holds: each side is strictly less than the sum of the other two. True or False?",
 
    # Template 5 — conversational
    "A triangle has sides of length {a}, {b}, and {c}. Someone claims that the sum of any two sides is {comparison} the third. Is this claim correct? Answer:",
]
```
 
Where `{comparison}` is `"greater than"` for True-direction prompts and `"less than"` or `"not greater than"` for False-direction prompts. Add at least 2 more templates of your own design.
 
**Important:** Template 4 is hardcoded as always a True prompt (the triangle inequality always holds for valid triangles, so the statement is always correct). This means template 4 cannot appear in the False split. Two consequences:
1. When stratifying splits, exclude template 4 from the False-label stratum — only templates 1, 2, 3, 5+ can generate False prompts
2. Create a variant of template 4 that can produce False prompts by negating the claim: "Mathematical claim: Given side lengths {a}, {b}, {c}, the triangle inequality does NOT hold. True or False?" — this variant uses degenerate triples where the inequality is indeed violated, yielding True labels for that negated claim form. Assign this variant as template 4b and treat it as a separate template ID in the metadata.
 
### Dataset Schema
 
Each entry in the JSONL files must have this exact schema:
 
```json
{
  "prompt_id": "tri_001",
  "prompt": "Statement: For a triangle with sides 3, 4, and 5, ...",
  "label": true,
  "label_token": " True",
  "sides": [3, 4, 5],
  "triangle_valid": true,
  "template_id": 1,
  "claim_direction": "greater_than",
  "split": "train"
}
```
 
`label_token` must always be either `" True"` or `" False"` with the leading space.
 
### Dataset Splits
 
After generating the triangle inequality dataset, run the split step:
- **train_triangle:** 150 prompts (75 True, 75 False), saved to `data/splits/train_triangle.jsonl`
- **eval:** 75 prompts (balanced), saved to `data/splits/eval.jsonl` — used for accuracy evaluation only, never for circuit analysis
- **analysis:** 75 prompts (balanced), saved to `data/splits/analysis.jsonl` — used for attribution graph generation and fingerprinting, never seen during training
 
The eval and analysis splits are drawn exclusively from triangle inequality prompts and must not overlap with `train_triangle`. Stratify by label only for now; template-level holdout checks are optional future work.

### Optional Stage 5 Expansion (math-broad)

Broader math fine-tuning is explicitly optional and out-of-scope for the core run. If time allows, add a second branch after the triangle-only pipeline is complete and stable.
 
---
 
## Phase 2: Baseline Attribution Graphs (`02_baseline_attribution.ipynb`)
 
### Goal
 
Run attribution graph generation on the base model (no fine-tuning) across the full analysis split. Compute graph statistics for each prompt. This is the pre-fine-tuning fingerprint.
 
### Model Loading
 
```python
from circuit_tracer import ReplacementModel, attribute
 
model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b",
    "gemma",
    dtype=torch.bfloat16,
    backend="transformerlens",
    device=device,
    lazy_encoder=True,
    lazy_decoder=True,
)
tokenizer = model.tokenizer
```
 
### Attribution Graph Generation
 
For each prompt in the analysis split:
 
```python
graph = attribute(
    prompt=prompt,
    model=model,
    attribution_targets=[" True", " False"],  # leading spaces mandatory
    batch_size=256,
    max_feature_nodes=8192,
    offload="disk",
    verbose=False,
)
```
 
Save each graph immediately to `/workspace/results/graphs_base/{prompt_id}.pt` or as JSON using `create_graph_files`. Do not keep all graphs in memory simultaneously.
 
Track for each prompt:
- Whether attribution succeeded (some prompts may fail if the replacement model coverage is too low)
- The probabilities assigned to `" True"` and `" False"` at attribution time
- The number of active features found
 
**Minimum success rate requirement:** At least 70% of analysis prompts must succeed attribution. If fewer than 70% succeed, do not proceed with fingerprinting — instead investigate whether the prompt set is causing unusually low CLT coverage (e.g. very short prompts, unusual token patterns) and regenerate or filter accordingly. Report the success rate explicitly in Phase 5 results. If between 70–85% succeed, note this as a limitation.
 
### Graph Statistics Computation
 
For each successfully attributed prompt, compute the following metrics. Implement in `utils/graph_statistics.py`:
 
**1. Active feature count**
```python
n_active = graph.active_features.shape[0]
```
This is the total number of CLT features that were active on this prompt before pruning.
 
**2. Layer distribution of top-K features**
Extract the top-K features by multi-hop influence score (use K=50 as default). For each of these features, record which layer it belongs to (layer index 0–25). Compute a normalized histogram over the 26 layers. Store as a list of 26 floats summing to 1.0.
 
```python
features, scores = get_top_features(graph, n=50)
layer_counts = [0] * N_LAYERS
for (layer, pos, feat_idx) in features:
    layer_counts[layer] += 1
layer_dist = [c / sum(layer_counts) for c in layer_counts]
```
 
**3. Edge density**
After pruning the graph at a fixed threshold, compute edge density as the number of edges above the threshold divided by the number of node pairs.
 
```python
from circuit_tracer.graph import prune_graph
node_mask, edge_mask, _ = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)
n_nodes = node_mask.sum().item()
n_edges = edge_mask.sum().item()
max_edges = n_nodes * (n_nodes - 1)
edge_density = n_edges / max_edges if max_edges > 0 else 0.0
```
 
Use `node_threshold=0.8` and `edge_threshold=0.98` consistently across all phases. Never change these thresholds between base and post-LoRA runs — the comparison depends on identical thresholds.
 
**4. Mean top-50 influence score**

```python
features, scores = get_top_features(graph, n=50)
mean_top50_score = float(np.mean(scores)) if len(scores) else 0.0
```

**5. Influence concentration (top-10 / top-50)**

```python
top10 = np.sum(np.abs(scores[:10])) if len(scores) >= 10 else np.sum(np.abs(scores))
top50 = np.sum(np.abs(scores)) if len(scores) else 0.0
top10_over_top50 = float(top10 / top50) if top50 > 0 else 0.0
```

**6. Layer entropy of top-50 distribution**

```python
eps = 1e-12
layer_entropy = float(-np.sum(np.array(layer_dist) * np.log(np.array(layer_dist) + eps)))
```

**7. Mean error node weight (transcoder approximation quality)**
 
This is a diagnostic statistic that tracks how well the GemmaScope transcoders approximate the fine-tuned model's MLP computations. Error nodes absorb the residual discrepancy between the CLT reconstruction and the true MLP output. Higher error node weight means the transcoder is a worse approximation — and therefore the attribution graph is less faithful.
 
```python
# Error nodes are the last N nodes in the graph, where N = number of layers
# Their influence on the output logits is the mean error node weight
node_mask, edge_mask, node_scores = prune_graph(
    graph, node_threshold=1.0, edge_threshold=1.0  # no pruning, get all scores
)
# Error nodes are identifiable by their type in graph.node_types
# Compute mean absolute influence of error nodes on the attribution target
error_node_indices = [i for i, t in enumerate(graph.node_types) if t == "error"]
error_node_weight = float(np.mean(np.abs(node_scores[error_node_indices].numpy())))
```
 
**Note:** The exact API for accessing node types may differ — Cursor should inspect `graph` object attributes to find error nodes. If `node_types` is not available, error nodes can be identified as nodes with no incoming edges from CLT features.
 
Store as `mean_error_node_weight` in the output schema. This is the control variable for Phase 5: if error node weight increases substantially post-fine-tuning (e.g. more than 2x), any observed circuit changes must be interpreted with caution as they may partly reflect transcoder degradation rather than genuine circuit reorganization.
 
### Output Schema
 
For each prompt, write to `results/statistics/stats_base.json`:
 
```json
{
  "prompt_id": "tri_001",
  "phase": "base",
  "label": true,
  "label_token": " True",
  "template_id": 1,
  "prob_true": 0.1396,
  "prob_false": 0.0311,
  "logit_gap": 1.5,
  "attribution_succeeded": true,
  "n_active_features": 23758,
  "layer_distribution": [0.02, 0.0, ...],
  "edge_density": 0.0034,
  "mean_top50_score": 0.00180,
  "top10_over_top50": 0.61,
  "layer_entropy": 2.11,
  "mean_error_node_weight": 0.00041,
  "top50_features": [[16, 21, 14143], [20, 21, 12997], ...]
}
```
 
Store `top50_features` as a list of `[layer, pos, feat_idx]` triples — these will be used for Neuronpedia categorization in Phase 5.
 
### Baseline Accuracy Check
 
Before running full attribution, run a quick accuracy check using first-token prediction only: for each prompt in the analysis split, compute whether `argmax(logits[-1]) == label_token`. Report accuracy broken down by True/False. Template breakdown is optional.
 
Expected: ~50–65% accuracy on the base model. If accuracy is already above 80%, note this as it will constrain the interpretability of the fine-tuning experiment.
 
---
 
## Phase 3: Baseline Structural Go/No-Go (`03_baseline_structural_analysis.ipynb`)
 
### Goal

Decide whether to proceed to LoRA by testing whether base-model structural statistics show meaningful prompt-level variation.

### Go/No-Go Criterion

Proceed to LoRA only if both conditions hold:
1. At least two primary structural metrics (from `n_active_features`, `edge_density`, `mean_top50_score`, `top10_over_top50`, `layer_entropy`) show non-trivial spread across prompts (IQR > 0 and visually non-collapsed distributions).
2. A simple classifier (logistic regression) using structural metrics achieves above-majority performance for `correct vs incorrect` on cross-validation, or at minimum shows stable non-zero coefficients on multiple metrics.

If neither condition holds, stop and document that the structural-signal hypothesis is unsupported on this setup.

---

## Phase 4: LoRA Fine-Tuning (`04_lora_finetuning_triangle.ipynb`)
 
### Critical architectural note: train on HuggingFace model, not ReplacementModel
 
The `ReplacementModel` explicitly freezes all parameters (`param.requires_grad = False`) during setup and wraps MLP blocks with custom hook infrastructure. It is designed purely for inference and attribution — LoRA cannot be applied to it directly.
 
The correct sequence is:
1. Load `google/gemma-2-2b` as a standard HuggingFace `AutoModelForCausalLM`
2. Apply and train LoRA on this standard model
3. Merge LoRA weights into the base weights and save
4. In Phase 4, load the merged checkpoint via `ReplacementModel.from_pretrained(local_path, ...)`
 
`ReplacementModel.from_pretrained` accepts local directories with HuggingFace model files, so the merged checkpoint is a drop-in replacement. The CLT transcoders are loaded separately and attached on top.
 
### Training Objective
 
The model must learn to produce `" True"` or `" False"` as the first generated token after the prompt. The loss is cross-entropy on the label token only — not on the full continuation.
 
For each training example, the target is a single token: either vocab ID 5569 (`" True"`) or 7662 (`" False"`). The loss is computed at the last position of the prompt (the position where the model generates its first new token).
 
This is a next-token prediction fine-tuning on a single target token per example, not a sequence-to-sequence task.
 
### Model Loading for Training
 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
 
# Load as standard HuggingFace model — NOT as ReplacementModel
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
```
 
### LoRA Configuration
 
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # rank — single setting for now, may change
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
 
lora_model = get_peft_model(hf_model, lora_config)
lora_model.print_trainable_parameters()  # sanity check — should be ~0.1% of total
```
 
**Note:** The rank r=16 and target modules are a starting point. These may need to be revisited depending on training dynamics. If loss does not decrease after 50 steps, try r=32 or adding `"k_proj"` and `"o_proj"` to target_modules.
 
### Training Setup
 
```python
training_args = TrainingArguments(
    output_dir="/workspace/checkpoints/lora_triangle",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # effective batch size = 16
    learning_rate=2e-4,
    warmup_steps=20,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=False,
    bf16=True,                        # Gemma-2 was trained in bfloat16
    dataloader_num_workers=0,
)
```
 
### Training Loop and Merging
 
Run triangle-only FT:
 
```python
# Train
trainer = Trainer(model=lora_model, args=training_args, ...)
trainer.train()
 
# Merge LoRA weights into base weights — produces a standard HF model
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("/workspace/checkpoints/lora_triangle_merged")
tokenizer.save_pretrained("/workspace/checkpoints/lora_triangle_merged")
```
 
Never chain fine-tunes — always start from the original `google/gemma-2-2b` weights.
 
### Post-Training Evaluation
 
After the LoRA run, evaluate on the eval split (75 prompts, never seen during training) using the standard HuggingFace model directly — do NOT load through ReplacementModel for this step. The CLT wrapping is irrelevant for accuracy evaluation and adds unnecessary overhead.
 
```python
# Evaluation uses the merged HuggingFace model directly, NOT ReplacementModel
from transformers import AutoModelForCausalLM, AutoTokenizer
 
eval_model = AutoModelForCausalLM.from_pretrained(
    "/workspace/checkpoints/lora_triangle_merged",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
eval_tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/checkpoints/lora_triangle_merged"
)
eval_model.eval()
```
 
For each eval prompt:
1. Tokenize with `eval_tokenizer`, run a forward pass with `eval_model`
2. Extract logits at the last token position
3. Check whether `argmax(logits[-1]) == label_token_id` (5569 for `" True"`, 7662 for `" False"`)
4. Record accuracy, broken down by True/False label and by template
 
Report:
- Overall accuracy (target: >80% to make the circuit comparison meaningful)
- Accuracy by template (to check for template overfitting)
- Accuracy by True vs False (to check for class imbalance effects)
- The logit gap `logit(" True") - logit(" False")` distribution before and after
 
If overall accuracy is below 70% after 3 epochs, extend training or adjust LoRA rank. Document any changes.
 
---
 
## Phase 5: Post-Fine-Tuning Attribution Graphs (`05_postft_attribution_triangle.ipynb`)
 
### Goal
 
Repeat the exact same attribution graph generation and statistics computation as Phase 2, but using the LoRA-merged triangle model. The analysis prompts are identical to Phase 2 — same prompt IDs, same prompt strings.
 
### Model Loading with Merged LoRA Weights
 
Do NOT use `ReplacementModel` during training. In Phase 4, load the already-merged checkpoint as if it were a normal model:
 
```python
# Load the merged fine-tuned model through ReplacementModel
# The merged checkpoint is a standard HuggingFace model directory —
# ReplacementModel.from_pretrained accepts local paths
model_post_ft = ReplacementModel.from_pretrained(
    "/workspace/checkpoints/lora_triangle_merged",
    "gemma",                                         # transcoder set name unchanged
    dtype=torch.bfloat16,
    backend="transformerlens",
    device=device,
    lazy_encoder=True,
    lazy_decoder=True,
)
tokenizer = model_post_ft.tokenizer
```
 
The CLT transcoders (GemmaScope) are loaded fresh and attached on top of the merged weights, exactly as in the base model run. The transcoders themselves are not modified by fine-tuning — they were trained on the base model and remain fixed throughout.
 
### Attribution and Statistics
 
Run the identical pipeline as Phase 2. Save results to:
- `results/graphs_lora_triangle/{prompt_id}/`
- `results/statistics/stats_lora_triangle.json`
 
Use identical parameters: `attribution_targets=[" True", " False"]`, `max_feature_nodes=8192`, `node_threshold=0.8`, `edge_threshold=0.98`, K=50 for top features.
 
---

## Phase 6: Comparison Analysis (`06_comparison_analysis.ipynb`)
 
### Goal
 
Compare the pre/post fine-tuning fingerprints along two axes: structural (Version A) and semantic via feature categorization (Version B). Primary analyses are paired by prompt ID and first-token objective only.
 
### Confounder: Transcoder Approximation Degradation
 
Before interpreting any structural or semantic differences between phases, check the `mean_error_node_weight` diagnostic across both phases. If the post-fine-tuning error node weight is substantially higher than the base model (rule of thumb: more than 2x the base mean), all findings must be interpreted with the following caveat: observed circuit changes may partly reflect the GemmaScope transcoders becoming a worse approximation of the fine-tuned MLP computations, rather than genuine circuit reorganization.
 
Report the mean error node weight alongside every other statistic. If degradation is large, note it explicitly in the Phase 5 write-up as a limitation. If degradation is small (less than 50% increase), findings can be reported with reasonable confidence.
 
This diagnostic is the honesty mechanism for the entire comparison — do not skip it.
 
### 5A — Structural Comparison (Version A)
 
Load `stats_base.json` and `stats_lora_triangle.json`. For each statistic, compute the distribution across all successfully attributed prompts.
 
**For each of the four statistics, produce:**
 
1. A box plot or violin plot showing the distribution pre/post
2. A paired difference plot (post minus pre, per prompt) to show individual-level changes
3. A Wilcoxon signed-rank test for statistical significance (primary test family)
4. Rank-biserial correlation as effect size
5. Benjamini-Hochberg correction across the structural metric family
 
**Specific hypotheses to test (based on Yang et al. 2502.11812):**
- H1: Active feature count does not change significantly pre/post fine-tuning (nodes stable)
- H2: Layer distribution changes significantly (edges reorganize, affecting which layers dominate)
- H3: Edge density changes significantly
- H4: Mean attribution score of top-K features changes
 
Report which hypotheses are supported, partially supported, or rejected.
 
**Stratify results by:**
- True vs False prompts
- Template ID
- Triangle valid vs degenerate
 
### 5B — Semantic Feature Categorization (Version B)
 
#### Neuronpedia Cache (`utils/neuronpedia.py`)
 
The cache is a persistent JSON file at `/workspace/cache/neuronpedia_cache.json`. Schema:
 
```json
{
  "16_14143": {
    "layer": 16,
    "feat_idx": 14143,
    "label": "assertions of truth/falsity, logical statements",
    "top_tokens": ["True", "False", "assert"],
    "fetched_at": "2026-03-01T14:23:00",
    "category": "boolean-logic"
  }
}
```
 
The cache key is `"{layer}_{feat_idx}"` — position (`pos`) is intentionally excluded from the key because the same feature can appear at different token positions across prompts and its semantic identity does not change with position. The `pos` field in `top50_features` is stored in the statistics JSON for the position heatmap analysis (Phase 5B secondary visualization) but must NOT be used as part of the Neuronpedia cache key. Cursor must not modify this design.
 
**Fetch logic in `utils/neuronpedia.py`:**
 
```python
def get_feature_info(layer: int, feat_idx: int, cache: dict) -> dict:
    key = f"{layer}_{feat_idx}"
    if key in cache:
        return cache[key]
   
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{layer}-{NEURONPEDIA_SET}/{feat_idx}"
    # fetch, parse label and top_tokens
    # classify using categorize_feature()
    # write to cache
    # save cache to disk
    return cache[key]
```
 
Add a rate-limiting delay between Neuronpedia requests (minimum 0.5 seconds between calls) to avoid being rate-limited. Save the cache to disk after every 10 new fetches, not just at the end of the run.

Manual override is allowed through `cache/category_overrides.json` keyed by `"{layer}_{feat_idx}"`. The categorizer must apply overrides first, then fallback to keyword matching.
 
#### Feature Categorization (`utils/feature_categorizer.py`)
 
Classification is keyword-based on the Neuronpedia label string. The six categories and their keyword lists:
 
```python
CATEGORIES = {
    "geometry": [
        "triangle", "triangles", "triangular", "side", "sides", "angle", "angles",
        "polygon", "polygons", "hypotenuse", "vertex", "vertices", "geometric",
        "equilateral", "isosceles", "shape", "shapes", "spatial"
    ],
    "math-general": [
        "equation", "equations", "formula", "arithmetic", "calculation",
        "mathematical", "math", "number", "integer", "expression", "notation",
        "algebra", "inequality", "inequalities", "operator", "symbol", "symbols"
    ],
    "boolean-logic": [
        "true", "false", "boolean", "logical", "logic", "conditional",
        "assertion", "assert", "truth", "falsity", "predicate", "valid", "invalid"
    ],
    "format-template": [
        "answer", "statement", "delimiter", "punctuation", "colon", "period",
        "template", "format", "instruction", "prompt", "question", "syntax"
    ],
    "language-comparative": [
        "greater", "less", "more", "fewer", "larger", "smaller", "than",
        "comparison", "comparative", "ordinal", "ordering", "rank", "ranking",
        "exceed", "exceeds", "above", "below", "higher", "lower"
    ],
    "other": []   # catch-all
}
```
 
Classification rule: assign to the first category whose keyword list has at least one match in the lowercased label string. Priority order: geometry > boolean-logic > language-comparative > math-general > format-template > other. If no keywords match, assign "other".
 
The category is stored in the cache alongside the label — reclassification is possible by re-passing cached labels through the classifier without re-fetching from Neuronpedia.
 
**This is an on-demand system.** Only classify features that appear in the top-50 across the analysis prompt set. Do not attempt to classify all 16,000 features.
 
#### Fingerprint Computation
 
For each phase (base, lora_triangle) and for each prompt, the top-50 features are already stored in the statistics JSON. Aggregate across all prompts:
 
```python
def compute_fingerprint(stats: list[dict], cache: dict) -> dict:
    category_counts = defaultdict(int)
    total_feature_appearances = 0
   
    for entry in stats:
        for (layer, pos, feat_idx) in entry["top50_features"]:
            info = get_feature_info(layer, feat_idx, cache)
            category_counts[info["category"]] += 1
            total_feature_appearances += 1
   
    # Normalize to proportions
    fingerprint = {
        cat: count / total_feature_appearances
        for cat, count in category_counts.items()
    }
    return fingerprint
```
 
**Output:** A fingerprint is a dictionary mapping each of the 6 categories to a proportion (0–1, summing to 1). Compute one fingerprint per phase.
 
#### Visualization
 
Produce the following visualizations:
 
**Primary — fingerprint bar chart:** A stacked bar chart or grouped bar chart showing the six category proportions for each of the two phases (base, lora_triangle), broken down further by True vs False prompts. This is the primary visual output of Version B.
 
**Secondary — category × token position heatmap:** For each phase and each category, compute the distribution of token positions at which top-50 features appear. Produce a heatmap with categories on one axis and token positions (0–29, covering the full prompt length) on the other axis, with cell values being the normalized count of feature appearances. This connects Version B directly to the pos=21 / Answer-slot finding from the baseline notebooks. Specifically, ask: after fine-tuning, does the format-template category shift away from pos=21 toward content token positions (pos 5–18)?
 
```python
def compute_position_heatmap(stats: list[dict], cache: dict) -> dict:
    # Returns {category: {position: count}} for all phases
    position_counts = defaultdict(lambda: defaultdict(int))
    for entry in stats:
        for (layer, pos, feat_idx) in entry["top50_features"]:
            info = get_feature_info(layer, feat_idx, cache)
            position_counts[info["category"]][pos] += 1
    return position_counts
```
 
**Tertiary — top-10 feature list per category:** For each category, a list of the top-10 most frequently appearing features across the analysis set (by appearance count), with their Neuronpedia labels. This makes the categorization inspectable and provides qualitative grounding for the quantitative fingerprint.
 
#### Hypotheses for Version B
 
- H5: The base model's fingerprint is dominated by format-template features (consistent with your notebook finding that pos=21 / Answer: slot dominates)
- H6: After fine-tuning on triangle inequality data (FT option 1), the geometry category proportion increases
- H7: After fine-tuning on broad math data (FT option 2), the geometry proportion does not increase but math-general does
- H8: The boolean-logic category proportion increases after both fine-tuning runs
 
---
 
## Utility Modules
 
### `utils/prompt_generator.py`
 
Must export:
- `verify_triangle_claim(a, b, c, claim_type: str) -> bool` — programmatic ground truth
- `generate_triangle_prompts(n: int, seed: int) -> list[dict]` — full dataset generation
- `generate_math_broad_prompts(n: int, seed: int) -> list[dict]` — optional Stage 5 expansion only
- `split_dataset(prompts: list[dict], train_frac, eval_frac, analysis_frac, seed) -> dict`
 
### `utils/graph_statistics.py`
 
Must export:
- `compute_statistics(graph, prompt_id: str, phase: str) -> dict` — returns the output schema from Phase 2, including `mean_error_node_weight`
- `load_statistics(path: str) -> list[dict]`
- `aggregate_statistics(stats: list[dict]) -> dict` — mean, std, median per statistic including error node weight
 
### `utils/neuronpedia.py`
 
Must export:
- `load_cache(path: str) -> dict`
- `save_cache(cache: dict, path: str) -> None`
- `get_feature_info(layer: int, feat_idx: int, cache: dict) -> dict`
- `fetch_and_cache_batch(features: list[tuple], cache: dict) -> dict` — fetches all uncached features with rate limiting
 
### `utils/feature_categorizer.py`
 
Must export:
- `categorize_label(label: str) -> str` — keyword-based classification
- `apply_category_overrides(layer: int, feat_idx: int, category: str, override_map: dict) -> str`
- `compute_fingerprint(stats: list[dict], cache: dict) -> dict`
- `compute_position_heatmap(stats: list[dict], cache: dict) -> dict` — returns `{category: {position: count}}`
- `get_top_features_by_category(stats: list[dict], cache: dict, category: str, n: int) -> list`
 
---
 
## Infrastructure Notes
 
- Use relative project paths rooted at `my_work/` for reproducibility across local and remote environments.
- Model weights can be cached either in HF default cache or a custom path.
- Run `tmux new -s thesis` at the start of each session so long-running cells survive SSH disconnection
- The Neuronpedia cache at `/workspace/cache/neuronpedia_cache.json` must be committed to git or backed up manually — it accumulates significant value over time
- Attribution graph generation for 75 analysis prompts should fit a 1-2 hour single-GPU budget if batching/offload settings are stable. Run as a script, not interactively, to avoid kernel timeouts: `jupyter nbconvert --to script 02_baseline_attribution.ipynb && python 02_baseline_attribution.py`
- Save intermediate results after every 10 prompts in case of interruption
 
---
 
## What This Does NOT Cover
 
- Literature review, thesis writing, or section drafting
- RunPod setup or SSH configuration (already working)
- Multilingual prompts (deferred to future work)
- Multiple LoRA hyperparameter sweeps (single configuration for now, unless Phase 4 fails quality gate)
- Attention circuit analysis (MLP/CLT circuits only, consistent with circuit-tracer methodology)
- Any model other than `google/gemma-2-2b`
 
---
 
## Reference Papers
 
- Zhao et al. (2025) CRV — arXiv:2510.09312 — structural graph statistics for CoT verification
- Yang et al. (2025) — arXiv:2502.11812 — nodes stable, edges change during fine-tuning
- Prakash et al. (2024) — arXiv:2402.14811 — fine-tuning enhances existing mechanisms
- Li et al. (2025) circuit-tuning — arXiv:2502.06106 — fine-tuning as subgraph search
- Ameisen et al. (2025) circuit-tracer methods — https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Lindsey et al. (2025) biology paper — https://transformer-circuits.pub/2025/attribution-graphs/biology.html
