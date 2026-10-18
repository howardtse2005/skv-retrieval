# Implementation Plan: The Effect of Target Placement on LLM Retrieval Accuracy in a Synthetic Key-Value Task

## 1. Overview

This plan describes the construction, execution, and analysis of a controlled experiment investigating whether the position of a target fact within a long context affects an LLM's retrieval accuracy. The experiment uses a synthetic key-value (KV) retrieval paradigm, closely following Liu et al. (2023), with compact four-digit subject identifiers and controlled subject–key–value triples. Two conditions are contrasted: **high similarity** (all KV pairs share the same key type, e.g., date of birth) and **low similarity** (mixed key types: dates, colors, names). Three open-weight models are benchmarked locally via Ollama: **Qwen3-8B**, **Llama-3.1-8B**, and **Qwen3.5-9B** (if available; otherwise Qwen3-14B as a proxy).

The core research question: **Does retrieval accuracy vary as a function of target position, and is this effect modulated by the semantic similarity of surrounding distractors?**

The experiment will be conducted in miniconda virtual environment named "kv_pos" with Python 3.12. The virtual environment has been created and can be directly used.

---

## 2. Research Question and Hypotheses

### 2.1 Theoretical Background

Liu et al. (2023) demonstrated a U-shaped performance curve in long-context retrieval: models perform best when relevant information is at the beginning or end of the context and worst when it is in the middle. This "lost-in-the-middle" effect has been attributed to a positional attention bias (U-shaped attention patterns) and to interference from surrounding content.

### 2.2 Hypotheses

| ID | Hypothesis | Prediction |
|---|---|---|
| **H1** | Position effect | Accuracy will be highest at the beginning and end, lowest in the middle (U-shaped curve). |
| **H2** | Similarity effect | High-similarity distractors will reduce overall accuracy compared to low-similarity distractors. |
| **H3** | Interaction | The middle-position deficit will be larger under high similarity than under low similarity. |

**Critical comparison:** If only H1 holds, the effect is primarily architectural (attention sink at boundaries). If H2 or H3 holds, interference-based explanations gain support.

---

## 3. Synthetic Key-Value Dataset Construction

### 3.1 Design Rationale

Following Liu et al. (2023), we use a synthetic KV retrieval task to isolate exact-match retrieval from natural-language reasoning. Unlike Liu et al., who used random UUIDs for values, we retain readable value prefixes while using compact numeric subject IDs and controlled semantic similarity to introduce a controlled manipulation of semantic similarity.

**Advantages of meaningful triples:**

- Enables precise manipulation of distractor similarity (same key type vs. mixed key types).
- Allows clean error analysis: an incorrect answer is unambiguously a retrieval failure (not a reasoning failure).
- Maintains the minimal testbed property of the original paradigm while adding a theoretically relevant factor.

### 3.2 Data Format

Each example consists of:

- **`ordered_kv_records`** : a list of `[subject, key, value]` triples.
- **`target_key`** : the subject–key pair to query.
- **`target_value`** : the expected value.
- **`position`** : the index of the target triple in the list (0-indexed).
- **`similarity_condition`** : `"high"` or `"low"`.

**Example (high similarity):**

```json
{
  "ordered_kv_records": [
    ["Alice", "date of birth", "12 March 1990"],
    ["Bob", "date of birth", "29 June 2004"],
    ["Carol", "date of birth", "5 November 1985"],
    ...
  ],
  "target_key": ["Bob", "date of birth"],
  "target_value": "29 June 2004",
  "position": 1,
  "similarity_condition": "high"
}
```

**Example (low similarity):**

```json
{
  "ordered_kv_records": [
    ["Alice", "favorite color", "red"],
    ["Bob", "date of birth", "29 June 2004"],
    ["Carol", "pet name", "Fluffy"],
    ...
  ],
  "target_key": ["Bob", "date of birth"],
  "target_value": "29 June 2004",
  "position": 1,
  "similarity_condition": "low"
}
```

### 3.3 Generation Procedure

**Step 1: Generate a pool of subjects.**

- Create unique subject IDs in the format `Subject_<four digits>`.
- For high similarity, all subjects share the same key type.
- For low similarity, subjects are assigned different key types.

**Step 2: Define key types and value generators.**

| Key type | Value format | Example |
|---|---|---|
| `date of birth` | DD Month YYYY | "29 June 2004" |
| `favorite color` | Color word | "red" |
| `pet name` | Pet name | "Fluffy" |
| `phone number` | Excluded from high similarity | — |
| `employee ID` | Excluded from high similarity | — |

For **high similarity**, all KV pairs use one shared-attribute key type and
cycle through all eight reusable base values. One-to-one mappings are excluded
because their values identify individual subjects rather than shared attributes.
In low-similarity datasets, suffixes are omitted: size 75 uses 75 categories,
size 140 uses all 100 categories with at most two records per category, and
size 300 uses all 100 categories exactly three times.

For **low similarity**, KV pairs cycle through the 100 key types; when a
haystack is larger than 100 records, the category cycle repeats.

**Step 3: Construct haystacks for each condition and size.**

- **Sizes:** 75, 140, and 300 KV pairs (matching Liu et al., 2023).
- **Examples per dataset:** 100.
- **Dataset files:** 3 sizes × 2 similarity conditions × 5 primary positions = 30 datasets.
- **Total examples:** 3 sizes × 2 similarity conditions × 5 positions × 100 = 3,000 examples per model.

For each example:

1. Generate \( N \) KV pairs (75, 140, or 300).
2. Randomly select one pair as the target.
3. Place the target pair at the designated position.
   The five position variants for the same size, condition, and example index
   reuse the identical generated haystack, subjects, values, and target; only
   the target position changes.
4. Ensure all keys are unique (no duplicate subject–key combinations).
5. Ensure the target value is unique across the haystack (no distractor has the same value).

**Step 4: Position manipulation.**

Liu et al. (2023) varied position across a fine-grained grid. We will use a simplified but sufficient set of positions:

| Position label | Index (for \( N \) pairs) |
|---|---|
| Beginning | 0 (first) |
| 25th percentile | \(\lfloor N/4 \rfloor\) |
| 50th percentile | \(\lfloor N/2 \rfloor\) |
| 75th percentile | \(\lfloor 3N/4 \rfloor\) |
| End | \(N-1\) (last) |

All five positions are primary: **Beginning, 25th percentile, 50th
percentile, 75th percentile, and End**.

**Step 5: Serialise to prompt format.**

The prompt template (adapted from Liu et al., 2023) is:

```
You are given a list of subject–key–value records. Each record is formatted as:
Subject | Key | Value

Records:
{records}

Question: What is {target_subject}'s {target_key}?
Respond with exactly one line and nothing else:
ANSWER: <one answer>
```

Where `{records}` is the list of triples, one per line, formatted as:

```
Alice | date of birth | 12 March 1990
Bob | date of birth | 29 June 2004
...
```

This format is explicit, unambiguous, and easy to parse. It also ensures that the model cannot rely on JSON syntax cues.

**Step 6: Validation.**

- Verify that the target value appears exactly once in the haystack.
- Verify that no distractor contradicts the target.
- Verify that the target key is unique.
- Check token length for each size to ensure it fits within the model's context window.

---

## 4. Model Setup and Benchmarking

### 4.1 Models

| Model | Ollama ID | Parameters | Quantisation | Notes |
|---|---|---|---|---|
| Qwen3-8B | `qwen3:8b` | 8B | Q4_K_M | Pull via `ollama pull qwen3:8b` |
| Llama-3.1-8B | `llama3.1:8b` | 8B | Q4_K_M | Pull via `ollama pull llama3.1:8b` |
| Qwen3.5-9B | *Not currently available* | — | — | Use `qwen3:14b` as a proxy if unavailable, or note as a limitation. |

**Note:** If Qwen3.5-9B is not available on Ollama, substitute `qwen3:14b` (if VRAM permits) or document the deviation.

### 4.2 Context Length Configuration

Ollama's default context window is 4096 tokens. For 300 KV pairs, the prompt may exceed this. Set `num_ctx` appropriately:

- For 75 pairs: ~2,000 tokens (safe with default 4096).
- For 140 pairs: ~3,500 tokens (use `num_ctx` 8192 to leave room for instructions).
- For 300 pairs: ~7,500 tokens (use `num_ctx` 16384 to leave room for instructions).

**Setting `num_ctx`:**

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

Or per-request via the Python API:

```python
import ollama
response = ollama.chat(
    model='qwen3:8b',
    messages=[{'role': 'user', 'content': prompt}],
    options={'num_ctx': 8192, 'temperature': 0, 'top_p': 1, 'num_predict': 20},
    think=False
)
```

### 4.3 Inference Settings

- **Temperature:** 0 (deterministic).
- **Top_p:** 1.0.
- **Max tokens:** 20 (sufficient for a short answer).
- **Seed:** Fixed (e.g., 42) for reproducibility.

---

## 5. Experimental Pipeline

### 5.1 Step-by-Step Procedure

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Generate synthetic KV datasets (3 sizes × 2 similarities) │
│    → 30 dataset files, each with 100 examples                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. For each model (3 models):                                │
│    For each dataset (30 datasets):                           │
│      For each example (100 examples):                        │
│        - Format prompt with target at designated position    │
│        - Send to Ollama                                      │
│        - Record response                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Score responses (exact match)                             │
│    - Record accuracy, invalid responses, error types         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Analyse results                                          │
│    - Accuracy vs. position (U-shaped curve?)                 │
│    - Accuracy vs. similarity                                 │
│    - Interaction effect                                      │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Python Implementation Sketch

```python
import ollama
import json
import pandas as pd

# Load dataset
with open('kv_dataset_75_high_beginning.jsonl', 'r') as f:
    examples = [json.loads(line) for line in f]

results = []

for ex in examples:
    # Build the records string
    records_str = "\n".join(
        f"{subj} | {key} | {val}"
        for subj, key, val in ex['ordered_kv_records']
    )

    # Build the prompt
    prompt = f"""You are given a list of subject–key–value records.
Each record is formatted as: Subject | Key | Value

Records:
{records_str}

Question: What is {ex['target_key'][0]}'s {ex['target_key'][1]}?
Answer:"""

    # Call Ollama
    response = ollama.chat(
        model='qwen3:8b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'num_ctx': 8192, 'temperature': 0, 'seed': 42}
    )

    answer = response['message']['content'].strip()

    # Score only an exact single-line ANSWER: response.
    match = re.fullmatch(r'ANSWER: ([^\r\n]+)', answer)
    correct = int(match is not None and match.group(1).strip().casefold()
                  == ex['target_value'].strip().casefold())

    results.append({
        'model': 'qwen3:8b',
        'dataset': '75_high',
        'position': ex['position'],
        'similarity': ex['similarity_condition'],
        'correct': correct,
        'raw_response': answer
    })

# Save results
pd.DataFrame(results).to_csv('results_qwen3_8b_75_high.csv', index=False)
```

### 5.3 Pilot Phase

Before the main experiment:

- Run 10 examples per dataset (30 datasets × 10 = 300 trials).
- Check for floor/ceiling effects.
- Verify that prompts fit within context limits.
- Verify that responses are parseable.
- Check token generation speed and GPU memory usage.

**Revise materials based on pilot, then freeze the plan.**

---

## 6. Scoring and Analysis

### 6.1 Primary Outcome

**Exact-match accuracy:** 1 if the response is exactly one line in the strict
format `ANSWER: <one answer>` and the answer matches the target value
(case-insensitive), including every suffix and separator, 0 otherwise.
Inference disables model thinking and instructs the model to copy the complete
literal value rather than a shortened semantic prefix.

### 6.2 Secondary Outcomes

- **Invalid-response rate:** proportion of empty, no-answer, or wrong-format responses, or API failures.
- **Error type:** `wrong_answer`, `wrong_format`, `no_answer`, or `api_error`.
- **Response latency:** tokens generated before the answer (optional).

### 6.3 Statistical Analysis

**Primary analysis:**

A 5 (Position: Beginning, 25th, 50th, 75th percentile, End) × 2 (Similarity: High, Low) × 3 (Size: 75, 140, 300) mixed-effects logistic regression with Accuracy as the dependent variable. Model and item (example) as random effects.

**Post-hoc tests:**

- Compare Beginning vs. Middle and Middle vs. End within each similarity condition.
- Bonferroni correction for multiple comparisons.

**Visualisation:**

- Line plot of accuracy vs. position, with separate lines for high and low similarity.
- Bar plot of accuracy by similarity condition.
- Heatmap of accuracy across position × similarity × size.

**Expected pattern if H1 holds:** U-shaped curve with accuracy highest at Beginning and End, lowest at Middle.

**Expected pattern if H3 holds:** The U-shape is more pronounced under high similarity (deeper trough at Middle).

---

## 7. Reproducibility and Documentation

### 7.1 Logging Requirements

For each trial, record:

- Model ID and version (e.g., `qwen3:8b`, Ollama version).
- Quantisation level.
- The raw system prompt, while record-specific user prompts are not stored.
- Sampling settings (`temperature`, `top_p`, `seed`, `num_ctx`).
- Raw response (before scoring).
- Scoring rule and outcome.
- Timestamp.

### 7.2 Code and Data

- Save all dataset generation scripts.
- Save all analysis scripts.
- Preserve raw response files.
- Remove credentials from submitted materials.

### 7.3 AI Use Statement

Include an explicit statement describing AI use for writing, analysis, or model discovery, placed immediately before references.

---

## 8. Timeline

| Week | Task |
|---|---|
| 1 | Finalise design, write pre-registration, generate datasets |
| 2 | Run pilot, freeze materials, set up Ollama models |
| 3 | Run main experiment (3 models × 30 datasets × 100 examples = 9,000 trials) |
| 4 | Score responses, run statistical analysis, create visualisations |
| 5 | Write report (CogSci format), prepare AI Use Statement, submit PDF |

**Estimated runtime:** Assuming ~2 seconds per trial, 9,000 trials ≈ 5 hours of GPU time (parallelisable across models).

---

## 9. Alignment with Grading Rubric

| Component | How This Plan Addresses It |
|---|---|
| **Experimental design (30%)** | Clear research question, two competing explanations (position vs. interference), controlled manipulations, pre-registered predictions, pilot phase |
| **Data analysis (20%)** | Mixed-effects logistic regression, post-hoc tests, separate error-type analysis, U-shaped curve visualisation |
| **Interpretation (20%)** | Connects findings to attention-bias theory and interference accounts; discusses which explanation is supported |
| **Reproducibility (20%)** | Full documentation of models, quantisation, prompts, dataset generation, scoring rules; raw responses preserved |
| **Clarity (10%)** | CogSci formatting, concise writing, clear figures |

---

## 10. References

- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics*.
- Hsieh, C.-Y., et al. (2024). RULER: What's the Real Context Size of Your Long-Context Language Models? *arXiv:2404.06654*.
- Modarressi, A., et al. (2025). NoLiMa: Long-Context Evaluation Beyond Literal Matching. *arXiv*.

---

*This plan is a living document. All changes after pilot should be labelled and transparently reported.*