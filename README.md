# Target Placement in Synthetic SKV Retrieval

This repo contains the code implementation for my HKU Department of Psychology's PSYC2036 final essay. 


**Abstract:**

Large language models (LLMs) demonstrate a well-documented "lost-in-the-middle" effect: retrieval accuracy is highest when relevant information appears at the beginning or end of a long context and lowest when it is buried in the middle. However, the mechanism underlying this positional bias remains contested. One account attributes it to architectural attention sink at sequence boundaries (positional bias), while another implicates semantic interference from surrounding distractors (similarity bias). This study manipulates both target position (beginning, 25th percentile, 50th percentile, 75th percentile, end) and distractor similarity (high vs. low semantic similarity) in a synthetic key-value retrieval task (subject-key-value format) administered to six open-weight models via Ollama, three belong to 1B model group and three belong to 3B group. Most models showed a beginning-position advantage, with lower accuracy at the middle and end positions in several conditions. Crucially, the middle-end position deficits were larger under high-similarity distractors than under low-similarity distractors, consistent with an additional semantic similarity bias on top of positional bias. These findings suggest that positional bias in LLM retrieval is not merely a function of token index but is modulated by the semantic competition between the target and its surrounding context. Therefore, we propose a synthetic dataset design with subject-key-value (SKV) triples to manipulate only the information positions or relative semantic similarity to test their isolated effects on LLM retrieval accuracy.

Full report:

[View the report PDF](./HowardTse_PSYC2036A_Essay.pdf)
## Setup

The intended environment is `skv_retrieval` with Python 3.12:

```bash
conda activate skv_retrieval
pip install -r requirements.txt
```

Ollama must be running locally. The benchmark selects
`qwen3.5:9b` when it is installed and otherwise uses `qwen3:14b` as the
documented proxy.

## Workflow

1. Generate the 30 datasets (100 examples per size, similarity condition, and
target position):

```bash
python generate_dataset.py --output-dir data
```

2. Run a 300-trial pilot (10 examples per dataset):

```bash
python run_benchmark.py --data-dir data --output-dir results --pilot
```

or run the full benchmark for all available models:

```bash
python run_benchmark.py --data-dir data --output-dir results
```

3. Analyse one or more result files:

```bash
python analyze_results.py --results-dir results --output-dir analysis
```

Every trial stores the raw system instruction in `system_prompt`, the raw
response, scoring decision, sampling settings, model identifier, and timestamp.
The record-specific user prompt is sent to Ollama but is not stored in results.
Use `--resume` to skip
trial IDs already present in an existing result file. Dataset generation is
seeded (`42` by default), and each example records its own seed-independent
position and target metadata.

Inference disables model thinking (`think=False`) and uses 8,192 tokens for
140-record prompts and 16,384 tokens for 300-record prompts. Responses are scored as `correct`,
`wrong_answer`, or `no_answer`; Ollama/API failures are
recorded separately as `api_error`.
The system prompt requires copying the complete literal value rather than
returning an abbreviated or inferred answer.

The benchmark intentionally does not silently replace an unavailable model:
the fallback substitution is reported in the run metadata and console output.
API failures are recorded as `api_error` trials rather than being discarded.
