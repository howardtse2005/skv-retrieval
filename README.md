# Target Placement in Synthetic SKV Retrieval

This repo contains the code implementation for my HKU Department of Psychology's PSYC2036 final essay. The report:


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
