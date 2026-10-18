"""Run synthetic KV retrieval examples against local Ollama models."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ollama
from tqdm import tqdm

MODEL_CANDIDATES = ("qwen3:8b", "llama3.1:8b", "qwen3.5:9b")
MODEL_FALLBACK = "qwen3:14b"
NUM_CTX = {75: 4096, 140: 8192, 300: 16384}
ANSWER_PATTERN = re.compile(r"^ANSWER: ([^\r\n]+)$")
SYSTEM_PROMPT = (
    "You are given a list of subject-key-value records.\n"
    "Each record is formatted as: Subject's key is value.\n"
    "Answer the question using only the records.\n"
    "Respond with exactly one line and nothing else, using this format:\n"
    "ANSWER: <one answer>\n"
    "Do not provide reasoning, explanation, "
    "full sentences, or additional lines.\n"
    "For example, 'ANSWER: red', "
    "not 'ANSWER: Subject_1539's favorite color is red'\n"
    "ANSWER:"
)


def installed_models() -> set[str]:
    try:
        output = subprocess.run(["ollama", "list"], check=True, text=True,
                                capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not query Ollama. Is Ollama installed and running?") from exc
    return {line.split()[0] for line in output.splitlines()[1:] if line.split()}


def resolve_models(requested: list[str] | None) -> tuple[list[str], dict[str, str]]:
    installed = installed_models()
    requested = requested or list(MODEL_CANDIDATES)
    resolved: list[str] = []
    substitutions: dict[str, str] = {}
    for model in requested:
        if model in installed:
            resolved.append(model)
        elif model == "qwen3.5:9b" and MODEL_FALLBACK in installed:
            resolved.append(MODEL_FALLBACK)
            substitutions[model] = MODEL_FALLBACK
        else:
            print(f"Skipping unavailable model: {model}")
    if not resolved:
        raise RuntimeError("None of the requested Ollama models are installed.")
    return list(dict.fromkeys(resolved)), substitutions


def build_prompt(example: dict[str, Any]) -> str:
    target_subject, target_key = example["target_key"]
    matching_records = [
        record for record in example["ordered_kv_records"]
        if record[0] == target_subject and record[1] == target_key
    ]
    if len(matching_records) != 1:
        raise ValueError(
            f"Expected exactly one target record for {target_subject!r}, "
            f"{target_key!r}; found {len(matching_records)}"
        )
    records = "\n".join(
        f"{subject}'s {key} is {value}"
        for subject, key, value in example["ordered_kv_records"]
    )
    return (
        f"Records:\n{records}\n\n"
        f"Question: What is {target_subject}'s {target_key}?"
    )


def score_response(response: str, target_value: str, error: str | None = None) -> tuple[bool, str]:
    if error:
        return False, "api_error"
    response = response.strip()
    if not response or re.search(r"\b(?:don't know|do not know|cannot|unable|no answer)\b",
                                response, re.I):
        return False, "no_answer"
    match = ANSWER_PATTERN.fullmatch(response)
    if match is None:
        return False, "wrong_format"
    if match.group(1).strip().casefold() == target_value.strip().casefold():
        return True, "correct"
    return False, "wrong_answer"


def load_examples(data_dir: Path, pilot: bool, pilot_examples: int) -> list[dict[str, Any]]:
    examples = []
    for path in sorted(data_dir.glob("kv_dataset_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        examples.extend(rows[:pilot_examples] if pilot else rows)
    if not examples:
        raise FileNotFoundError(f"No kv_dataset_*.jsonl files found in {data_dir}")
    return examples


def ollama_version() -> str:
    try:
        return subprocess.run(["ollama", "--version"], check=True, text=True,
                              capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_model(model: str, examples: list[dict[str, Any]], output: Path,
              resume: bool, seed: int) -> None:
    existing_ids: set[str] = set()
    if resume and output.exists():
        with output.open(encoding="utf-8") as handle:
            existing_ids = {json.loads(line)["trial_id"] for line in handle if line.strip()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in tqdm(examples, desc=model):
            trial_id = f"{model}::{example['example_id']}"
            if trial_id in existing_ids:
                continue
            prompt = build_prompt(example)
            options = {
                "num_ctx": NUM_CTX[example["size"]],
                "temperature": 0,
                "top_p": 1,
                "seed": seed,
                "num_predict": 20,
            }
            response_text = ""
            api_error = None
            started = time.perf_counter()
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options=options,
                    think=False,
                )
                response_text = response["message"]["content"].strip()
            except (ollama.ResponseError, KeyError, ConnectionError, TimeoutError) as exc:
                api_error = f"{type(exc).__name__}: {exc}"
            correct, error_type = score_response(response_text, example["target_value"], api_error)
            row = {
                "trial_id": trial_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "ollama_version": ollama_version(),
                "platform": platform.platform(),
                "dataset": f"{example['size']}_{example['similarity_condition']}",
                "size": example["size"],
                "similarity": example["similarity_condition"],
                "position": example["position"],
                "position_label": example["position_label"],
                "target_key": example["target_key"],
                "target_value": example["target_value"],
                "system_prompt": SYSTEM_PROMPT,
                "raw_response": response_text,
                "correct": int(correct),
                "error_type": error_type,
                "api_error": api_error,
                "latency_seconds": time.perf_counter() - started,
                "sampling_options": options,
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-examples", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    examples = load_examples(args.data_dir, args.pilot, args.pilot_examples)
    models, substitutions = resolve_models(args.models)
    if substitutions:
        print(f"Model substitutions: {substitutions}")
    for model in models:
        suffix = "_pilot" if args.pilot else ""
        run_model(model, examples, args.output_dir / f"results_{model.replace(':', '_')}{suffix}.jsonl",
                  args.resume, args.seed)


if __name__ == "__main__":
    main()
