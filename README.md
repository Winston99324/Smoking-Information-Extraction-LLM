# Smoking Information Extraction

Uses a local LLM to extract smoking history from clinical notes and screen patients against the smoking criteria for **lung cancer screening (LCS)**.

For each free-text note, the pipeline asks an instruction-tuned model (Qwen2.5 by default) for a structured JSON record:

```json
{"smoking_status": "current|former|never|unknown",
 "pack_years": 30,
 "quit_year": 2018,
 "quit_year_ago": null}
```

It then applies the smoking part of the **USPSTF 2021** LCS recommendation:

| Criterion | Rule |
|---|---|
| Pack-years | ≥ 20 |
| Recency | Currently smokes, **or** quit ≤ 15 years ago |
| Age | 50–80 (*not extracted; see [Limitations](#limitations)*) |

## Repository contents

| File | Purpose |
|---|---|
| `qwen_2.py` | Main pipeline. Loads notes, checks the GPU, runs few-shot extraction with Hugging Face `transformers`, parses the JSON, scores eligibility and writes the results. |
| `prompts/system_prompt.txt` | Instructions: the JSON schema and rules for status, pack-years and quit year. Edit this to change the prompt without touching code. |
| `prompts/fewshot_examples.txt` | The worked examples shown to the model before each note. |
| `test_vllm.sh` | Smoke test that starts a vLLM OpenAI-compatible server for the same model. `qwen_2.py` doesn't use it yet. |

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA. The script checks for `sm_75` kernels, which it was written for (e.g. a T4 on AWS).
- Packages:

```bash
pip install torch transformers accelerate
# optional, for test_vllm.sh
pip install vllm
```

## Usage

### 1. Prepare the notes

Put each clinical note in its own `.txt` file under `./data/`. The filename stem becomes the `note_id`.

```
data/
├── 10001234_note1.txt
├── 10001234_note2.txt
└── ...
```

> ⚠️ **Never commit patient data.** `data/` and `results/` are listed in `.gitignore`. MIMIC-IV notes fall under the PhysioNet Credentialed Health Data Use Agreement and must not be shared publicly.

### 2. Run the extraction

```bash
python qwen_2.py
```

To change the model, edit the `MODEL` constant at the top of `qwen_2.py`. `Qwen/Qwen3-4B-Instruct-2507` is included as a commented-out option.

### 3. Output

The results go to `./results/smoking_history_<model>_<timestamp>.json`, with one record per note:

| Field | Description |
|---|---|
| `note_id`, `note_length` | Note identifier and character count |
| `raw_output` | Raw model completion |
| `smoking_status`, `pack_years`, `quit_year`, `quit_year_ago` | Extracted fields |
| `_parse_error` | `null` on success, otherwise the error message (JSON parse failure, CUDA OOM, etc.) |
| `smoking_criteria_met` | `true` when the pack-year and recency criteria both pass |
| `pack_years_ok`, `recency_ok` | The two sub-criteria |
| `years_since_quit` | Years since quitting, taken from `quit_year_ago` or worked out from `quit_year` (former smokers only) |

At the end of a run the script prints how many notes parsed successfully and how many met the smoking criteria.

### Optional: vLLM server

```bash
bash test_vllm.sh
# or override the defaults
MODEL=Qwen/Qwen3-4B-Instruct-2507 PORT=8001 bash test_vllm.sh
```

## How it works

1. **Environment checks.** The script confirms CUDA is available, that the torch build has `sm_75` kernels, and that an fp16 matmul on the GPU succeeds.
2. **Prompting.** It combines a system prompt with the extraction rules, four few-shot examples and the note. The completion is pre-filled with `{"smoking_status": "` to keep the model in JSON form.
3. **Decoding.** Decoding is greedy (`do_sample=False`), capped at 128 new tokens and stopped at the first `}`.
4. **Parsing.** A regex pulls out the first `{...}` object. If the generation was cut off, the script tries to close the brace before parsing.
5. **Eligibility.** `compute_lcs_eligibility()` applies the pack-year and recency rules.

## Limitations

- **Age isn't checked.** `smoking_criteria_met` covers only the smoking criteria. Before calling a patient LCS-eligible, join on age (e.g. the MIMIC-IV `patients` / `admissions` tables).
- **Years since quitting needs a reference year.** If a note gives only `quit_year`, years since quitting is worked out as `reference_year - quit_year`, and `reference_year` defaults to the current year. MIMIC-IV dates are shifted for de-identification, so pass the note's own (shifted) year to `compute_lcs_eligibility(fields, reference_year=...)`.
- Long notes go into the prompt untruncated and can exceed the model's context window or GPU memory. CUDA OOM errors are caught and recorded per note.
- This is a research prototype, not a clinical decision tool.

## Citation

If you use this code, please cite the related work (citation to be added).

## License

Released under the [MIT License](LICENSE).
