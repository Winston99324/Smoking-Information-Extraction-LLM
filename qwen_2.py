import sys
import json
import gc
import re
from datetime import datetime
from pathlib import Path

import torch


txt_files = sorted(Path("./data").glob("*.txt"))
print(f"Found {len(txt_files)} note files")

NOTES = []
NOTE_IDS = []
for f in txt_files:
    with open(f, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    NOTES.append(content)
    NOTE_IDS.append(f.stem) 

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # "BioMistral/BioMistral-7B"
# MODEL = "Qwen/Qwen3-4B-Instruct-2507"

print(f"torch version : {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    sys.exit("No CUDA device visible. Check nvidia-smi and the driver.")

arch_list = torch.cuda.get_arch_list()
print(f"arch list     : {arch_list}")
print(f"gpu           : {torch.cuda.get_device_name(0)}")
cc_major, cc_minor = torch.cuda.get_device_capability(0)
print(f"compute cap   : {cc_major}.{cc_minor}")

if "sm_75" not in arch_list:
    sys.exit(
        "\nThis torch build has no sm_75 kernels -- it will fail with "
        "'no kernel image is available' the moment it runs a real op.\n"
        "Try the stock AWS DLAMI env (source /opt/pytorch/bin/activate) "
        "instead of a cloned/modified one."
    )

try:
    _ = (torch.randn(64, 64, device="cuda", dtype=torch.float16) @
         torch.randn(64, 64, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    print("gpu matmul    : OK\n")
except Exception as e:
    sys.exit(f"\nGPU matmul failed -- kernels are not usable here:\n{e}")


from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"Loading {MODEL} (fp16)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,  
    device_map="cuda:0",
)
model.eval()
print("Loaded.\n")

PROMPT_DIR = Path(__file__).parent / "prompts"
SYSTEM = (PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()
FEWSHOT = "\n" + (PROMPT_DIR / "fewshot_examples.txt").read_text(encoding="utf-8").strip() + "\n"


def parse_json_output(prefix_and_generated: str, note_id: str) -> dict:
    """Extract and parse the JSON object from a completion, tolerating
    trailing text after the closing brace or a cut-off generation."""
    text = prefix_and_generated.strip()

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
    else:
       
        candidate = text if text.startswith("{") else "{" + text
        candidate = candidate.rstrip().rstrip(",") + "}"

    try:
        parsed = json.loads(candidate)
        parsed["_parse_error"] = None
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse failed for {note_id}: {e}")
        parsed = {
            "smoking_status": None,
            "pack_years": None,
            "quit_year": None,
            "quit_year_ago": None,
            "_parse_error": str(e),
        }
    return parsed


def compute_lcs_eligibility(fields: dict, reference_year: int | None = None) -> dict:
    """
    USPSTF 2021 LCS criteria are: age 50-80, >=20 pack-years, and currently
    smokes or quit within the past 15 years. Age isn't extracted from the
    note here, so this only evaluates the SMOKING half of eligibility.
    Join on patient age (e.g. from MIMIC-IV admissions/patients tables)
    before calling anyone definitively "LCS eligible".

    If only quit_year is given, years since quitting is derived as
    reference_year - quit_year. reference_year should be the year the note
    was written; it defaults to the current year. MIMIC-IV dates are shifted,
    so pass the (shifted) note year from the notes table for MIMIC data.
    """
    status = fields.get("smoking_status")
    pack_years = fields.get("pack_years")
    quit_year_ago = fields.get("quit_year_ago")
    quit_year = fields.get("quit_year")

    if reference_year is None:
        reference_year = datetime.now().year
    if not isinstance(quit_year_ago, (int, float)) and isinstance(quit_year, (int, float)):
        years_since = reference_year - quit_year
        quit_year_ago = years_since if years_since >= 0 else None

    pack_years_ok = isinstance(pack_years, (int, float)) and pack_years >= 20

    if status == "current":
        recency_ok = True
    elif status == "former":
        recency_ok = isinstance(quit_year_ago, (int, float)) and quit_year_ago <= 15
    else:
        recency_ok = False

    return {
        "smoking_criteria_met": bool(pack_years_ok and recency_ok),
        "pack_years_ok": pack_years_ok,
        "recency_ok": recency_ok,
        "years_since_quit": quit_year_ago if status == "former" else None,
        "eligibility_note": "Smoking criteria only -- confirm age 50-80 separately.",
    }


def run(note: str) -> str:
    prompt = f"""{SYSTEM}

{FEWSHOT}
Note: {note}

Output: {{"smoking_status": \""""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=["}"],
            tokenizer=tokenizer,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return '{"smoking_status": "' + text.strip()


results = []
for note_id, note in zip(NOTE_IDS, NOTES):
    print("-" * 60)
    print(f"NOTE ID: {note_id}")
    print(f"length: {len(note)}")
    try:
        raw_output = run(note)
        print(f"OUTPUT : {raw_output}")
        fields = parse_json_output(raw_output, note_id)
        eligibility = compute_lcs_eligibility(fields)
        results.append({
            "note_id": note_id,
            "note_length": len(note),
            "raw_output": raw_output,
            **fields,
            **eligibility,
        })
    except torch.cuda.OutOfMemoryError as e:
        print(f"  [WARN] CUDA OOM on {note_id}: {e}")
        results.append({
            "note_id": note_id,
            "note_length": len(note),
            "raw_output": None,
            "_parse_error": f"CUDA OOM: {e}",
        })
    except Exception as e:
        print(f"  [WARN] Unexpected error on {note_id}: {e}")
        results.append({
            "note_id": note_id,
            "note_length": len(note),
            "raw_output": None,
            "_parse_error": str(e),
        })
    finally:
        torch.cuda.empty_cache()
        gc.collect()

print("-" * 60)

out_dir = Path("./results")
out_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_tag = MODEL.split("/")[-1]
out_path = out_dir / f"smoking_history_{model_tag}_{timestamp}.json"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

n_parsed = sum(1 for r in results if r.get("_parse_error") is None)
n_eligible = sum(1 for r in results if r.get("smoking_criteria_met"))

print(f"Saved {len(results)} results to {out_path}")
print(f"Parsed successfully: {n_parsed} / {len(results)}")
print(f"Smoking-criteria met (pending age check): {n_eligible} / {len(results)}")