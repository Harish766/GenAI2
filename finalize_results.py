"""Audit the Colab backup and prepare an honest three-model submission report.

This never changes assignment_results_backup.zip or reruns model training.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / "assignment_results_backup.zip"
RESULTS = ROOT / "results"
COMPLETED = ("basic", "llama", "gemma")
TEMPERATURES = (0.2, 0.7, 1.0)
MODEL_LABELS = {
    "basic": "Basic Transformer",
    "mt5": "mT5-small with LoRA",
    "llama": "Llama 3.2 1B Instruct",
    "gemma": "Gemma 3 270M IT",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_number(value, label, minimum=0, maximum=None):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} is outside its valid range")


def load_json(archive, name):
    return json.loads(archive.read(name).decode("utf-8"))


def check_backup():
    if not BACKUP.is_file():
        raise FileNotFoundError(BACKUP)
    with ZipFile(BACKUP) as archive:
        if archive.testzip() is not None:
            raise ValueError("The backup ZIP has a corrupt member")
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            Path(name).is_absolute() or ".." in Path(name).parts for name in names
        ):
            raise ValueError("The backup ZIP has duplicate or unsafe paths")
        required = {"results/run_status.json", "results/comparison.png",
                    "results/training_curves.png", "results/basic_history.json",
                    "results/mt5_history.json", "experiment.py", "README.md"}
        required.update(f"results/{name}_{suffix}" for name in COMPLETED
                        for suffix in ("metrics.json", "translations.jsonl", "temperature.json"))
        missing = required - set(names)
        if missing:
            raise ValueError(f"Required evidence missing: {sorted(missing)}")
        original_status = load_json(archive, "results/run_status.json")
        if original_status["requested_models"] != ["mt5"]:
            raise ValueError("Expected the archived status to describe an mT5-only attempt")
        mt5 = original_status["models"]["mt5"]
        if mt5["status"] != "failed" or "sentinel token" not in mt5["reason"]:
            raise ValueError("The archived mT5 failure is not the reported sentinel error")
        if any(f"results/mt5_{suffix}" in names for suffix in ("metrics.json", "translations.jsonl")):
            raise ValueError("mT5 has scores or translations despite its failed status")
        if original_status["test_size"] != 200:
            raise ValueError("The archived status does not use the required 200-item test set")
        if sha256(archive.read("experiment.py")) != sha256((ROOT / "experiment.py").read_bytes()):
            raise ValueError("Local experiment.py differs from the code archived with the results")

        metrics, translations, temperatures, histories = {}, {}, {}, {}
        for name in COMPLETED:
            metric = load_json(archive, f"results/{name}_metrics.json")
            rows = [json.loads(line) for line in archive.read(
                f"results/{name}_translations.jsonl").decode("utf-8").splitlines()]
            if metric["model"] != name or metric["test_size"] != len(rows) or len(rows) != 200:
                raise ValueError(f"{name} does not have exactly 200 matching metric/test records")
            if [row["id"] for row in rows] != list(range(200)):
                raise ValueError(f"{name} translation IDs are missing or out of order")
            if any(not all(isinstance(row[key], str) and row[key].strip()
                           for key in ("source", "reference", "prediction")) for row in rows):
                raise ValueError(f"{name} has an empty source, reference, or prediction")
            if any(re.search(r"<extra_id_\d+>", row["prediction"]) for row in rows):
                raise ValueError(f"{name} has an mT5 sentinel in a completed output")
            if metric["empty_translations"] != 0:
                raise ValueError(f"{name} empty-output count disagrees with saved translations")
            for key in ("bleu", "chrf"):
                finite_number(metric[key], f"{name} {key}", maximum=100)
            for key in ("inference_seconds_total", "inference_seconds_per_sentence"):
                finite_number(metric[key], f"{name} {key}")
            finite_number(metric["parameters"], f"{name} parameters", minimum=1)
            if metric["training_seconds"] is not None:
                finite_number(metric["training_seconds"], f"{name} training seconds")
            times = [row["seconds"] for row in rows]
            for value in times:
                finite_number(value, f"{name} per-item seconds")
            if not math.isclose(sum(times), metric["inference_seconds_total"], rel_tol=1e-9):
                raise ValueError(f"{name} total inference time differs from per-item times")
            if not math.isclose(sum(times) / 200, metric["inference_seconds_per_sentence"], rel_tol=1e-9):
                raise ValueError(f"{name} average inference time differs from per-item times")
            with_digits = sum(bool(re.search(r"\d", row["source"])) for row in rows)
            changed_digits = sum(
                re.findall(r"\d+", row["source"]) != re.findall(r"\d+", row["prediction"])
                for row in rows if re.search(r"\d", row["source"]))
            if (with_digits != metric["sources_with_digits"]
                    or changed_digits != metric["changed_digit_sequences"]):
                raise ValueError(f"{name} digit diagnostics disagree with saved translations")
            metrics[name], translations[name] = metric, rows

            study = load_json(archive, f"results/{name}_temperature.json")
            if len(study) != 3 or tuple(item["temperature"] for item in study) != TEMPERATURES:
                raise ValueError(f"{name} temperature settings are incomplete")
            for item in study:
                if item["test_size"] != 8 or not 0 <= item["empty_outputs"] <= 8:
                    raise ValueError(f"{name} temperature sample count is invalid")
                for key in ("bleu_on_8", "chrf_on_8"):
                    finite_number(item[key], f"{name} {key}", maximum=100)
                if len(item["examples"]) != 3:
                    raise ValueError(f"{name} temperature examples are missing")
                for index, example in enumerate(item["examples"]):
                    if (example["source"] != rows[index]["source"]
                            or example["reference"] != rows[index]["reference"]):
                        raise ValueError(f"{name} temperature examples use a different test split")
            temperatures[name] = study

        shared = [(row["source"], row["reference"]) for row in translations["basic"]]
        if any(shared != [(row["source"], row["reference"]) for row in translations[name]]
               for name in ("llama", "gemma")):
            raise ValueError("Completed models do not share identical test source/reference rows")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=["en", "fr"])
        writer.writeheader()
        writer.writerows({"en": source, "fr": reference} for source, reference in shared)
        test_hash = sha256(buffer.getvalue().encode("utf-8"))
        if test_hash != original_status["split_sha256"]["test"]:
            raise ValueError("Reconstructed test CSV does not match the saved test-split SHA-256")

        for name in ("basic", "mt5"):
            history = load_json(archive, f"results/{name}_history.json")
            expected_epochs = 10 if name == "basic" else 2
            if len(history) != expected_epochs:
                raise ValueError(f"{name} training history has the wrong epoch count")
            for index, epoch in enumerate(history, 1):
                if epoch["epoch"] != index:
                    raise ValueError(f"{name} history epochs are out of order")
                finite_number(epoch["train"], f"{name} training loss")
                finite_number(epoch["val"], f"{name} validation loss")
            histories[name] = history
        for name in ("comparison", "training_curves"):
            with Image.open(io.BytesIO(archive.read(f"results/{name}.png"))) as image:
                image.verify()
                if image.width < 600 or image.height < 300:
                    raise ValueError(f"{name} plot resolution is too small")

        return {
            "source_archive_sha256": sha256(BACKUP.read_bytes()),
            "original_status": original_status,
            "metrics": metrics,
            "translations": translations,
            "temperatures": temperatures,
            "histories": histories,
            "test_hash": test_hash,
            "source_copies": {
                name: sum(row["prediction"].strip().casefold()
                          == row["source"].strip().casefold()
                          for row in translations[name]) for name in COMPLETED},
            "copied_files": sorted(required - {"results/run_status.json", "experiment.py", "README.md"}),
        }


def write_report(evidence):
    metrics = evidence["metrics"]
    temperatures = evidence["temperatures"]
    rows = [
        "# English to French Machine Translation Comparison", "",
        "**Harish Bharathwaj | 230953594 | ICT4403 Foundations of Generative AI Assignment 3**", "",
        "## Result and evidence status", "",
        "Three systems have saved, validated 200-item test outputs: Basic Transformer, Llama 3.2 1B Instruct, and Gemma 3 270M IT. mT5-small was fine-tuned for two epochs, but generation raised `ValueError: mT5 emitted a sentinel token despite suppression`. It has no valid 200-item translation file, BLEU, chrF, or inference timing. Llama ranks first by the recorded BLEU and chrF scores **among the three completed systems**; no four-model winner can be claimed.", "",
        "The backup combines completed-model artifacts from an earlier run with a later mT5-only failure status. The 200 source/reference rows in all three completed translation files are identical. Reconstructing the test CSV gives SHA-256 `" + evidence["test_hash"] + "`, matching the saved split hash. The original mT5-only status is preserved as `mt5_attempt_run_status.json`; `run_status.json` identifies this reconciled submission.", "",
        "## Dataset and methodology", "",
        "The code uses the public [OPUS Books English-French parallel corpus](https://huggingface.co/datasets/Helsinki-NLP/opus_books). It normalizes Unicode and whitespace, filters sentence lengths, removes duplicate pairs, shuffles with seed 42, and specifies **4,000 training, 400 validation, and 200 held-out test pairs**. All completed models used the same 200 test pairs, verified by row equality and the test-split hash. The original train/validation CSVs and dataset manifest were not in the backup, so their contents and counts cannot be independently rechecked here.", "",
        "The Basic model is a two-layer PyTorch encoder-decoder Transformer (width 128, four heads) trained from scratch for 10 epochs. mT5-small received rank-8 LoRA adaptation for two epochs in FP32; its finite training history does not establish successful translation. Llama and Gemma were prompted zero-shot without task-specific training. Primary decoding was greedy with a 64-new-token cap. The mT5 path applied sentinel and repetition controls; causal models used a chat prompt with a plain-prompt retry for empty responses. These decoding and adaptation differences limit architecture-only interpretation.", "",
        "## Measured comparison", "",
        "| System | Status | BLEU | chrF | Train time (s) | Inference total (s) | Inference / sentence (s) | Parameters |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("basic", "mt5", "llama", "gemma"):
        if name == "mt5":
            rows.append("| mT5-small with LoRA | Failed during generation | - | - | Not recorded | - | - | Not recorded |")
            continue
        m = metrics[name]
        train = f"{m['training_seconds']:.2f}" if m["training_seconds"] is not None else "N/A (zero-shot)"
        rows.append(f"| {MODEL_LABELS[name]} | Completed | {m['bleu']:.3f} | {m['chrf']:.3f} | {train} | {m['inference_seconds_total']:.2f} | {m['inference_seconds_per_sentence']:.3f} | {m['parameters']:,} |")
    rows += [
        "", "The archived scores are SacreBLEU values produced by `experiment.py`. This audit verified finite score ranges, exact record counts, shared test rows, timing arithmetic, and diagnostics against the translation files. SacreBLEU was unavailable in the verification environment, so the BLEU and chrF numbers were **not independently recomputed**. Model loading time is excluded from the recorded per-sentence inference time.", "",
        "![BLEU, chrF, and inference-time comparison for completed models](comparison.png)", "",
        "**Figure 1.** The archived comparison plot contains only the three completed models. mT5 has no bar because evaluation failed.", "",
        "The Basic model is smallest (4,698,250 parameters) and fastest (0.116 s/sentence), but its recorded BLEU is 0.665. Llama has the highest recorded BLEU (10.439) and chrF (33.333) at 1,235,814,400 parameters and 0.682 s/sentence. Gemma records BLEU 6.103 and chrF 27.482 at 268,098,176 parameters and 0.644 s/sentence. These are results for this corpus, hardware, prompts, and decoding setup, not general architecture rankings.", "",
        "## Training evidence and mT5 failure", "",
        "The Basic training history contains 10 finite epochs; training loss falls from 7.173 to 4.563 and validation loss from 6.292 to 5.246. The mT5 history contains two finite epochs; training loss falls from 7.825 to 5.098 and validation loss from 3.908 to 3.725. mT5 then emitted a sentinel token such as `<extra_id_0>` during generation despite suppression. The code rejected that output and did not create a validated metric or translation artifact. Its training duration and parameter count are absent from the backup's mT5 metrics because no such metrics file exists.", "",
        "![Basic Transformer training and validation history](training_curves.png)", "",
        "**Figure 2.** The archived training-curve plot shows the completed Basic system only. mT5's finite loss history is described above without implying successful evaluation.", "",
        "## Saved translation examples", "",
        "These are verbatim predictions from the shared held-out test items; they are not edited translations.", "",
    ]
    for index in (0, 1, 3):
        item = evidence["translations"]["basic"][index]
        rows += [f"### Test item {index + 1}", "",
                 f"**English:** {item['source']}", "",
                 f"**Reference French:** {item['reference']}", ""]
        for name in COMPLETED:
            rows += [f"**{MODEL_LABELS[name]}:** {evidence['translations'][name][index]['prediction']}", ""]
    rows += [
        "## Error analysis and responsible AI", "",
        "The saved examples expose distinct error patterns. In item 1, the Basic model repeats *le capitaine*; Llama describes Prudence as seated although the English says she was standing; Gemma keeps the two people separate but paraphrases the scene. In item 2, Gemma returns the English sentence unchanged. Across all 200 saved outputs, exact English-source copies occur in 0 Basic, 5 Llama, and 45 Gemma predictions. This is a simple failure indicator, not a full language-quality test. In item 3 of the test set, the French reference includes an extra sentence absent from the English source, showing possible corpus alignment noise that can affect lexical scores.", "",
        "Named entities, gender, social roles, politeness, and cultural references can be distorted. The Prudence stance change is a concrete role error; it should not be silently accepted in a sensitive translation. Digit checks found 4 sources with digits: all 4 digit sequences changed in Basic outputs and 2 changed in each Llama and Gemma output. For medical, legal, safety, or public-facing text, retain the source, require human review, check names and numbers, flag source-language copying and repetition, and allow correction. BLEU and chrF measure reference overlap, not factual adequacy, bias, or safety. No model-specific claim about untested sensitive content is made.", "",
        "## Sampling-temperature analysis", "",
        "The optional study sampled the first **8** shared test items at temperatures 0.2, 0.7, and 1.0. These are exploratory scores on a much smaller sample than the 200-item greedy comparison. The backup contains valid study files for Basic, Llama, and Gemma; mT5 has none.", "",
        "| Model | Temperature | BLEU on 8 | chrF on 8 | Empty outputs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in COMPLETED:
        for item in temperatures[name]:
            rows.append(f"| {MODEL_LABELS[name]} | {item['temperature']:.1f} | {item['bleu_on_8']:.3f} | {item['chrf_on_8']:.3f} | {item['empty_outputs']} |")
    rows += [
        "", "The small sample does not support a stable temperature ranking. Its scores should not be compared directly with the 200-item greedy scores.", "",
        "## Conclusion and limitations", "",
        "Llama is the best-performing **completed** model by both recorded lexical metrics on the shared 200-item test set. The Basic model is much smaller and faster but often repetitive; Gemma has 45 exact source copies. Because mT5 failed during generation, the assignment's requested four-model performance comparison remains incomplete. The report and plots do not assign mT5 a translation, score, or inference time. The archive lacks the original split CSVs and manifest, and this audit could not independently rerun SacreBLEU. Results also reflect a small literary subset, possible alignment noise, one T4 run, and differing pretraining and prompting. Human review remains necessary.", "",
        "## Reproducibility and sources", "",
        "The source code, requirements, original model metrics and translations, histories, plots, archived mT5 failure status, reconciled status, and validation record are supplied with this report. Test-split SHA-256: `" + evidence["test_hash"] + "`. Dataset: [Helsinki-NLP OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books). Model pages: [mT5-small](https://huggingface.co/google/mt5-small), [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [Gemma 3 270M IT](https://huggingface.co/google/gemma-3-270m-it). Repository supplied by the student: [GenAI2](https://github.com/Harish766/GenAI2.git).", "",
    ]
    return "\n".join(rows)


def materialize(evidence):
    RESULTS.mkdir(exist_ok=True)
    with ZipFile(BACKUP) as archive:
        for name in evidence["copied_files"]:
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
        (RESULTS / "mt5_attempt_run_status.json").write_bytes(
            archive.read("results/run_status.json"))
    combined = {
        "record_type": "reconciled_submission_from_backup_not_a_single_invocation",
        "source_archive": BACKUP.name,
        "source_archive_sha256": evidence["source_archive_sha256"],
        "archived_mt5_only_status": "results/mt5_attempt_run_status.json",
        "test_size": 200,
        "recorded_split_sha256": evidence["original_status"]["split_sha256"],
        "reconstructed_test_sha256": evidence["test_hash"],
        "models": {
            name: ({"status": "failed",
                    "reason": evidence["original_status"]["models"]["mt5"]["reason"],
                    "source": "archived mT5-only run status; no validated metrics or translations"}
                   if name == "mt5" else
                   {"status": "completed", "test_size": 200,
                    "source": "metric and translation artifacts in backup; same hashed test split"})
            for name in ("basic", "mt5", "llama", "gemma")
        },
    }
    (RESULTS / "run_status.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    validation = {
        "source_archive_sha256": evidence["source_archive_sha256"],
        "completed_models": list(COMPLETED),
        "failed_models": ["mt5"],
        "translation_rows_per_completed_model": 200,
        "all_completed_models_have_identical_sources_and_references": True,
        "reconstructed_test_csv_sha256_matches_archived_status": True,
        "test_csv_sha256": evidence["test_hash"],
        "metrics_numeric_ranges_and_timing_arithmetic_checked": True,
        "digit_diagnostics_checked_against_saved_translations": True,
        "temperature_artifacts_checked": True,
        "sacrebleu_scores_independently_recomputed": False,
        "sacrebleu_recalculation_limitation": "SacreBLEU is unavailable in this workspace; recorded scores are not recomputed.",
        "original_train_val_csv_and_manifest_present": False,
        "exact_english_source_copies": evidence["source_copies"],
        "note": "The archived run_status is mT5-only. Three completed artifacts came from earlier runs; shared test rows and hash were checked.",
    }
    (RESULTS / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "REPORT.md").write_text(write_report(evidence) + "\n", encoding="utf-8")
    return validation


if __name__ == "__main__":
    evidence = check_backup()
    validation = materialize(evidence)
    print("Audited three completed 200-item models and mT5 failure; wrote results/REPORT.md")
    print("Test split SHA-256:", validation["test_csv_sha256"])
