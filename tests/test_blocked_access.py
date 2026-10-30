"""Exercise the blocked-model control flow without installing model packages."""
import argparse
import ast
import copy
import gc
import json
import re
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "experiment.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCTIONS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}


def load_functions(*names):
    body = [copy.deepcopy(FUNCTIONS[n]) for n in names]
    for node in body:
        node.decorator_list = []
    code = compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec")
    namespace = {
        "argparse": argparse, "gc": gc, "json": json, "re": re, "Path": Path,
        "datetime": datetime, "timezone": timezone, "SEED": 42,
        "MODEL_ORDER": ("basic", "mt5", "llama", "gemma"),
        "MODEL_IDS": {"llama": "meta-llama/Llama-3.2-1B-Instruct",
                      "gemma": "google/gemma-3-270m-it"},
    }
    exec(code, namespace)
    return namespace


class BlockedAccessTests(unittest.TestCase):
    def test_pending_403_is_labelled_pending(self):
        ns = load_functions("access_failure_reason")
        exc = Exception("403: access is awaiting review")
        exc.response = SimpleNamespace(status_code=403)
        self.assertIn("Pending Hugging Face access approval", ns["access_failure_reason"](exc))

    def test_run_completes_available_models_and_ignores_stale_scores(self):
        ns = load_functions("save_run_status", "report", "validate_splits", "main")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            (root / "results").mkdir()
            # An older result must not enter the current run's comparison.
            (root / "results" / "llama_metrics.json").write_text('{"bleu": 99.99}', encoding="utf-8")
            (root / "data" / "manifest.json").write_text(json.dumps({
                "counts": {"train": 4000, "val": 400, "test": 200}}), encoding="utf-8")
            ns["__file__"] = str(root / "experiment.py")
            ns["seed_all"] = lambda: None
            ns["prepare_data"] = lambda *a: {
                "train": [{"en": f"English {i}", "fr": f"French {i}"} for i in range(4000)],
                "val": [{"en": f"English {i}", "fr": f"French {i}"} for i in range(4000, 4400)],
                "test": [{"en": f"English {i}", "fr": f"French {i}"} for i in range(4400, 4600)],
            }
            ns["torch"] = SimpleNamespace(
                device=lambda x: SimpleNamespace(type=x),
                cuda=SimpleNamespace(is_available=lambda: False),
            )
            ns["train_basic"] = lambda *a: (object(), object(), object(), 3.0)
            ns["train_mt5"] = lambda *a: (object(), object(), 4.0)
            ns["check_gated_access"] = lambda model_id: "Pending Hugging Face access approval (HTTP 403)."
            ns["temperature_study"] = lambda name, *a: (
                root / "results" / f"{name}_temperature.json").write_text("[]", encoding="utf-8")
            ns["make_plots"] = lambda *a: []
            ns["failure_reason"] = lambda exc: f"{type(exc).__name__}: {exc}"
            def fake_evaluate(name, model, tok, rows, project, device, max_len, train_seconds):
                metric = {"model": name, "bleu": 10.0 if name == "basic" else 20.0,
                          "chrf": 30.0, "training_seconds": train_seconds,
                          "inference_seconds_per_sentence": 0.1, "parameters": 100,
                          "empty_translations": 0, "sources_with_digits": 0,
                          "changed_digit_sequences": 0}
                (root / "results" / f"{name}_translations.jsonl").write_text(
                    "".join(json.dumps({"source": row["en"], "reference": row["fr"],
                                        "prediction": "Bonjour"}) + "\n" for row in rows),
                    encoding="utf-8")
                (root / "results" / f"{name}_metrics.json").write_text(json.dumps(metric), encoding="utf-8")
                (root / "results" / f"{name}_history.json").write_text("[]", encoding="utf-8")
                return metric
            ns["evaluate"] = fake_evaluate
            with patch.object(sys, "argv", ["experiment.py", "--no-temperature-study"]):
                ns["main"]()
            status = json.loads((root / "results" / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["models"]["basic"]["status"], "completed")
            self.assertEqual(status["models"]["mt5"]["status"], "completed")
            self.assertEqual(status["models"]["llama"]["status"], "not_run")
            self.assertEqual(status["models"]["gemma"]["status"], "not_run")
            self.assertNotIn("results/llama_metrics.json", status["current_run_files"])
            self.assertNotIn("results/llama_translations.jsonl", status["current_run_files"])
            self.assertNotIn("results/gemma_translations.jsonl", status["current_run_files"])
            report = (root / "results" / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("| llama | Not run: Pending Hugging Face access approval", report)
            self.assertIn("| gemma | Not run: Pending Hugging Face access approval", report)
            self.assertNotIn("99.99", report)
            self.assertIn("completed systems only", report)

    def test_plots_use_completed_metrics_and_histories_only(self):
        ns = load_functions("make_plots")
        bars, curves = [], []
        class Axis:
            def __init__(self):
                self.tick_params = lambda **kwargs: None
            def bar(self, names, values, **kwargs):
                bars.append(list(names))
            def plot(self, x, y, **kwargs):
                curves.append(kwargs["label"])
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        class Figure:
            def savefig(self, path, **kwargs):
                Path(path).write_bytes(b"plot created in current run")
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        ns["plt"] = SimpleNamespace(
            subplots=lambda *args, **kwargs: (Figure(), [Axis(), Axis(), Axis()] if args == (1, 3) else Axis()),
            close=lambda fig: None,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "results").mkdir()
            for name in ("basic", "mt5"):
                (root / "results" / f"{name}_history.json").write_text(
                    json.dumps([{"epoch": 1, "train": 1.0, "val": 1.1}]), encoding="utf-8")
            metric = {"model": "basic", "bleu": 4.0, "chrf": 20.0,
                      "inference_seconds_per_sentence": 0.2}
            created = ns["make_plots"](root, [metric])
            self.assertTrue(all(names == ["basic"] for names in bars))
            self.assertEqual(curves, ["basic train", "basic validation"])
            self.assertEqual(created, ["results/comparison.png", "results/training_curves.png"])


if __name__ == "__main__":
    unittest.main()
