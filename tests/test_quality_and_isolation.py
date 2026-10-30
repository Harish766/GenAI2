"""Check quality gates and failure isolation without downloading model weights."""
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_blocked_access import FUNCTIONS, load_functions


class QualityTests(unittest.TestCase):
    def test_mt5_training_is_fp32_and_has_finite_loss_checks(self):
        source = __import__("ast").unparse(FUNCTIONS["train_mt5"])
        self.assertIn("torch.float32", source)
        self.assertIn("require_finite(loss", source)
        self.assertNotIn("autocast", source)
        self.assertNotIn("GradScaler", source)

    def test_shared_split_rejects_overlap(self):
        ns = load_functions("validate_splits")
        row = {"en": "Hello", "fr": "Bonjour"}
        with self.assertRaisesRegex(ValueError, "overlap"):
            ns["validate_splits"]({"train": [row], "val": [row],
                                    "test": [{"en": "Bye", "fr": "Au revoir"}]},
                                   {"train": 1, "val": 1, "test": 1})

    def test_empty_and_sentinel_outputs_are_rejected(self):
        ns = load_functions("validate_predictions")
        rows = [{"en": str(i), "fr": str(i)} for i in range(20)]
        with self.assertRaisesRegex(ValueError, "empty translations"):
            ns["validate_predictions"]("gemma", [""] * 20, rows)
        with self.assertRaisesRegex(ValueError, "sentinel"):
            ns["validate_predictions"]("mt5", ["bonjour <extra_id_0>"] * 20, rows)
        self.assertEqual(ns["validate_predictions"]("basic", ["Bonjour"] * 19 + [""], rows), 1)

    def test_nonfinite_or_out_of_range_metrics_are_rejected(self):
        ns = load_functions("validate_metrics")
        ns["math"] = math
        metric = {"test_size": 200, "bleu": 20.0, "chrf": 30.0,
                  "inference_seconds_total": 12.0,
                  "inference_seconds_per_sentence": 0.06,
                  "parameters": 100, "training_seconds": 5.0}
        ns["validate_metrics"](metric, 200)
        with self.assertRaisesRegex(ValueError, "Invalid bleu"):
            ns["validate_metrics"]({**metric, "bleu": float("nan")}, 200)
        with self.assertRaisesRegex(ValueError, "Invalid chrf"):
            ns["validate_metrics"]({**metric, "chrf": 101.0}, 200)
        with self.assertRaisesRegex(ValueError, "test size"):
            ns["validate_metrics"](metric, 199)

    def test_chat_prompt_does_not_add_bos_twice(self):
        ns = load_functions("causal_prompt", "causal_inputs", "clean_causal_output")
        calls = []
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "<bos><start_of_turn>" + messages[0]["content"]
            def __call__(self, prompt, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(to=lambda device: prompt)
        output = ns["causal_inputs"](Tokenizer(), "Hello", "cpu")
        self.assertTrue(output.startswith("<bos>"))
        self.assertIs(calls[0]["add_special_tokens"], False)
        self.assertEqual(ns["clean_causal_output"](" French: Bonjour\nMore text"), "Bonjour")

    def test_gemma_retries_an_empty_chat_output_with_plain_prompt(self):
        ns = load_functions("causal_prompt", "causal_inputs", "clean_causal_output", "generate_one")
        class Batch(dict):
            def __init__(self):
                super().__init__(input_ids=[1, 2])
                self.input_ids = SimpleNamespace(shape=(1, 2))
            def to(self, device):
                return self
        class Tokenizer:
            eos_token_id, pad_token_id = 1, 0
            def __init__(self):
                self.calls = []
            def apply_chat_template(self, messages, **kwargs):
                return "<bos>chat"
            def __call__(self, prompt, **kwargs):
                self.calls.append(kwargs)
                return Batch()
            def decode(self, ids, **kwargs):
                return "" if ids == [99] else "Bonjour"
        class Model:
            def __init__(self):
                self.calls = []
            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return [[1, 2, 99 if len(self.calls) == 1 else 7]]
        tok, model = Tokenizer(), Model()
        result = ns["generate_one"]("gemma", model, tok, "Hello", "cpu", 64)
        self.assertEqual(result, "Bonjour")
        self.assertEqual(len(model.calls), 2)
        self.assertIs(tok.calls[0]["add_special_tokens"], False)

    def test_mt5_generation_suppresses_sentinels_and_trigrams(self):
        ns = load_functions("generate_one")
        ns["re"] = __import__("re")
        class Batch(dict):
            def to(self, device):
                return self
        class Tokenizer:
            eos_token_id, pad_token_id = 1, 0
            additional_special_tokens_ids = [100, 101]
            def __call__(self, prompt, **kwargs):
                return Batch(input_ids=[1, 2])
            def decode(self, ids, **kwargs):
                return "Bonjour"
        class Model:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [[7]]
        model = Model()
        self.assertEqual(ns["generate_one"]("mt5", model, Tokenizer(), "Hello", "cpu", 64), "Bonjour")
        self.assertEqual(model.kwargs["bad_words_ids"], [[100], [101]])
        self.assertEqual(model.kwargs["no_repeat_ngram_size"], 3)

    def test_all_empty_evaluation_saves_no_metrics(self):
        ns = load_functions("validate_predictions", "evaluate")
        ns["time"] = time
        ns["re"] = __import__("re")
        ns["generate_one"] = lambda *a: ""
        ns["torch"] = SimpleNamespace(cuda=SimpleNamespace())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results").mkdir()
            model = SimpleNamespace(eval=lambda: None)
            rows = [{"en": "Hello", "fr": "Bonjour"} for _ in range(200)]
            with self.assertRaisesRegex(ValueError, "200/200 empty translations"):
                ns["evaluate"]("gemma", model, object(), rows, root,
                               SimpleNamespace(type="cpu"), 64, None)
            self.assertFalse((root / "results" / "gemma_metrics.json").exists())

    def test_saved_score_must_recompute_from_translations(self):
        ns = load_functions("validate_predictions", "validate_saved_artifacts")
        ns["math"] = math
        ns["re"] = __import__("re")
        ns["json"] = json
        ns["sacrebleu"] = SimpleNamespace(
            corpus_bleu=lambda preds, refs: SimpleNamespace(score=10.0),
            corpus_chrf=lambda preds, refs: SimpleNamespace(score=20.0))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results").mkdir()
            rows = [{"en": "Hello", "fr": "Bonjour"}]
            metric = {"model": "basic", "test_size": 1, "bleu": 10.0, "chrf": 20.0}
            (root / "results" / "basic_metrics.json").write_text(json.dumps(metric), encoding="utf-8")
            (root / "results" / "basic_translations.jsonl").write_text(
                json.dumps({"id": 0, "source": "Hello", "reference": "Bonjour",
                            "prediction": "Bonjour"}) + "\n", encoding="utf-8")
            ns["validate_saved_artifacts"]("basic", root, rows, metric)
            ns["sacrebleu"].corpus_bleu = lambda *a: SimpleNamespace(score=9.0)
            with self.assertRaisesRegex(ValueError, "BLEU cannot be reproduced"):
                ns["validate_saved_artifacts"]("basic", root, rows, metric)

    def test_model_and_temperature_failure_preserve_good_report(self):
        ns = load_functions("split_fingerprints", "save_run_status", "validate_splits",
                            "failure_reason", "report", "main")
        ns["hashlib"] = __import__("hashlib")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            (root / "results").mkdir()
            (root / "data" / "manifest.json").write_text(
                json.dumps({"counts": {"train": 1, "val": 1, "test": 2}}), encoding="utf-8")
            ns["__file__"] = str(root / "experiment.py")
            ns["seed_all"] = lambda: None
            ns["prepare_data"] = lambda *a: {
                "train": [{"en": "One", "fr": "Un"}],
                "val": [{"en": "Two", "fr": "Deux"}],
                "test": [{"en": "Three", "fr": "Trois"},
                         {"en": "Four", "fr": "Quatre"}]}
            ns["torch"] = SimpleNamespace(
                device=lambda value: SimpleNamespace(type=value),
                cuda=SimpleNamespace(is_available=lambda: False))
            ns["train_basic"] = lambda *a: (object(), object(), object(), 2.0)
            ns["train_mt5"] = lambda *a: (_ for _ in ()).throw(ValueError("Non-finite mT5 loss"))
            ns["check_gated_access"] = lambda model_id: "Pending Hugging Face access approval (HTTP 403)."
            ns["make_plots"] = lambda *a: []
            ns["load_for_temperature"] = lambda *a: (object(), object())
            ns["temperature_study"] = lambda *a: (_ for _ in ()).throw(RuntimeError("CUDA invalid probability"))
            def evaluate(name, model, tok, rows, project, device, max_len, train_seconds):
                metric = {"model": name, "bleu": 11.0, "chrf": 22.0,
                          "training_seconds": train_seconds,
                          "inference_seconds_per_sentence": 0.1, "parameters": 100,
                          "empty_translations": 0, "sources_with_digits": 0,
                          "changed_digit_sequences": 0}
                (root / "results" / "basic_metrics.json").write_text(json.dumps(metric), encoding="utf-8")
                (root / "results" / "basic_translations.jsonl").write_text(
                    "".join(json.dumps({"source": row["en"], "reference": row["fr"],
                                        "prediction": "Bonjour"}) + "\n" for row in rows), encoding="utf-8")
                (root / "results" / "basic_history.json").write_text("[]", encoding="utf-8")
                return metric
            ns["evaluate"] = evaluate
            args = ["experiment.py", "--train-size", "1", "--val-size", "1", "--test-size", "2"]
            with patch.object(sys, "argv", args):
                ns["main"]()
            status = json.loads((root / "results" / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["models"]["basic"]["status"], "completed")
            self.assertEqual(status["models"]["mt5"]["status"], "failed")
            self.assertEqual(status["models"]["llama"]["status"], "not_run")
            self.assertEqual(status["models"]["basic"]["temperature_study"]["status"], "failed")
            self.assertNotIn("results/mt5_metrics.json", status["current_run_files"])
            report = (root / "results" / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("| basic | Completed | 11.00 | 22.00", report)
            self.assertIn("| mt5 | Failed: ValueError: Non-finite mT5 loss", report)
            self.assertIn("Optional temperature study failures", report)


if __name__ == "__main__":
    unittest.main()
