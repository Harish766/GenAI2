"""Reproducible English-to-French comparison for ICT4403 Assignment 3.

Run after installing requirements.txt. No scores are fabricated: every result is
computed from generated translations on the saved, shared test split.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sacrebleu
import torch
from datasets import load_dataset
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


SEED = 42
DATASET = "Helsinki-NLP/opus_books"
CONFIG = "en-fr"
MODEL_IDS = {
    "mt5": "google/mt5-small",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma": "google/gemma-3-270m-it",
}
MODEL_ORDER = ("basic", "mt5", "llama", "gemma")
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def words(s):
    return TOKEN_RE.findall(s.lower())


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["en", "fr"])
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_fingerprints(root):
    """Identify the exact saved split used by this invocation."""
    return {name: hashlib.sha256((root / "data" / f"{name}.csv").read_bytes()).hexdigest()
            for name in ("train", "val", "test")}


def validate_splits(rows, expected):
    seen = set()
    for name in ("train", "val", "test"):
        if len(rows[name]) != expected[name]:
            raise ValueError(f"{name} split has {len(rows[name])} rows; expected {expected[name]}")
        for row in rows[name]:
            if not row.get("en", "").strip() or not row.get("fr", "").strip():
                raise ValueError(f"{name} split contains an empty text pair")
            pair = (row["en"].casefold(), row["fr"].casefold())
            if pair in seen:
                raise ValueError("Saved splits contain a duplicate pair or cross-split overlap")
            seen.add(pair)


def prepare_data(root, n_train=2000, n_val=200, n_test=100):
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    paths = {s: data_dir / f"{s}.csv" for s in ("train", "val", "test")}
    if any(p.exists() for p in paths.values()) and not all(p.exists() for p in paths.values()):
        raise ValueError("Only some saved split CSVs exist; restore the complete split before running")
    if all(p.exists() for p in paths.values()):
        existing = {s: read_csv(p) for s, p in paths.items()}
        expected = {"train": n_train, "val": n_val, "test": n_test}
        if {s: len(v) for s, v in existing.items()} != expected:
            raise ValueError("Saved data split sizes differ from requested sizes. Move data/ aside and rerun.")
        manifest = data_dir / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("counts") != expected:
            raise ValueError("Saved split manifest does not match the requested sample counts")
        validate_splits(existing, expected)
        return existing
    raw = load_dataset(DATASET, CONFIG, split="train")
    clean, seen = [], set()
    for row in raw:
        en = normalize(row["translation"]["en"])
        fr = normalize(row["translation"]["fr"])
        key = (en.casefold(), fr.casefold())
        if (3 <= len(en.split()) <= 35 and 3 <= len(fr.split()) <= 40
                and key not in seen):
            seen.add(key)
            clean.append({"en": en, "fr": fr})
    rng = random.Random(SEED)
    rng.shuffle(clean)
    required = n_train + n_val + n_test
    if len(clean) < required:
        raise ValueError(f"Only {len(clean)} usable pairs; requested {required}")
    selected = clean[:required]
    result = {
        "train": selected[:n_train],
        "val": selected[n_train:n_train+n_val],
        "test": selected[n_train+n_val:],
    }
    for s, rows in result.items():
        write_csv(paths[s], rows)
    validate_splits(result, {"train": n_train, "val": n_val, "test": n_test})
    (data_dir / "manifest.json").write_text(json.dumps({
        "dataset": DATASET, "config": CONFIG, "source": "https://huggingface.co/datasets/Helsinki-NLP/opus_books",
        "seed": SEED, "counts": {k: len(v) for k, v in result.items()},
        "selection": "NFC, whitespace normalization, length filter, exact-pair deduplication, seeded shuffle",
    }, indent=2), encoding="utf-8")
    return result


class Vocabulary:
    def __init__(self, rows, language, max_size=12000):
        counts = Counter(t for row in rows for t in words(row[language]))
        self.itos = SPECIALS + [t for t, _ in counts.most_common(max_size-len(SPECIALS))]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, text, max_len):
        return [1] + [self.stoi.get(t, 3) for t in words(text)[:max_len-2]] + [2]

    def decode(self, ids):
        toks = []
        for idx in ids:
            idx = int(idx)
            if idx == 2:
                break
            if idx not in (0, 1):
                toks.append(self.itos[idx] if idx < len(self.itos) else UNK)
        return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(toks))


class ParallelDataset(Dataset):
    def __init__(self, rows, src_vocab, tgt_vocab, max_len):
        self.rows, self.src_vocab, self.tgt_vocab, self.max_len = rows, src_vocab, tgt_vocab, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return (self.src_vocab.encode(row["en"], self.max_len),
                self.tgt_vocab.encode(row["fr"], self.max_len))


def collate(batch):
    src, tgt = zip(*batch)
    def pad(seqs):
        length = max(map(len, seqs))
        return torch.tensor([x + [0] * (length-len(x)) for x in seqs], dtype=torch.long)
    return pad(src), pad(tgt)


def require_finite(value, context):
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"Non-finite {context}; model result is invalid")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class BasicTransformer(nn.Module):
    def __init__(self, src_size, tgt_size, d_model=128, nhead=4, layers=2, ff=256):
        super().__init__()
        self.src_emb = nn.Embedding(src_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        self.net = nn.Transformer(d_model=d_model, nhead=nhead,
                                  num_encoder_layers=layers, num_decoder_layers=layers,
                                  dim_feedforward=ff, dropout=0.1, batch_first=True)
        self.out = nn.Linear(d_model, tgt_size)
        self.scale = math.sqrt(d_model)

    def forward(self, src, tgt):
        src_pad, tgt_pad = src.eq(0), tgt.eq(0)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        hidden = self.net(self.pos(self.src_emb(src) * self.scale),
                          self.pos(self.tgt_emb(tgt) * self.scale),
                          tgt_mask=mask, src_key_padding_mask=src_pad,
                          tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        return self.out(hidden)

    @torch.inference_mode()
    def translate(self, src, max_new_tokens=64, temperature=0):
        generated = torch.ones((src.size(0), 1), dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_new_tokens):
            logits = self(src, generated)[:, -1]
            require_finite(logits, "basic Transformer generation logits")
            if temperature > 0:
                probs = torch.softmax(logits.float() / temperature, -1)
                require_finite(probs, "basic Transformer sampling probabilities")
                if not bool(probs.sum(-1).gt(0).all().item()):
                    raise ValueError("Basic Transformer sampling probabilities sum to zero")
                next_id = torch.multinomial(probs, 1).squeeze(1)
            else:
                next_id = logits.argmax(-1)
            next_id = torch.where(finished, torch.zeros_like(next_id), next_id)
            generated = torch.cat([generated, next_id[:, None]], dim=1)
            finished |= next_id.eq(2)
            if bool(finished.all()):
                break
        return generated


def train_basic(rows, root, device, epochs, batch_size, max_len):
    sv, tv = Vocabulary(rows["train"], "en"), Vocabulary(rows["train"], "fr")
    model = BasicTransformer(len(sv.itos), len(tv.itos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loaders = {s: DataLoader(ParallelDataset(rows[s], sv, tv, max_len),
                             batch_size=batch_size, shuffle=(s == "train"), collate_fn=collate)
               for s in ("train", "val")}
    history, best = [], float("inf")
    path = root / "checkpoints" / "basic.pt"
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        losses = {}
        for split in ("train", "val"):
            model.train(split == "train")
            total, tokens = 0.0, 0
            for src, tgt in loaders[split]:
                src, tgt = src.to(device), tgt.to(device)
                with torch.set_grad_enabled(split == "train"):
                    logits = model(src, tgt[:, :-1])
                    gold = tgt[:, 1:]
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
                    require_finite(loss, f"basic {split} loss at epoch {epoch}")
                    if split == "train":
                        opt.zero_grad()
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        require_finite(norm, "basic gradient norm")
                        opt.step()
                count = int(gold.ne(0).sum())
                total += float(loss) * count
                tokens += count
            losses[split] = total / max(1, tokens)
        history.append({"epoch": epoch, **losses})
        print(f"basic epoch {epoch}: train={losses['train']:.4f} val={losses['val']:.4f}", flush=True)
        if losses["val"] < best:
            best = losses["val"]
            torch.save({"model": model.state_dict(), "src_vocab": sv.itos,
                        "tgt_vocab": tv.itos, "max_len": max_len}, path)
    train_seconds = time.perf_counter() - start
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    (root / "results" / "basic_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model.eval(), sv, tv, train_seconds


def train_mt5(rows, root, device, epochs, batch_size, max_len):
    model_id = MODEL_IDS["mt5"]
    tok = AutoTokenizer.from_pretrained(model_id)
    # mT5 produced NaN losses with FP16 on the target Colab GPU. Keep weights
    # and forward/backward computation in FP32 for this small LoRA run.
    base = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16,
                     lora_dropout=0.05, target_modules=["q", "v"])
    model = get_peft_model(base, cfg)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        losses = {}
        for split in ("train", "val"):
            model.train(split == "train")
            total = 0.0
            samples = 0
            indices = list(range(len(rows[split])))
            if split == "train":
                random.shuffle(indices)
            for off in range(0, len(indices), batch_size):
                batch = [rows[split][i] for i in indices[off:off+batch_size]]
                x = tok(["translate English to French: " + r["en"] for r in batch],
                        padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
                y = tok(text_target=[r["fr"] for r in batch], padding=True,
                        truncation=True, max_length=max_len, return_tensors="pt")["input_ids"].to(device)
                y[y == tok.pad_token_id] = -100
                with torch.set_grad_enabled(split == "train"):
                    loss = model(**x, labels=y).loss
                    require_finite(loss, f"mT5 {split} loss at epoch {epoch}")
                    if split == "train":
                        opt.zero_grad()
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        require_finite(norm, "mT5 gradient norm")
                        opt.step()
                total += float(loss.detach()) * len(batch)
                samples += len(batch)
            losses[split] = total / samples
        history.append({"epoch": epoch, **losses})
        print(f"mt5 epoch {epoch}: train={losses['train']:.4f} val={losses['val']:.4f}", flush=True)
    train_seconds = time.perf_counter() - start
    model.save_pretrained(root / "checkpoints" / "mt5_lora")
    (root / "results" / "mt5_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.eval()
    return model, tok, train_seconds


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def causal_prompt(tok, source):
    messages = [{"role": "user", "content":
                 "Translate the following English sentence into French. Output only the French translation.\nEnglish: " + source}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def causal_inputs(tok, source, device, use_chat=True):
    if use_chat:
        # Chat templates already contain BOS/control tokens. A second tokenizer
        # pass with add_special_tokens=True can duplicate BOS (notably Gemma).
        prompt = causal_prompt(tok, source)
        return tok(prompt, return_tensors="pt", add_special_tokens=False,
                   truncation=True, max_length=256).to(device)
    prompt = ("Translate English to French. Reply with only the French sentence.\n"
              f"English: {source}\nFrench:")
    return tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)


def clean_causal_output(text):
    text = text.strip().splitlines()[0].strip() if text.strip() else ""
    return re.sub(r"^(French|Français)\s*:\s*", "", text, flags=re.IGNORECASE).strip()


@torch.inference_mode()
def generate_one(name, model, tok_or_vocabs, source, device, max_len, temperature=0):
    if name == "basic":
        sv, tv = tok_or_vocabs
        ids = torch.tensor([sv.encode(source, max_len)], device=device)
        return tv.decode(model.translate(ids, max_len, temperature)[0].tolist())
    tok = tok_or_vocabs
    kwargs = {"max_new_tokens": max_len, "do_sample": temperature > 0,
              "num_beams": 1, "remove_invalid_values": True,
              "renormalize_logits": True}
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 0.9
    if name == "mt5":
        x = tok("translate English to French: " + source, return_tensors="pt",
                truncation=True, max_length=max_len).to(device)
        sentinel_ids = [i for i in getattr(tok, "additional_special_tokens_ids", [])
                        if i not in (tok.eos_token_id, tok.pad_token_id)]
        mt5_kwargs = {**kwargs, "no_repeat_ngram_size": 3,
                       "repetition_penalty": 1.15}
        if sentinel_ids:
            mt5_kwargs["bad_words_ids"] = [[i] for i in sentinel_ids]
        out = model.generate(**x, **mt5_kwargs)
        translation = tok.decode(out[0], skip_special_tokens=True).strip()
        if re.search(r"<extra_id_\d+>", translation):
            raise ValueError("mT5 emitted a sentinel token despite suppression")
        return translation
    for use_chat in (True, False):
        x = causal_inputs(tok, source, device, use_chat=use_chat)
        out = model.generate(**x, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                             **kwargs)
        translation = clean_causal_output(
            tok.decode(out[0][x.input_ids.shape[1]:], skip_special_tokens=True))
        if translation:
            return translation
    return ""


def validate_predictions(name, predictions, rows):
    if len(predictions) != len(rows) or not rows:
        raise ValueError(f"{name}: prediction count does not match the nonempty test split")
    if any(not isinstance(p, str) for p in predictions):
        raise ValueError(f"{name}: a prediction is not text")
    empty = sum(not p.strip() for p in predictions)
    if empty > max(1, len(rows) // 10):
        raise ValueError(f"{name}: {empty}/{len(rows)} empty translations; evaluation rejected")
    sentinel = sum(bool(re.search(r"<extra_id_\d+>", p)) for p in predictions)
    if sentinel:
        raise ValueError(f"{name}: {sentinel} translations contain mT5 sentinel tokens")
    return empty


def validate_metrics(metric, expected_size):
    if metric["test_size"] != expected_size:
        raise ValueError("Metric test size does not match the shared test split")
    for key in ("bleu", "chrf"):
        value = metric[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid {key} score")
    for key in ("inference_seconds_total", "inference_seconds_per_sentence"):
        value = metric[key]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if metric["parameters"] <= 0:
        raise ValueError("Parameter count must be positive")
    train_seconds = metric["training_seconds"]
    if train_seconds is not None and (not math.isfinite(train_seconds) or train_seconds < 0):
        raise ValueError("Invalid training duration")


def validate_saved_artifacts(name, root, rows, metric):
    """Reopen artifacts and independently recompute reported corpus scores."""
    result_dir = root / "results"
    saved_metric = json.loads((result_dir / f"{name}_metrics.json").read_text(encoding="utf-8"))
    if saved_metric != metric:
        raise ValueError(f"{name}: saved metrics differ from computed metrics")
    lines = (result_dir / f"{name}_translations.jsonl").read_text(encoding="utf-8").splitlines()
    if len(lines) != len(rows):
        raise ValueError(f"{name}: saved translation count differs from test size")
    saved = [json.loads(line) for line in lines]
    if any(item["id"] != i or item["source"] != row["en"]
           or item["reference"] != row["fr"]
           for i, (item, row) in enumerate(zip(saved, rows))):
        raise ValueError(f"{name}: saved translations do not match the shared test set")
    predictions = [item["prediction"] for item in saved]
    validate_predictions(name, predictions, rows)
    refs = [row["fr"] for row in rows]
    if not math.isclose(sacrebleu.corpus_bleu(predictions, [refs]).score, metric["bleu"], abs_tol=1e-8):
        raise ValueError(f"{name}: saved BLEU cannot be reproduced")
    if not math.isclose(sacrebleu.corpus_chrf(predictions, [refs]).score, metric["chrf"], abs_tol=1e-8):
        raise ValueError(f"{name}: saved chrF cannot be reproduced")


def evaluate(name, model, tok_or_vocabs, rows, root, device, max_len, train_seconds):
    model.eval()
    predictions, timings = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for i, row in enumerate(rows):
        start = time.perf_counter()
        prediction = generate_one(name, model, tok_or_vocabs, row["en"], device, max_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        predictions.append(prediction)
        if (i+1) % 25 == 0:
            print(f"{name}: translated {i+1}/{len(rows)}", flush=True)
    empty_count = validate_predictions(name, predictions, rows)
    refs = [r["fr"] for r in rows]
    metric = {
        "model": name, "model_id": MODEL_IDS.get(name, "BasicTransformer-from-scratch"),
        "test_size": len(rows), "bleu": sacrebleu.corpus_bleu(predictions, [refs]).score,
        "chrf": sacrebleu.corpus_chrf(predictions, [refs]).score,
        "training_seconds": train_seconds,
        "inference_seconds_total": sum(timings),
        "inference_seconds_per_sentence": sum(timings) / len(rows),
        "parameters": count_params(model),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_cuda_memory_gb": (torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None),
        "decoding": f"greedy, maximum {max_len} new tokens",
        "empty_translations": empty_count,
        "changed_digit_sequences": sum(
            re.findall(r"\d+", r["en"]) != re.findall(r"\d+", p)
            for r, p in zip(rows, predictions) if re.search(r"\d", r["en"])),
        "sources_with_digits": sum(bool(re.search(r"\d", r["en"])) for r in rows),
    }
    validate_metrics(metric, len(rows))
    path = root / "results" / f"{name}_translations.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i, (row, pred, seconds) in enumerate(zip(rows, predictions, timings)):
            f.write(json.dumps({"id": i, "source": row["en"], "reference": row["fr"],
                                "prediction": pred, "seconds": seconds}, ensure_ascii=False) + "\n")
    (root / "results" / f"{name}_metrics.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")
    validate_saved_artifacts(name, root, rows, metric)
    return metric


def temperature_study(name, model, tok_or_vocabs, rows, root, device, max_len):
    subset = rows[:8]
    output = []
    for temp in (0.2, 0.7, 1.0):
        seed_all(SEED)
        preds = [generate_one(name, model, tok_or_vocabs, r["en"], device, max_len, temp)
                 for r in subset]
        empty = validate_predictions(name, preds, subset)
        output.append({"temperature": temp, "bleu_on_8": sacrebleu.corpus_bleu(preds, [[r["fr"] for r in subset]]).score,
                       "chrf_on_8": sacrebleu.corpus_chrf(preds, [[r["fr"] for r in subset]]).score,
                       "test_size": len(subset), "empty_outputs": empty,
                       "examples": [{"source": r["en"], "reference": r["fr"], "prediction": p}
                                    for r, p in zip(subset[:3], preds[:3])]})
        for key in ("bleu_on_8", "chrf_on_8"):
            if not math.isfinite(output[-1][key]):
                raise ValueError(f"Non-finite {name} temperature {key}")
    (root / "results" / f"{name}_temperature.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def access_failure_reason(exc):
    """Describe a gated-model access failure without claiming a model was tested."""
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    message = str(exc).lower()
    if "awaiting review" in message or "pending approval" in message or "pending review" in message:
        return "Pending Hugging Face access approval (HTTP 403)."
    if code in (401, 403) or "gated" in message:
        return f"Hugging Face model access unavailable (HTTP {code or 'unknown'})."
    return f"Access check failed ({type(exc).__name__})."


def check_gated_access(model_id):
    """Use a metadata request so no large weights are downloaded to check access."""
    try:
        get_hf_file_metadata(hf_hub_url(model_id, "config.json"), token=True)
        return None
    except Exception as exc:
        return access_failure_reason(exc)


def save_run_status(root, statuses, args, created_files, started_at):
    split_paths = [root / "data" / f"{name}.csv" for name in ("train", "val", "test")]
    payload = {
        "started_at_utc": started_at,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "requested_models": args.models,
        "test_size": args.test_size,
        "split_sha256": (split_fingerprints(root) if all(p.exists() for p in split_paths) else None),
        "models": statuses,
        "current_run_files": sorted(set(created_files)),
    }
    path = root / "results" / "run_status.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def report(root, metrics, statuses, temperature_names, plot_files=None):
    manifest_file = root / "data" / "manifest.json"
    counts = json.loads(manifest_file.read_text(encoding="utf-8"))["counts"] if manifest_file.exists() else None
    split_desc = (f'{counts["train"]:,} train, {counts["val"]:,} validation, and {counts["test"]:,} test pairs'
                  if counts else "the saved train, validation, and test splits")
    rows = ["# English to French Machine Translation Comparison", "",
            "Harish Bharathwaj | 230953594 | ICT4403 Assignment 3", "",
            "## Dataset and method", "",
            f"OPUS Books English-French parallel text was normalized, length-filtered, deduplicated, and shuffled with seed 42. The saved CSV splits are fixed across all four models: {split_desc}. The experiment is configured to train a basic Transformer from scratch, adapt mT5-small with FP32 LoRA, and use zero-shot translation prompts for Llama 3.2 1B Instruct and Gemma 3 270M IT. Primary evaluation uses greedy decoding; mT5 also suppresses sentinel tokens and repeated trigrams, and an empty causal chat output is retried with a plain prompt. Only completed rows below represent validated models.", "",
            "## Measured results", "",
            "| Model | Status | BLEU | chrF | Train seconds | Inference seconds per sentence | Parameters |",
            "|---|---|---:|---:|---:|---:|---:|"]
    test_csv = root / "data" / "test.csv"
    if test_csv.exists():
        split_hash = hashlib.sha256(test_csv.read_bytes()).hexdigest()
        rows[7:7] = [f"Saved test split SHA-256: `{split_hash}`. This identifies the exact evaluation rows.", ""]
    measured = {m["model"]: m for m in metrics}
    for name in MODEL_ORDER:
        if name in measured:
            m = measured[name]
            train = "N/A (prompt only)" if m["training_seconds"] is None else f'{m["training_seconds"]:.1f}'
            rows.append(f'| {name} | Completed | {m["bleu"]:.2f} | {m["chrf"]:.2f} | {train} | {m["inference_seconds_per_sentence"]:.3f} | {m["parameters"]:,} |')
        else:
            reason = statuses[name]["reason"].replace("|", "/").replace("\n", " ")
            label = "Failed" if statuses[name]["status"] == "failed" else "Not run"
            rows.append(f"| {name} | {label}: {reason} | - | - | - | - | - |")
    rows.append("")
    if metrics and plot_files and "results/comparison.png" in plot_files:
        rows += ["![BLEU, chrF, and inference-time comparison](comparison.png)", "",
                 "The comparison plot contains completed models only. Model loading time is excluded from inference timing. Parameter count is total parameters, including frozen weights. Results are specific to this small literary corpus and one hardware setup.", ""]
    elif not metrics:
        rows += ["No models completed in this run, so no quality scores or comparison plot were generated.", ""]
    else:
        rows += ["A comparison plot was not generated in this run; the measured table above remains valid.", ""]
    if len(metrics) > 1:
        winner = max(metrics, key=lambda m: (m["bleu"], m["chrf"]))
        scope = "all four systems" if len(metrics) == 4 else "completed systems only"
        rows += [f'On BLEU, **{winner["model"]}** ranks first among {scope}. This is not a four-model conclusion while any system remains unavailable. Check chrF and individual translations before treating this as a quality judgment.', ""]
    elif metrics:
        rows += ["Only one model completed, so this run cannot rank translation quality across systems.", ""]
    rows += [
             "## Interpretation and limitations", "",
             "The from-scratch Transformer has no multilingual pretraining. mT5 uses multilingual pretraining plus task-specific LoRA adaptation. Llama and Gemma use instruction pretraining and zero-shot prompting. Therefore differences cannot be attributed to architecture alone; model size, tokenizers, pretraining data, and adaptation differ. BLEU and chrF are lexical reference metrics and do not fully capture adequacy or cultural nuance.", "",
             "## Responsible AI review", "",
             "Inspect the available saved test translations for omitted names, added claims, changes in gender or politeness, and mistranslated sensitive material. No translation or safety claim is possible for a model marked Failed or Not run. Literary text may carry dated cultural assumptions. Translation outputs should be labeled as machine generated; human review is needed for medical, legal, safety, and public-facing content. Reduce risk by checking names and numbers, preserving uncertainty and register, using diverse evaluation sets, and allowing users to flag harmful or inaccurate translations.", ""]
    if metrics:
        rows += ["The following automatic checks are error flags, not proof of safety or correctness:", "",
                 "| Model | Empty outputs | Source sentences with digits | Changed digit sequences |",
                 "|---|---:|---:|---:|"]
        for m in metrics:
            rows.append(f'| {m["model"]} | {m["empty_translations"]} | {m["sources_with_digits"]} | {m["changed_digit_sequences"]} |')
        rows.append("")
        examples = {}
        for m in metrics:
            p = root / "results" / f'{m["model"]}_translations.jsonl'
            examples[m["model"]] = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
        rows += ["## Side-by-side translation examples", "",
                 "The examples below come from the identical held-out test rows. Compare adequacy, names, register, omissions, and additions against the source and reference.", ""]
        for idx in range(min(3, len(next(iter(examples.values()))))):
            first = next(iter(examples.values()))[idx]
            rows += [f"### Test item {idx+1}", "", f'**English:** {first["source"]}', "",
                     f'**Reference French:** {first["reference"]}', ""]
            for name, items in examples.items():
                rows += [f'**{name}:** {items[idx]["prediction"]}', ""]
    rows += [
             "## Temperature analysis", "",
             ("Three sampled temperatures were evaluated on the same first eight test items for: " + ", ".join(temperature_names) + ". See their *_temperature.json files. This small exploratory analysis is separate from the greedy primary comparison."
              if temperature_names else "No temperature study was completed in this run."), "",
             "## Reproducibility", "",
             "See README.md for setup, model access, commands, and source links. All scores above were computed by experiment.py from saved translations."]
    failed_studies = [f"{name}: {entry['temperature_study']['reason']}"
                      for name, entry in statuses.items()
                      if entry.get("temperature_study", {}).get("status") == "failed"]
    if failed_studies:
        idx = rows.index("## Reproducibility")
        rows[idx:idx] = ["Optional temperature study failures: " + "; ".join(failed_studies), ""]
    path = root / "results" / "REPORT.md"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def make_plots(root, metrics):
    if not metrics:
        return []
    created = []
    names = [m["model"] for m in metrics]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, label, color in zip(
        axes,
        ("bleu", "chrf", "inference_seconds_per_sentence"),
        ("BLEU", "chrF", "Inference seconds / sentence"),
        ("#2463a6", "#24a091", "#d17c32"),
    ):
        ax.bar(names, [m[key] for m in metrics], color=color)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("English to French translation on the shared test set")
    fig.tight_layout()
    fig.savefig(root / "results" / "comparison.png", dpi=160)
    created.append("results/comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    has_curves = False
    completed_names = {m["model"] for m in metrics}
    for name in ("basic", "mt5"):
        if name not in completed_names:
            continue
        file = root / "results" / f"{name}_history.json"
        if file.exists():
            history = json.loads(file.read_text(encoding="utf-8"))
            ax.plot([h["epoch"] for h in history], [h["train"] for h in history], marker="o", label=f"{name} train")
            ax.plot([h["epoch"] for h in history], [h["val"] for h in history], marker="s", linestyle="--", label=f"{name} validation")
            has_curves = True
    if has_curves:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Token or sequence loss (model-specific)")
        ax.set_title("Training and validation history")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "results" / "training_curves.png", dpi=160)
        created.append("results/training_curves.png")
    plt.close(fig)
    return created


def load_causal_model(name, device):
    model_id = MODEL_IDS[name]
    tok = AutoTokenizer.from_pretrained(model_id)
    # FP32 avoids FP16 non-finite sampling logits on older Colab GPUs.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device).eval()
    return model, tok


def load_for_temperature(name, root, device):
    if name == "basic":
        saved = torch.load(root / "checkpoints" / "basic.pt", map_location=device,
                           weights_only=False)
        sv, tv = Vocabulary([], "en"), Vocabulary([], "fr")
        for vocab, key in ((sv, "src_vocab"), (tv, "tgt_vocab")):
            vocab.itos = saved[key]
            vocab.stoi = {token: i for i, token in enumerate(vocab.itos)}
        model = BasicTransformer(len(sv.itos), len(tv.itos)).to(device)
        model.load_state_dict(saved["model"])
        return model.eval(), (sv, tv)
    if name == "mt5":
        tok = AutoTokenizer.from_pretrained(MODEL_IDS[name])
        base = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_IDS[name], torch_dtype=torch.float32).to(device)
        model = PeftModel.from_pretrained(base, root / "checkpoints" / "mt5_lora")
        return model.to(device).eval(), tok
    return load_causal_model(name, device)


def failure_reason(exc):
    first_line = str(exc).splitlines()[0].strip() if str(exc) else ""
    return f"{type(exc).__name__}: {first_line[:220]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=["basic", "mt5", "llama", "gemma"],
                   default=["basic", "mt5", "llama", "gemma"])
    p.add_argument("--basic-epochs", type=int, default=10)
    p.add_argument("--mt5-epochs", type=int, default=2)
    p.add_argument("--train-size", type=int, default=4000)
    p.add_argument("--val-size", type=int, default=400)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--no-temperature-study", action="store_true")
    args = p.parse_args()
    if any(n <= 0 for n in (args.train_size, args.val_size, args.test_size,
                            args.basic_epochs, args.mt5_epochs, args.max_len)):
        p.error("split sizes, epoch counts, and maximum length must be positive")
    seed_all()
    root = Path(__file__).resolve().parent
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    statuses = {
        name: {"status": "not_run", "reason": "Not selected for this invocation."}
        for name in MODEL_ORDER
    }
    for name in args.models:
        statuses[name] = {"status": "not_run", "reason": "Not started."}
    created_files = []
    save_run_status(root, statuses, args, created_files, started_at)
    rows = prepare_data(root, args.train_size, args.val_size, args.test_size)
    validate_splits(rows, {"train": args.train_size, "val": args.val_size,
                           "test": args.test_size})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; split sizes: {[len(rows[x]) for x in ('train','val','test')]}", flush=True)
    metrics = []
    temperature_names = []
    for name in args.models:
        model = None
        tok_or_vocabs = None
        try:
            if name in ("llama", "gemma"):
                reason = check_gated_access(MODEL_IDS[name])
                if reason:
                    statuses[name] = {"status": "not_run", "reason": reason}
                    print(f"{name}: {reason}", flush=True)
                    continue
            if name == "basic":
                model, sv, tv, train_s = train_basic(
                    rows, root, device, args.basic_epochs, 32, args.max_len)
                tok_or_vocabs = (sv, tv)
            elif name == "mt5":
                model, tok_or_vocabs, train_s = train_mt5(
                    rows, root, device, args.mt5_epochs, 2, args.max_len)
            else:
                model, tok_or_vocabs = load_causal_model(name, device)
                train_s = None
            metric = evaluate(name, model, tok_or_vocabs, rows["test"], root,
                              device, args.max_len, train_s)
            # evaluate validates predictions, scores, and saved artifacts first.
            metrics.append(metric)
            created_files.extend([f"results/{name}_metrics.json",
                                  f"results/{name}_translations.jsonl"])
            if name in ("basic", "mt5"):
                created_files.append(f"results/{name}_history.json")
            statuses[name] = {"status": "completed", "reason": "Validated on the current test split."}
            print(f"{name}: completed; BLEU={metric['bleu']:.2f}, chrF={metric['chrf']:.2f}", flush=True)
        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if name in ("llama", "gemma") and (code in (401, 403) or "awaiting review" in str(exc).lower()):
                statuses[name] = {"status": "not_run", "reason": access_failure_reason(exc)}
            else:
                statuses[name] = {"status": "failed", "reason": failure_reason(exc)}
            print(f"{name}: {statuses[name]['status']} - {statuses[name]['reason']}", flush=True)
        finally:
            save_run_status(root, statuses, args, created_files, started_at)
            del model, tok_or_vocabs
            gc.collect()
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
    try:
        plot_files = make_plots(root, metrics)
        created_files.extend(plot_files)
    except Exception as exc:
        plot_files = []
        print(f"Plots failed: {failure_reason(exc)}", flush=True)
    report(root, metrics, statuses, temperature_names, plot_files)
    created_files.append("results/REPORT.md")
    save_run_status(root, statuses, args, created_files, started_at)
    # Primary results and report are now durable. Optional sampling runs last,
    # so a CUDA sampling failure cannot interrupt another model's evaluation.
    if not args.no_temperature_study:
        for metric in metrics:
            name = metric["model"]
            model = None
            tok_or_vocabs = None
            try:
                model, tok_or_vocabs = load_for_temperature(name, root, device)
                temperature_study(name, model, tok_or_vocabs, rows["test"], root,
                                  device, args.max_len)
                saved = json.loads((root / "results" / f"{name}_temperature.json").read_text(encoding="utf-8"))
                if len(saved) != 3 or any(item["test_size"] != min(8, len(rows["test"])) for item in saved):
                    raise ValueError("Temperature artifact has incorrect sample counts")
                created_files.append(f"results/{name}_temperature.json")
                temperature_names.append(name)
                statuses[name]["temperature_study"] = {"status": "completed"}
            except Exception as exc:
                statuses[name]["temperature_study"] = {
                    "status": "failed", "reason": failure_reason(exc)}
                print(f"{name} temperature study failed: {failure_reason(exc)}", flush=True)
            finally:
                save_run_status(root, statuses, args, created_files, started_at)
                del model, tok_or_vocabs
                gc.collect()
                if device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
        report(root, metrics, statuses, temperature_names, plot_files)
        save_run_status(root, statuses, args, created_files, started_at)
    print("Current-run status and report saved in results/", flush=True)


if __name__ == "__main__":
    main()
