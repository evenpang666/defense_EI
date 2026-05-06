import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    denom = sum(exps)
    if denom <= 0.0:
        return [1.0 / float(len(logits)) for _ in logits]
    return [x / denom for x in exps]


class TextRouterModel:
    def __init__(
        self,
        *,
        labels: list[str],
        vocab: list[str],
        weights: list[list[float]],
        bias: list[float],
        created_at: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.labels = list(labels)
        self.vocab = list(vocab)
        self.weights = [list(row) for row in weights]
        self.bias = list(bias)
        self.created_at = created_at or _now()
        self.meta = dict(meta or {})
        self._label_to_index = {label: i for i, label in enumerate(self.labels)}
        self._vocab_to_index = {token: i for i, token in enumerate(self.vocab)}

    @staticmethod
    def _encode_counts(text: str, vocab_to_index: dict[str, int]) -> Counter[int]:
        counts: Counter[int] = Counter()
        for token in _tokenize(text):
            idx = vocab_to_index.get(token)
            if idx is not None:
                counts[idx] += 1
        return counts

    def predict_proba(self, text: str) -> list[float]:
        if not self.labels:
            return []
        if not self.vocab:
            return [1.0 / float(len(self.labels)) for _ in self.labels]
        counts = self._encode_counts(text, self._vocab_to_index)
        logits: list[float] = []
        for label_index in range(len(self.labels)):
            score = float(self.bias[label_index])
            row = self.weights[label_index]
            for token_index, count in counts.items():
                score += float(row[token_index]) * float(count)
            logits.append(score)
        return _softmax(logits)

    def predict(self, text: str) -> tuple[str, list[float]]:
        probs = self.predict_proba(text)
        if not probs:
            return "", []
        best_index = max(range(len(probs)), key=probs.__getitem__)
        return self.labels[best_index], probs

    def top_k(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        probs = self.predict_proba(text)
        ranked = sorted(zip(self.labels, probs), key=lambda item: item[1], reverse=True)
        return ranked[: max(1, int(k))]

    @classmethod
    def train(
        cls,
        samples: list[dict[str, str]],
        *,
        epochs: int = 120,
        learning_rate: float = 0.15,
        l2: float = 1e-4,
        balance_strength: float = 0.15,
        seed: int = 42,
    ) -> tuple["TextRouterModel", dict[str, Any]]:
        clean_samples = [
            {"text": str(s.get("text", "")).strip(), "label": str(s.get("label", "")).strip()}
            for s in samples
            if str(s.get("text", "")).strip() and str(s.get("label", "")).strip()
        ]
        if not clean_samples:
            raise ValueError("No router samples provided")

        labels = sorted({s["label"] for s in clean_samples})
        if not labels:
            raise ValueError("No router labels found")

        vocab_set: set[str] = set()
        for sample in clean_samples:
            vocab_set.update(_tokenize(sample["text"]))
        vocab = sorted(vocab_set)
        label_to_index = {label: i for i, label in enumerate(labels)}
        vocab_to_index = {token: i for i, token in enumerate(vocab)}

        class_counts = Counter(sample["label"] for sample in clean_samples)
        total = float(len(clean_samples))
        class_weights = {
            label: total / (float(len(labels)) * float(class_counts[label]))
            for label in labels
        }
        if class_weights:
            mean_weight = sum(class_weights.values()) / float(len(class_weights))
            if mean_weight > 0.0:
                class_weights = {label: weight / mean_weight for label, weight in class_weights.items()}

        weights = [[0.0 for _ in vocab] for _ in labels]
        bias = [0.0 for _ in labels]
        rng = random.Random(seed)
        order = list(range(len(clean_samples)))

        def _sample_weight(label: str) -> float:
            return (1.0 - float(balance_strength)) + float(balance_strength) * float(class_weights.get(label, 1.0))

        for _epoch in range(max(1, int(epochs))):
            rng.shuffle(order)
            for sample_index in order:
                sample = clean_samples[sample_index]
                counts = Counter(vocab_to_index[token] for token in _tokenize(sample["text"]) if token in vocab_to_index)
                if not counts:
                    continue
                logits: list[float] = []
                for label_index in range(len(labels)):
                    score = float(bias[label_index])
                    row = weights[label_index]
                    for token_index, count in counts.items():
                        score += float(row[token_index]) * float(count)
                    logits.append(score)
                probs = _softmax(logits)
                target_index = label_to_index[sample["label"]]
                sample_w = _sample_weight(sample["label"])
                for label_index in range(len(labels)):
                    target = 1.0 if label_index == target_index else 0.0
                    grad = (probs[label_index] - target) * sample_w
                    bias[label_index] -= float(learning_rate) * grad
                    row = weights[label_index]
                    for token_index, count in counts.items():
                        row[token_index] -= float(learning_rate) * (grad * float(count) + float(l2) * row[token_index])

        model = cls(labels=labels, vocab=vocab, weights=weights, bias=bias)
        metrics = model.evaluate(clean_samples)
        model.meta.update(
            {
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "l2": float(l2),
                "balance_strength": float(balance_strength),
                "sample_count": len(clean_samples),
                "class_counts": dict(class_counts),
                "train_accuracy": float(metrics.get("accuracy", 0.0)),
            }
        )
        return model, metrics

    def evaluate(self, samples: list[dict[str, str]]) -> dict[str, Any]:
        clean_samples = [
            {"text": str(s.get("text", "")).strip(), "label": str(s.get("label", "")).strip()}
            for s in samples
            if str(s.get("text", "")).strip() and str(s.get("label", "")).strip()
        ]
        if not clean_samples:
            return {"accuracy": 0.0, "sample_count": 0}
        correct = 0
        for sample in clean_samples:
            predicted, _ = self.predict(sample["text"])
            if predicted == sample["label"]:
                correct += 1
        return {
            "accuracy": float(correct) / float(len(clean_samples)),
            "sample_count": len(clean_samples),
        }

    def save(self, path: Path) -> None:
        payload = {
            "model_type": "text_router_softmax_v1",
            "labels": self.labels,
            "vocab": self.vocab,
            "weights": self.weights,
            "bias": self.bias,
            "created_at": self.created_at,
            "meta": self.meta,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TextRouterModel":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid router model payload: {path}")
        return cls(
            labels=[str(x) for x in raw.get("labels", [])],
            vocab=[str(x) for x in raw.get("vocab", [])],
            weights=[[float(v) for v in row] for row in raw.get("weights", [])],
            bias=[float(v) for v in raw.get("bias", [])],
            created_at=str(raw.get("created_at", "")) or None,
            meta=dict(raw.get("meta", {})) if isinstance(raw.get("meta", {}), dict) else {},
        )


def collect_router_samples(summary: dict[str, Any]) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    explicit_samples = summary.get("router_samples", [])
    if isinstance(explicit_samples, list):
        for entry in explicit_samples:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "")).strip()
            label = str(entry.get("label", "")).strip()
            if text and label:
                samples.append({"text": text, "label": label})

    rounds = summary.get("rounds", [])
    if isinstance(rounds, list):
        for entry in rounds:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("status", "")).strip().upper() != "ATOMIC_VERIFIED":
                continue
            text = str(entry.get("task", "")).strip()
            label = str(entry.get("primitive", "")).strip()
            if text and label:
                samples.append({"text": text, "label": label})

    if not samples:
        task_list = summary.get("tasks", [])
        task_states = summary.get("task_states", [])
        if isinstance(task_list, list) and isinstance(task_states, list):
            for task, state in zip(task_list, task_states):
                if not isinstance(state, dict):
                    continue
                text = str(task).strip()
                label = str(state.get("primitive", "")).strip()
                if text and label:
                    samples.append({"text": text, "label": label})

    return samples
