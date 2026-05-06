import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from router_model import TextRouterModel, collect_router_samples


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_path(text: str) -> Path:
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train lightweight router model from pipeline summary.")
    p.add_argument("--summary-json", required=True, help="Pipeline summary.json path")
    p.add_argument("--router-state-json", required=True, help="Output router state JSON path")
    p.add_argument("--result-json-out", default="", help="Optional training result JSON path")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=0.15)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--balance-strength", type=float, default=0.15)
    p.add_argument("--min-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = _resolve_path(args.summary_json)
    router_state_path = _resolve_path(args.router_state_json)
    summary = _read_json(summary_path)
    samples = collect_router_samples(summary)

    if len(samples) < max(1, int(args.min_samples)):
        raise RuntimeError(f"Not enough router samples: {len(samples)}")

    model, metrics = TextRouterModel.train(
        samples,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
        balance_strength=float(args.balance_strength),
        seed=int(args.seed),
    )
    model.meta.update(
        {
            "source_summary_json": str(summary_path),
            "sample_count": len(samples),
            "label_count": len(model.labels),
        }
    )
    model.save(router_state_path)

    result = {
        "status": "SUCCESS",
        "router_state_json": str(router_state_path),
        "summary_json": str(summary_path),
        "sample_count": len(samples),
        "label_count": len(model.labels),
        "train_accuracy": float(metrics.get("accuracy", 0.0)),
        "labels": model.labels,
        "created_at": _now(),
    }

    out = args.result_json_out.strip()
    if out:
        out_path = _resolve_path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
