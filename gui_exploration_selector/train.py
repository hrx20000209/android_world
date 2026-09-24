#!/usr/bin/env python3
"""Train the V0 GUI exploration selector without third-party packages.

The feature map is deliberately fixed: each vocabulary item contributes one
binary feature, and the model is logistic regression with a bias.  The output
format is also the input format used by the standalone Android executable, so
``weights.txt`` can be copied to a phone without Python.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_vocabulary(path: Path) -> list[str]:
  tokens: list[str] = []
  seen: set[str] = set()
  for raw in path.read_text(encoding="utf-8").splitlines():
    token = raw.strip().lower()
    if token and not token.startswith("#") and token not in seen:
      tokens.append(token)
      seen.add(token)
  if not tokens:
    raise ValueError(f"empty vocabulary: {path}")
  return tokens


def tokenize(*values: object) -> set[str]:
  result: set[str] = set()
  for value in values:
    result.update(TOKEN_RE.findall(str(value or "").lower()))
  return result


def feature_vector(row: dict[str, object], vocabulary: list[str]) -> list[float]:
  fields = row.get("element") if isinstance(row.get("element"), dict) else row
  assert isinstance(fields, dict)
  tokens = tokenize(
      fields.get("text", ""), fields.get("content_desc", ""),
      fields.get("resource_id", ""), fields.get("resourceId", ""),
  )
  return [1.0 if token in tokens else 0.0 for token in vocabulary]


def load_examples(path: Path) -> tuple[list[dict[str, object]], list[float]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, list):
    raise ValueError("training JSON must be an array")
  rows: list[dict[str, object]] = []
  labels: list[float] = []
  for item in payload:
    if not isinstance(item, dict):
      raise ValueError("each training example must be an object")
    label = item.get("label", item.get("target"))
    if label is None:
      raise ValueError("training example is missing label/target")
    rows.append(item)
    labels.append(1.0 if float(label) > 0.0 else 0.0)
  if not rows:
    raise ValueError("training JSON contains no examples")
  return rows, labels


def load_initial_weights(path: Path | None, vocabulary: list[str]) -> tuple[list[float], float]:
  weights = [0.0] * len(vocabulary)
  bias = 0.0
  if path is None:
    return weights, bias
  by_token = {token: index for index, token in enumerate(vocabulary)}
  for raw in path.read_text(encoding="utf-8").splitlines():
    parts = raw.strip().split()
    if len(parts) < 2 or parts[0].startswith("#"):
      continue
    key, value = parts[0].lower(), float(parts[1])
    if key == "bias":
      bias = value
    elif key in by_token:
      weights[by_token[key]] = value
  return weights, bias


def sigmoid(value: float) -> float:
  if value >= 0.0:
    z = math.exp(-value)
    return 1.0 / (1.0 + z)
  z = math.exp(value)
  return z / (1.0 + z)


def train(
    features: list[list[float]], labels: list[float], weights: list[float],
    bias: float, *, epochs: int, learning_rate: float, l2: float,
) -> tuple[list[float], float]:
  if len(features) != len(labels):
    raise ValueError("features and labels have different lengths")
  count = float(len(features))
  for _ in range(max(0, epochs)):
    grad_w = [0.0] * len(weights)
    grad_b = 0.0
    for row, label in zip(features, labels):
      margin = bias + sum(weight * value for weight, value in zip(weights, row))
      error = sigmoid(margin) - label
      grad_b += error
      for index, value in enumerate(row):
        grad_w[index] += error * value
    bias -= learning_rate * grad_b / count
    for index in range(len(weights)):
      weights[index] -= learning_rate * (grad_w[index] / count + l2 * weights[index])
  return weights, bias


def write_weights(path: Path, vocabulary: list[str], weights: list[float], bias: float) -> None:
  lines = [
      "# explorer_selector_weights_v1",
      "version 1",
      f"vocab_size {len(vocabulary)}",
      f"bias {bias:.9g}",
  ]
  lines.extend(f"{token} {weight:.9g}" for token, weight in zip(vocabulary, weights))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
  here = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train", type=Path, default=here / "example_train.json")
  parser.add_argument("--vocabulary", type=Path, default=here / "vocabulary.txt")
  parser.add_argument("--output", type=Path, default=here / "weights.trained.txt")
  parser.add_argument("--weights-in", type=Path,
                      help="optional manually initialized model")
  parser.add_argument("--epochs", type=int, default=300)
  parser.add_argument("--learning-rate", type=float, default=0.25)
  parser.add_argument("--l2", type=float, default=0.001)
  args = parser.parse_args()

  vocabulary = load_vocabulary(args.vocabulary)
  rows, labels = load_examples(args.train)
  features = [feature_vector(row, vocabulary) for row in rows]
  weights, bias = load_initial_weights(args.weights_in, vocabulary)
  weights, bias = train(
      features, labels, weights, bias, epochs=args.epochs,
      learning_rate=args.learning_rate, l2=args.l2,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  write_weights(args.output, vocabulary, weights, bias)
  print(json.dumps({
      "examples": len(rows), "vocabulary": len(vocabulary),
      "epochs": max(0, args.epochs), "output": str(args.output),
  }, ensure_ascii=False))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
