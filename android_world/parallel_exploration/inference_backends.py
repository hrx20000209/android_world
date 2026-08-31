"""Pluggable streaming inference backends used by Process A."""

from __future__ import annotations

import abc
import dataclasses
import json
import random
import time
from collections.abc import Iterator
from typing import Any, Mapping


@dataclasses.dataclass(frozen=True)
class TokenUpdate:
  token_count: int
  partial_text: str


class InferenceBackend(abc.ABC):

  @property
  @abc.abstractmethod
  def model_name(self) -> str:
    raise NotImplementedError

  @abc.abstractmethod
  def stream(self, prompt: str) -> Iterator[TokenUpdate]:
    raise NotImplementedError


class MockBackend(InferenceBackend):
  """A deterministic, duration-controlled backend requiring no accelerator."""

  def __init__(self, config: Mapping[str, Any], seed: int):
    self._config = config
    self._rng = random.Random(seed)

  @property
  def model_name(self) -> str:
    return str(self._config.get("model", "mock-vlm"))

  def _sample_duration_s(self) -> float:
    distribution = str(self._config.get("distribution", "fixed"))
    if distribution == "fixed":
      value = float(self._config.get("duration_s", 2.0))
    elif distribution == "uniform":
      value = self._rng.uniform(
          float(self._config["min_duration_s"]),
          float(self._config["max_duration_s"]),
      )
    elif distribution == "normal":
      value = self._rng.gauss(
          float(self._config["mean_duration_s"]),
          float(self._config["stddev_duration_s"]),
      )
      value = max(float(self._config.get("min_duration_s", 0.0)), value)
    else:
      raise ValueError(f"Unsupported mock distribution: {distribution}")
    if value < 0:
      raise ValueError("Mock inference duration must be non-negative")
    return value

  def stream(self, prompt: str) -> Iterator[TokenUpdate]:
    del prompt
    duration_s = self._sample_duration_s()
    token_count = max(1, int(self._config.get("token_count", 20)))
    first_token_fraction = float(self._config.get("first_token_fraction", 0.25))
    if not 0 <= first_token_fraction <= 1:
      raise ValueError("first_token_fraction must be between 0 and 1")
    first_delay = duration_s * first_token_fraction
    decode_delay = duration_s - first_delay
    time.sleep(first_delay)
    pieces: list[str] = []
    for index in range(1, token_count + 1):
      if index > 1:
        time.sleep(decode_delay / max(1, token_count - 1))
      pieces.append(f"t{index}")
      yield TokenUpdate(index, " ".join(pieces))


class HttpBackend(InferenceBackend):
  """OpenAI-compatible HTTP streaming backend."""

  def __init__(self, config: Mapping[str, Any]):
    self._config = config

  @property
  def model_name(self) -> str:
    return str(self._config["model"])

  def stream(self, prompt: str) -> Iterator[TokenUpdate]:
    import requests  # Imported lazily so mock mode stays dependency-light.

    endpoint = str(self._config["endpoint"])
    timeout_s = float(self._config.get("timeout_s", 120.0))
    headers = {"Content-Type": "application/json"}
    if self._config.get("api_key"):
      headers["Authorization"] = f"Bearer {self._config['api_key']}"
    response = requests.post(
        endpoint,
        headers=headers,
        json={
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            **dict(self._config.get("request_overrides", {})),
        },
        stream=True,
        timeout=timeout_s,
    )
    response.raise_for_status()
    pieces: list[str] = []
    count = 0
    for raw_line in response.iter_lines(decode_unicode=True):
      if not raw_line or not raw_line.startswith("data:"):
        continue
      data = raw_line[5:].strip()
      if data == "[DONE]":
        break
      item = json.loads(data)
      choice = item.get("choices", [{}])[0]
      text = choice.get("delta", {}).get("content") or choice.get("text") or ""
      if not text:
        continue
      count += 1
      pieces.append(text)
      yield TokenUpdate(count, "".join(pieces))


class LlamaCppBackend(InferenceBackend):
  """Local llama.cpp backend, loaded only when explicitly selected."""

  def __init__(self, config: Mapping[str, Any]):
    self._config = config
    try:
      from llama_cpp import Llama
    except ImportError as exc:
      raise RuntimeError(
          "llamacpp backend requires the optional llama-cpp-python package"
      ) from exc
    self._llm = Llama(
        model_path=str(config["model_path"]),
        n_ctx=int(config.get("n_ctx", 4096)),
        n_gpu_layers=int(config.get("n_gpu_layers", -1)),
    )

  @property
  def model_name(self) -> str:
    return str(self._config.get("model", self._config["model_path"]))

  def stream(self, prompt: str) -> Iterator[TokenUpdate]:
    pieces: list[str] = []
    count = 0
    for item in self._llm.create_completion(
        prompt=prompt,
        stream=True,
        max_tokens=int(self._config.get("max_tokens", 128)),
    ):
      text = item.get("choices", [{}])[0].get("text", "")
      if not text:
        continue
      count += 1
      pieces.append(text)
      yield TokenUpdate(count, "".join(pieces))


def create_backend(config: Mapping[str, Any], seed: int) -> InferenceBackend:
  backend = str(config.get("backend", "mock")).lower()
  if backend == "mock":
    return MockBackend(config, seed)
  if backend == "http":
    return HttpBackend(config)
  if backend == "llamacpp":
    return LlamaCppBackend(config)
  raise ValueError(f"Unsupported inference backend: {backend}")
