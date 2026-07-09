"""
Utilities for capturing MoE router distributions from real workloads.

This is intentionally lightweight and gated behind env vars so it has zero
runtime impact unless explicitly enabled.

Env vars (defaults from vllm.envs):
- VLLM_MOE_ROUTER_CAPTURE=0/1: enable capture (default: 0).
- VLLM_MOE_ROUTER_CAPTURE_DIR=/path: output directory for per-process dumps
  (default: /tmp).
- VLLM_MOE_ROUTER_CAPTURE_RANK=N: only capture on the given torch.distributed
  rank (default: -1; set to -1 to capture all ranks).
- VLLM_MOE_ROUTER_CAPTURE_MAX_LAYERS=N: max number of layers to record per
  process (default: 0; 0 = unlimited).
- VLLM_MOE_ROUTER_CAPTURE_NUM_TOKENS_GT=A: only record calls where router_logits
  has num_tokens > A (default: -1; <0 = disabled).
- VLLM_MOE_ROUTER_CAPTURE_NUM_TOKENS_LT=B: only record calls where router_logits
  has num_tokens < B (default: -1; 0 = disabled).

Output format:
- A single `.pt` per captured num_tokens (and per rank if torch.distributed is
  initialized).
- Payload includes `layers_by_num_tokens: dict[str, dict[layer_name, layer_state]]`.
- A convenience `layers` field is also included (same as
  `layers_by_num_tokens[str(num_tokens)]`) for easy loading.
- For each captured MoE layer, stores a list of 2D tensors
  `router_logits_chunks: list[Tensor[num_tokens_i, num_experts]]` on CPU,
  typically in fp16 for space efficiency.
"""

from __future__ import annotations

import atexit
import inspect
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch

import vllm.envs as envs

_DEFAULT_SKIP_STACK_FUNCS = ("profile_run", "_dummy_run",
                             "determine_available_memory")


@dataclass(frozen=True)
class RouterCaptureConfig:
    enabled: bool = False
    out_dir: str = "/tmp"
    skip_profile: bool = True
    skip_stack_funcs: tuple[str, ...] = _DEFAULT_SKIP_STACK_FUNCS
    only_rank: Optional[int] = 0
    max_layers: int = 0
    num_tokens_gt: Optional[int] = None
    num_tokens_lt: Optional[int] = None

    @staticmethod
    def from_env() -> "RouterCaptureConfig":
        enabled = envs.VLLM_MOE_ROUTER_CAPTURE
        out_dir = envs.VLLM_MOE_ROUTER_CAPTURE_DIR
        skip_profile = True
        skip_stack_funcs = _DEFAULT_SKIP_STACK_FUNCS
        only_rank: Optional[int] = None
        if envs.VLLM_MOE_ROUTER_CAPTURE_RANK >= 0:
            only_rank = envs.VLLM_MOE_ROUTER_CAPTURE_RANK
        max_layers = envs.VLLM_MOE_ROUTER_CAPTURE_MAX_LAYERS
        num_tokens_gt_opt = (envs.VLLM_MOE_ROUTER_CAPTURE_NUM_TOKENS_GT
                             if envs.VLLM_MOE_ROUTER_CAPTURE_NUM_TOKENS_GT >= 0
                             else None)
        num_tokens_lt_opt = (envs.VLLM_MOE_ROUTER_CAPTURE_NUM_TOKENS_LT
                             if envs.VLLM_MOE_ROUTER_CAPTURE_NUM_TOKENS_LT > 0
                             else None)

        # Per-size mode requires an explicit token-count filter to avoid
        # unbounded captures by default.
        if num_tokens_gt_opt is None and num_tokens_lt_opt is None:
            enabled = False
        if (num_tokens_gt_opt is not None and num_tokens_lt_opt is not None
                and num_tokens_gt_opt >= num_tokens_lt_opt):
            enabled = False
        return RouterCaptureConfig(enabled=enabled,
                                   out_dir=out_dir,
                                   skip_profile=skip_profile,
                                   skip_stack_funcs=skip_stack_funcs,
                                   only_rank=only_rank,
                                   max_layers=max_layers,
                                   num_tokens_gt=num_tokens_gt_opt,
                                   num_tokens_lt=num_tokens_lt_opt)


def _in_profile_run(skip_stack_funcs: tuple[str, ...]) -> bool:
    """
    Best-effort detection for vLLM startup profiling/warmup runs.

    Startup warmups often execute MoE kernels with synthetic shapes. When
    enabled, skip captures from these stacks so the first capture comes from a
    real request.
    """
    if not skip_stack_funcs:
        return False
    frame = inspect.currentframe()
    try:
        while frame is not None:
            name = frame.f_code.co_name
            if name in skip_stack_funcs:
                return True
            frame = frame.f_back
    finally:
        # Avoid reference cycles.
        del frame
    return False


class _RouterCapture:

    def __init__(self, cfg: RouterCaptureConfig) -> None:
        self.cfg = cfg
        # Bucket captures by token count.
        self._layers_by_num_tokens: dict[int, dict[str, dict[str, object]]] = {}
        self._layer_names: set[str] = set()
        self._completed_num_tokens: set[int] = set()
        self._lock = threading.Lock()
        self._flush_counter = 0

        self._pid = os.getpid()
        self._host = socket.gethostname()
        self._start_time = time.time()

        os.makedirs(cfg.out_dir, exist_ok=True)
        atexit.register(self.flush)

    def _bucket_for_num_tokens(self, num_tokens: int) -> Optional[int]:
        """Return the per-size bucket key for this record call, or None if filtered."""
        if self.cfg.num_tokens_gt is None and self.cfg.num_tokens_lt is None:
            return None
        if self.cfg.num_tokens_gt is not None:
            if int(num_tokens) <= int(self.cfg.num_tokens_gt):
                return None
        if self.cfg.num_tokens_lt is not None:
            if int(num_tokens) >= int(self.cfg.num_tokens_lt):
                return None
        bucket_num_tokens = int(num_tokens)
        if bucket_num_tokens != 0 and bucket_num_tokens in self._completed_num_tokens:
            return None
        return bucket_num_tokens

    def _snapshot_layers_by_num_tokens(
        self,
        layers_by_num_tokens: dict[int, dict[str, dict[str, object]]],
    ) -> dict[int, dict[str, dict[str, object]]]:
        snapshot: dict[int, dict[str, dict[str, object]]] = {}
        for num_tokens, bucket in layers_by_num_tokens.items():
            bucket_snapshot: dict[str, dict[str, object]] = {}
            for layer_name, state in bucket.items():
                chunks = state.get("router_logits_chunks", [])
                bucket_snapshot[layer_name] = {
                    "num_experts": int(state.get("num_experts", 0)),
                    "num_tokens": int(state.get("num_tokens", 0)),
                    "router_logits_chunks": list(chunks),
                }
            snapshot[int(num_tokens)] = bucket_snapshot
        return snapshot

    @torch.no_grad()
    def record(self, layer_name: str, router_logits: torch.Tensor,
               top_k: int) -> None:
        if self.cfg.skip_profile and _in_profile_run(self.cfg.skip_stack_funcs):
            return
        if self.cfg.only_rank is not None:
            rank = _get_rank()
            if rank is not None and rank != self.cfg.only_rank:
                return
        if router_logits.dim() != 2:
            return
        num_tokens, num_experts = router_logits.shape
        if num_tokens == 0 or num_experts == 0:
            return
        bucket_num_tokens = self._bucket_for_num_tokens(int(num_tokens))
        if bucket_num_tokens is None:
            return

        # Limit the number of recorded layers to avoid unbounded dumps.
        if layer_name not in self._layer_names:
            if self.cfg.max_layers != 0 and len(self._layer_names) >= self.cfg.max_layers:
                return
            self._layer_names.add(layer_name)

        # Store on CPU to avoid consuming GPU memory during long runs.
        # fp16 is typically sufficient because we primarily care about
        # distribution and relative ordering (top-k), not exact values.
        router_logits_cpu = router_logits.detach()
        if router_logits_cpu.is_cuda:
            router_logits_cpu = router_logits_cpu.to(device="cpu",
                                                     dtype=torch.float16)
        else:
            router_logits_cpu = router_logits_cpu.to(dtype=torch.float16)

        bucket_snapshot: Optional[dict[str, dict[str, object]]] = None
        should_flush = False
        with self._lock:
            bucket = self._layers_by_num_tokens.setdefault(bucket_num_tokens, {})
            if layer_name in bucket:
                return
            bucket[layer_name] = {
                "num_experts": int(num_experts),
                "num_tokens": int(num_tokens),
                "router_logits_chunks": [router_logits_cpu],
            }
            if self.cfg.max_layers != 0 and len(bucket) >= int(self.cfg.max_layers):
                should_flush = True
                bucket_snapshot = self._snapshot_layers_by_num_tokens(
                    {int(bucket_num_tokens): bucket})[int(bucket_num_tokens)]
                self._completed_num_tokens.add(int(bucket_num_tokens))
                self._layers_by_num_tokens.pop(int(bucket_num_tokens), None)

        if should_flush and bucket_snapshot is not None:
            self._flush_payload(
                layers_by_num_tokens={int(bucket_num_tokens): bucket_snapshot},
                file_tag=f"nt{int(bucket_num_tokens)}",
            )

    def _flush_payload(
        self,
        *,
        layers_by_num_tokens: dict[int, dict[str, dict[str, object]]],
        file_tag: Optional[str] = None,
    ) -> Optional[str]:
        if not self.cfg.enabled:
            return None
        if self.cfg.only_rank is not None:
            rank = _get_rank()
            if rank is not None and rank != self.cfg.only_rank:
                return None

        rank = _get_rank()
        now = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        ts_us = int(now * 1_000_000)
        with self._lock:
            flush_idx = self._flush_counter
            self._flush_counter += 1
        rank_str = f"rank{rank}" if rank is not None else "rankNA"
        tag = f"{file_tag}_" if file_tag else ""
        out_path = os.path.join(
            self.cfg.out_dir,
            f"moe_router_stats_{tag}{ts_us}_{self._host}_{rank_str}_pid{self._pid}_flush{flush_idx}.pt",
        )

        layers_by_num_tokens_out: dict[str, object] = {}
        for num_tokens, bucket in layers_by_num_tokens.items():
            bucket_out: dict[str, object] = {}
            for layer_name, state in bucket.items():
                bucket_out[layer_name] = {
                    "num_experts": int(state["num_experts"]),
                    "num_tokens": int(state["num_tokens"]),
                    "router_logits_chunks":
                    state["router_logits_chunks"],  # type: ignore[typeddict-item]
                }
            layers_by_num_tokens_out[str(int(num_tokens))] = bucket_out

        payload: dict[str, object] = {
            "meta": {
                "timestamp": ts,
                "timestamp_us": ts_us,
                "flush_index": int(flush_idx),
                "host": self._host,
                "pid": self._pid,
                "rank": rank,
                "wall_time_s": float(now - self._start_time),
            },
            "layers_by_num_tokens": layers_by_num_tokens_out,
        }

        # Backward-compatible convenience field when there is a single bucket.
        if len(layers_by_num_tokens) == 1:
            (only_bucket_key, ) = layers_by_num_tokens.keys()
            payload["layers"] = layers_by_num_tokens_out[str(int(only_bucket_key))]

        try:
            torch.save(payload, out_path)
        except Exception:
            return None
        return out_path

    def flush(self) -> Optional[str]:
        with self._lock:
            if not self._layers_by_num_tokens:
                return None
            snapshot = self._snapshot_layers_by_num_tokens(self._layers_by_num_tokens)

        return self._flush_payload(layers_by_num_tokens=snapshot)

    def reset(self) -> None:
        with self._lock:
            self._layers_by_num_tokens.clear()
            self._layer_names.clear()
            self._completed_num_tokens.clear()
            self._start_time = time.time()


_CAPTURE: Optional[_RouterCapture] = None
_CAPTURE_DISABLED: bool = False


def _disable_global_capture() -> None:
    global _CAPTURE, _CAPTURE_DISABLED
    _CAPTURE = None
    _CAPTURE_DISABLED = True


def _get_rank() -> Optional[int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_rank()
        except Exception:
            return None
    return None


def _get_capture() -> Optional[_RouterCapture]:
    global _CAPTURE, _CAPTURE_DISABLED
    if _CAPTURE_DISABLED:
        return None
    if _CAPTURE is not None:
        return _CAPTURE
    cfg = RouterCaptureConfig.from_env()
    if not cfg.enabled:
        _disable_global_capture()
        return None
    if cfg.only_rank is not None:
        rank = _get_rank()
        if rank is not None and rank != cfg.only_rank:
            _disable_global_capture()
            return None
    _CAPTURE = _RouterCapture(cfg)
    return _CAPTURE


@torch.no_grad()
def maybe_record_router_logits(*, layer_name: str, router_logits: torch.Tensor,
                              top_k: int) -> None:
    capture = _get_capture()
    if capture is None:
        return
    capture.record(layer_name=layer_name, router_logits=router_logits, top_k=top_k)


def maybe_flush_router_capture(*, reset: bool = False) -> Optional[str]:
    """Flush capture buffers to disk without exiting the process."""
    capture = _get_capture()
    if capture is None:
        return None
    out_path = capture.flush()
    if out_path is not None and reset:
        capture.reset()
    return out_path
