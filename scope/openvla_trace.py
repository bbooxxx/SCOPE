"""Deterministic OpenVLA Attention/FFN behavior-level cache-line traces.

The public OpenVLA artifacts expose model/operator shapes, not a hardware ISA
load/store trace.  This module therefore emits real tensor-tile reads/writes at
cache-line granularity.  It is deliberately a behavior model, not a claim about
the exact instruction order of a particular GPU kernel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TraceEvent = Tuple[str, int]


@dataclass(frozen=True)
class OpenVLATrace:
    workload: str
    events: Tuple[TraceEvent, ...]
    reads: int
    writes: int
    read_fraction: float
    memory_access_rate_per_s: float
    metadata: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("events")
        result["event_count"] = len(self.events)
        return result


class _Builder:
    def __init__(self, line_bytes: int) -> None:
        self.line_bytes = line_bytes
        self.events: List[TraceEvent] = []

    def span(self, op: str, base: int, offset: int, size: int) -> None:
        if size <= 0:
            return
        first = (base + offset) // self.line_bytes
        last = (base + offset + size - 1) // self.line_bytes
        self.events.extend((op, line) for line in range(first, last + 1))


def _attention(shape: Dict[str, int], line_bytes: int) -> Tuple[TraceEvent, ...]:
    seq = int(shape["sequence_tokens"])
    hidden = int(shape["hidden_size"])
    heads = int(shape["num_attention_heads"])
    head_dim = int(shape["head_dim"])
    tile = int(shape["tile_tokens"])
    output_tile = int(shape.get("projection_output_tile", 128))
    element = int(shape["bytes_per_element"])
    sampled_token_tiles = int(shape.get("sampled_token_tiles", 1))
    sampled_projection_tiles = int(shape.get("sampled_projection_tiles", 2))

    # Address regions mirror the real tensor shapes and stay non-overlapping.
    x = 0x1000_0000
    wqkv = 0x2000_0000
    qkv = 0x4000_0000
    score = 0x5000_0000
    context = 0x6000_0000
    wo = 0x7000_0000
    y = 0x9000_0000
    b = _Builder(line_bytes)

    for token_block in range(min(sampled_token_tiles, (seq + tile - 1) // tile)):
        tokens = min(tile, seq - token_block * tile)
        x_offset = token_block * tile * hidden * element
        for output_block in range(sampled_projection_tiles):
            channel = output_block * output_tile
            b.span("load", x, x_offset, tokens * hidden * element)
            b.span("load", wqkv, channel * hidden * element,
                   hidden * output_tile * element)
            b.span("store", qkv,
                   (token_block * tile * 3 * hidden + channel) * element,
                   tokens * output_tile * element)

        for head in range(heads):
            q_offset = ((token_block * tile * 3 * hidden) + head * head_dim) * element
            k_offset = (hidden + head * head_dim) * element
            v_offset = (2 * hidden + head * head_dim) * element
            b.span("load", qkv, q_offset, tokens * head_dim * element)
            # K/V traverse the actual sequence with the real per-token stride.
            for token in range(seq):
                base = token * 3 * hidden * element
                b.span("load", qkv, base + k_offset, head_dim * element)
            score_offset = (head * seq * seq + token_block * tile * seq) * element
            b.span("store", score, score_offset, tokens * seq * element)
            b.span("load", score, score_offset, tokens * seq * element)
            for token in range(seq):
                base = token * 3 * hidden * element
                b.span("load", qkv, base + v_offset, head_dim * element)
            ctx_offset = (token_block * tile * hidden + head * head_dim) * element
            b.span("store", context, ctx_offset, tokens * head_dim * element)

        for output_block in range(sampled_projection_tiles):
            channel = output_block * output_tile
            ctx_offset = token_block * tile * hidden * element
            b.span("load", context, ctx_offset, tokens * hidden * element)
            b.span("load", wo, channel * hidden * element,
                   hidden * output_tile * element)
            b.span("store", y,
                   (token_block * tile * hidden + channel) * element,
                   tokens * output_tile * element)
    return tuple(b.events)


def _ffn(shape: Dict[str, int], line_bytes: int) -> Tuple[TraceEvent, ...]:
    seq = int(shape["sequence_tokens"])
    hidden = int(shape["hidden_size"])
    intermediate = int(shape["intermediate_size"])
    tile = int(shape["tile_tokens"])
    channel_tile = int(shape.get("channel_tile", 256))
    element = int(shape["bytes_per_element"])
    sampled_token_tiles = int(shape.get("sampled_token_tiles", 1))
    sampled_channel_tiles = int(shape.get("sampled_channel_tiles", 2))

    x = 0x1000_0000
    wg = 0x2000_0000
    wu = 0x4000_0000
    gate = 0x6000_0000
    up = 0x7000_0000
    activation = 0x8000_0000
    wd = 0x9000_0000
    y = 0xC000_0000
    b = _Builder(line_bytes)

    for token_block in range(min(sampled_token_tiles, (seq + tile - 1) // tile)):
        tokens = min(tile, seq - token_block * tile)
        x_offset = token_block * tile * hidden * element
        for channel_block in range(sampled_channel_tiles):
            channel = channel_block * channel_tile
            block_offset = channel * hidden * element
            activation_offset = (token_block * tile * intermediate + channel) * element
            b.span("load", x, x_offset, tokens * hidden * element)
            b.span("load", wg, block_offset, hidden * channel_tile * element)
            b.span("store", gate, activation_offset, tokens * channel_tile * element)
            b.span("load", x, x_offset, tokens * hidden * element)
            b.span("load", wu, block_offset, hidden * channel_tile * element)
            b.span("store", up, activation_offset, tokens * channel_tile * element)
            b.span("load", gate, activation_offset, tokens * channel_tile * element)
            b.span("load", up, activation_offset, tokens * channel_tile * element)
            b.span("store", activation, activation_offset,
                   tokens * channel_tile * element)
            b.span("load", activation, activation_offset,
                   tokens * channel_tile * element)
            b.span("load", wd, channel * hidden * element,
                   channel_tile * hidden * element)
            b.span("store", y, x_offset, tokens * hidden * element)
    return tuple(b.events)


def build_trace(model: Dict[str, Any], line_bytes: int) -> OpenVLATrace:
    workload = str(model["operator"]).lower()
    shape = {key: int(value) for key, value in model["operator_shape"].items()}
    if workload == "attention":
        events = _attention(shape, line_bytes)
    elif workload == "ffn":
        events = _ffn(shape, line_bytes)
    else:
        raise ValueError("OpenVLA operator must be attention or ffn")
    if not events:
        raise ValueError("OpenVLA trace generator produced no accesses")
    reads = sum(op == "load" for op, _ in events)
    writes = len(events) - reads
    policy_hz = float(model.get("policy_frequency_hz", 5.0))
    metadata = {
        "model": "OpenVLA-7B / Llama-2-7B",
        "source_model_config":
            "https://github.com/openvla/openvla/blob/main/prismatic/conf/models.py",
        "trace_basis": "shape-faithful tiled tensor access trace",
        "hardware_trace_available": False,
        "limitation": (
            "Public OpenVLA/VLA-Trace artifacts do not contain an ISA-level "
            "load/store trace; addresses model tensor tiles, not a specific GPU kernel."
        ),
        "operator_shape": shape,
        "policy_frequency_hz": policy_hz,
    }
    return OpenVLATrace(
        workload=workload,
        events=events,
        reads=reads,
        writes=writes,
        read_fraction=reads / len(events),
        memory_access_rate_per_s=len(events) * policy_hz,
        metadata=metadata,
    )


def repeated(events: Sequence[TraceEvent], repetitions: int) -> Iterable[TraceEvent]:
    for _ in range(repetitions):
        yield from events
