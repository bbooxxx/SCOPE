#!/usr/bin/env python3
"""SCOPE: a three-level on-chip cache hierarchy built from DESTINY instances.

The original DESTINY binary remains the source of per-level array, peripheral,
bank, latency, energy, leakage, and refresh metrics.  This module composes three
independent DESTINY runs into a configurable hierarchy and evaluates both the
analytical FoM supplied by the SCOPE specification and concrete load/store
paths under write-back + write-allocate.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import itertools
import json
import math
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .bti import evaluate_bti_retention
from .edram import RefreshResult, evaluate_read, evaluate_row_refresh, no_refresh
from .m3d import evaluate_m3d
from .nonideal import evaluate_nonideal
from .openvla_trace import build_trace, repeated
from .sense_amp import evaluate_sense_amp


FLOAT_RE = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
VALID_POLICIES = {"LRU", "FIFO", "RANDOM"}
FEATURE_SWITCH_DEFAULTS = {
    "array_nonideal": True,
    "configurable_peripherals": True,
    "m3d": True,
}
GUIDANCE_METRICS = {
    "latency": "average_latency_ns",
    "latency_ns": "average_latency_ns",
    "average_latency_ns": "average_latency_ns",
    "power": "average_power_mw",
    "power_mw": "average_power_mw",
    "average_power_mw": "average_power_mw",
    "energy": "expected_dynamic_energy_nj_per_request",
    "energy_nj": "expected_dynamic_energy_nj_per_request",
    "expected_dynamic_energy_nj_per_request":
        "expected_dynamic_energy_nj_per_request",
}


class ScopeError(RuntimeError):
    """Raised for invalid configurations or failed DESTINY runs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScopeError(message)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def resolve_feature_switches(raw: Dict[str, Any]) -> Dict[str, bool]:
    """Return validated evaluation switches, preserving pre-v7 behavior."""
    configured = raw.get("features", {})
    require(isinstance(configured, dict), "features must be an object")
    unknown = set(configured) - set(FEATURE_SWITCH_DEFAULTS)
    require(not unknown, f"unknown feature switches: {sorted(unknown)}")
    resolved = dict(FEATURE_SWITCH_DEFAULTS)
    for name, value in configured.items():
        require(isinstance(value, bool), f"features.{name} must be boolean")
        resolved[name] = value
    return resolved


@dataclass(frozen=True)
class DestinyMetrics:
    capacity_bytes: int
    associativity: int
    line_bytes: int
    hit_latency_ns: float
    miss_latency_ns: float
    write_latency_ns: float
    hit_energy_nj: float
    miss_energy_nj: float
    write_energy_nj: float
    leakage_power_mw: float
    data_array_leakage_power_mw: float
    tag_array_leakage_power_mw: float
    refresh_latency_us: float = 0.0
    refresh_energy_nj_per_bank: float = 0.0
    optimization_target: str = ""
    bank_organization: str = ""
    mat_organization: str = ""
    subarray_size: str = ""
    local_wire: str = ""
    global_wire: str = ""
    buffer_design: str = ""
    subarray_rows: int = 0
    subarray_columns: int = 0
    rbl_capacitance_ff: float = 0.0
    rbl_wire_capacitance_ff: float = 0.0
    rbl_cell_capacitance_ff: float = 0.0
    rbl_length_um: float = 0.0
    rwl_length_um: float = 0.0
    rbl_resistance_ohm: float = 0.0
    rwl_resistance_ohm: float = 0.0
    rbl_delay_ns: float = 0.0
    rbl_read_energy_nj: float = 0.0
    peripheral_read_latency_ns: float = 0.0
    peripheral_read_energy_nj: float = 0.0
    data_array_area_mm2: float = 0.0
    native_sense_amp_type: str = ""
    sense_amp_count: int = 0
    sense_amp_latency_ns: float = 0.0
    sense_amp_read_energy_nj: float = 0.0
    sense_amp_leakage_mw: float = 0.0
    legacy_iv_converter_latency_ns: float = 0.0
    legacy_iv_converter_read_energy_nj: float = 0.0
    legacy_iv_converter_leakage_mw: float = 0.0
    tag_lookup_latency_ns: float = 0.0
    tag_lookup_energy_nj: float = 0.0
    data_bank_latency_ns: float = 0.0
    data_routing_latency_ns: float = 0.0
    data_predecoder_latency_ns: float = 0.0
    data_row_decoder_latency_ns: float = 0.0
    data_mux_latency_ns: float = 0.0
    data_precharge_latency_ns: float = 0.0
    tag_comparator_latency_ns: float = 0.0


@dataclass(frozen=True)
class LayerSpec:
    name: str
    device: str
    device_family: str
    destiny_config: Path
    capacity_bytes: int
    destiny_model_capacity_bytes: int
    associativity: int
    line_bytes: int
    replacement_policy: str
    banks: int
    peripheral_latency_ns: float
    peripheral_energy_nj: float
    ber: float
    ber_max: float
    allow_high_variation: bool
    endurance_writes_per_line: float
    wear_leveling_efficiency: float
    refresh_interval_us: float
    retention_time_us: float
    estimated_writebacks_per_request: float
    device_rows_per_bank: int
    stacked_tiers: int
    effective_density_f2: float
    data_cell_area_f2: float
    device_leakage_power_mw: float
    device_refresh_power_mw: float
    device_library_entry: Dict[str, Any]
    raw_metrics: DestinyMetrics
    metrics: DestinyMetrics
    edram_read: Optional[Dict[str, Any]] = None
    refresh: Optional[Dict[str, Any]] = None
    bti: Optional[Dict[str, Any]] = None
    sense_amp: Optional[Dict[str, Any]] = None
    m3d: Optional[Dict[str, Any]] = None
    nonideal: Optional[Dict[str, Any]] = None
    static_power_components: Optional[Dict[str, float]] = None

    @property
    def lines(self) -> int:
        return self.capacity_bytes // self.line_bytes


@dataclass(frozen=True)
class CrossbarSpec:
    name: str
    request_cycles: float
    response_cycles: float
    clock_ghz: float
    energy_pj_per_bit: float
    transaction_bits: int
    topology: str = "legacy crossbar"
    hops: int = 1
    router_pipeline_cycles: float = 0.0
    link_traversal_cycles: float = 0.0
    request_bits: int = 0
    response_bits: int = 0
    router_energy_pj_per_bit: float = 0.0
    link_energy_pj_per_bit_per_mm: float = 0.0
    link_length_mm: float = 0.0

    @property
    def latency_ns(self) -> float:
        return (self.request_cycles + self.response_cycles) / self.clock_ghz

    @property
    def energy_nj(self) -> float:
        if self.request_bits > 0 and self.response_bits > 0:
            per_bit = (
                self.router_energy_pj_per_bit
                + self.link_energy_pj_per_bit_per_mm * self.link_length_mm
            )
            return self.hops * (self.request_bits + self.response_bits) * per_bit / 1000.0
        return self.energy_pj_per_bit * self.transaction_bits / 1000.0

    @property
    def request_latency_ns(self) -> float:
        return self.request_cycles / self.clock_ghz

    @property
    def response_latency_ns(self) -> float:
        return self.response_cycles / self.clock_ghz


@dataclass(frozen=True)
class OffChipSpec:
    latency_ns: float
    energy_nj: float
    standard: str = "unspecified"
    bandwidth_gbps: float = 0.0
    bus_width_bits: int = 0
    data_rate_mtps: float = 0.0
    transaction_bytes: int = 64
    energy_pj_per_bit: float = 0.0
    timing_basis: str = ""


@dataclass(frozen=True)
class HitRateResult:
    hit_rates: Tuple[float, ...]
    accesses: Tuple[int, ...]
    hits: Tuple[int, ...]
    writebacks_per_request: Tuple[float, ...]
    offchip_writebacks_per_request: float
    observed_read_fraction: float
    trace_metadata: Optional[Dict[str, Any]] = None


@dataclass
class CacheEntry:
    line: int
    dirty: bool = False


class SetAssociativeCache:
    """Small, lazy set-associative tag model used only for hit-rate estimation."""

    def __init__(
        self,
        capacity_bytes: int,
        line_bytes: int,
        associativity: int,
        policy: str,
        rng: random.Random,
    ) -> None:
        require(capacity_bytes > 0, "cache capacity must be positive")
        require(line_bytes > 0, "cache line size must be positive")
        require(associativity > 0, "cache associativity must be positive")
        require(capacity_bytes % (line_bytes * associativity) == 0,
                "capacity must be divisible by line_bytes * associativity")
        self.num_sets = capacity_bytes // (line_bytes * associativity)
        require(self.num_sets > 0, "cache must contain at least one set")
        self.associativity = associativity
        self.policy = policy.upper()
        require(self.policy in VALID_POLICIES,
                f"unsupported replacement policy: {policy}")
        self.rng = rng
        self.sets: Dict[int, List[CacheEntry]] = {}

    def _entries(self, line: int) -> List[CacheEntry]:
        return self.sets.setdefault(line % self.num_sets, [])

    def probe(self, line: int, mark_dirty: bool = False) -> bool:
        entries = self._entries(line)
        for index, entry in enumerate(entries):
            if entry.line != line:
                continue
            if mark_dirty:
                entry.dirty = True
            if self.policy == "LRU":
                entries.append(entries.pop(index))
            return True
        return False

    def insert(self, line: int, dirty: bool = False) -> Optional[CacheEntry]:
        entries = self._entries(line)
        for index, entry in enumerate(entries):
            if entry.line != line:
                continue
            entry.dirty = entry.dirty or dirty
            if self.policy == "LRU":
                entries.append(entries.pop(index))
            return None

        evicted: Optional[CacheEntry] = None
        if len(entries) >= self.associativity:
            victim = self.rng.randrange(len(entries)) if self.policy == "RANDOM" else 0
            evicted = entries.pop(victim)
        entries.append(CacheEntry(line=line, dirty=dirty))
        return evicted


_CPP_HIT_RATE_CACHE: Dict[Tuple[Any, ...], HitRateResult] = {}


def _cpp_openvla_hit_rates(
    layers: Sequence[LayerSpec], model: Dict[str, Any], repo_root: Path,
) -> HitRateResult:
    binary = _resolve_path(
        repo_root, str(model.get("binary", "scope_model"))
    ).resolve()
    if not binary.is_file():
        completed = subprocess.run(
            ["make", "-j4", binary.name],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0 or not binary.is_file():
            raise ScopeError(f"failed to build C++ SCOPE model:\n{completed.stdout}")

    shape = dict(model.get("operator_shape", {}))
    require(shape, "cpp_openvla_trace requires operator_shape")
    capacities = tuple(layer.capacity_bytes for layer in layers)
    associativities = tuple(layer.associativity for layer in layers)
    policies = tuple(layer.replacement_policy for layer in layers)
    line_bytes = layers[0].line_bytes
    operator = str(model["operator"]).lower()
    sampled_working_set_bytes = int(model["sampled_working_set_bytes"])
    warmup_accesses = int(model.get("warmup_accesses", 0))
    sample_accesses = int(model.get("sample_accesses", 0))
    seed = int(model.get("seed", 7))
    access_bytes = int(model.get("isa_access_bytes", 16))
    bytes_per_element = int(model.get("bytes_per_element", 2))
    stride_bytes = int(model.get("working_set_stride_bytes", line_bytes))
    cycle_access_cap = int(model.get("trace_cycle_accesses", 0))
    key = (
        operator, capacities, associativities, policies, line_bytes,
        sampled_working_set_bytes, warmup_accesses, sample_accesses,
        seed, access_bytes, bytes_per_element, stride_bytes, cycle_access_cap,
        tuple(sorted((str(key), int(value)) for key, value in shape.items())),
    )
    cached = _CPP_HIT_RATE_CACHE.get(key)
    if cached is not None:
        return cached

    arguments = [
        str(binary),
        "--operator", operator,
        "--capacities", ",".join(str(value) for value in capacities),
        "--associativities", ",".join(str(value) for value in associativities),
        "--policies", ",".join(policies),
        "--line-bytes", str(line_bytes),
        "--sampled-working-set-bytes", str(sampled_working_set_bytes),
        "--access-bytes", str(access_bytes),
        "--bytes-per-element", str(bytes_per_element),
        "--working-set-stride-bytes", str(stride_bytes),
        "--cycle-access-cap", str(cycle_access_cap),
        "--seed", str(seed),
        "--sequence-tokens", str(shape["sequence_tokens"]),
        "--hidden-size", str(shape["hidden_size"]),
        "--attention-heads", str(shape.get("num_attention_heads", 32)),
        "--head-dimension", str(shape.get("head_dim", 128)),
        "--intermediate-size", str(shape.get("intermediate_size", 11008)),
        "--tile-m", str(shape.get("tile_m", shape.get("tile_tokens", 16))),
        "--tile-n", str(shape.get("tile_n", shape.get("channel_tile", 64))),
        "--tile-k", str(shape.get("tile_k", 32)),
    ]
    if warmup_accesses > 0:
        arguments.extend(["--warmup-accesses", str(warmup_accesses)])
    if sample_accesses > 0:
        arguments.extend(["--sample-accesses", str(sample_accesses)])
    completed = subprocess.run(
        arguments,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise ScopeError(f"C++ SCOPE model failed:\n{completed.stdout}")
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ScopeError(
            f"invalid JSON from C++ SCOPE model:\n{completed.stdout}"
        ) from exc
    rates = tuple(float(value) for value in raw["hit_rates"])
    accesses = tuple(int(value) for value in raw["accesses"])
    hits = tuple(int(value) for value in raw["hits"])
    writebacks = tuple(float(value) for value in raw["writebacks_per_request"])
    require(len(rates) == len(layers),
            "C++ SCOPE result does not match hierarchy depth")
    policy_hz = float(model.get("policy_frequency_hz", 5.0))
    raw.update({
        "model": "OpenVLA-7B / Llama-2-7B",
        "source_model_config":
            "https://github.com/openvla/openvla/blob/main/prismatic/conf/models.py",
        "cache_method_source": "ChampSim/gem5-style warmup + set-associative LRU",
        "trace_basis": (
            "seeded ISA-granularity 16B vector load/store sample emitted in "
            "128B cache-line bursts from one real-shape OpenVLA Attention or "
            "FFN layer"
        ),
        "hardware_trace_available": False,
        "selection": (
            "deterministic tensor-tile scheduling with cache-line spatial "
            "locality and bounded tile reuse"
        ),
        "memory_access_rate_per_s": int(raw["trace_cycle_accesses"]) * policy_hz,
        "policy_frequency_hz": policy_hz,
    })
    result = HitRateResult(
        hit_rates=rates,
        accesses=accesses,
        hits=hits,
        writebacks_per_request=writebacks,
        offchip_writebacks_per_request=float(
            raw["offchip_writebacks_per_request"]
        ),
        observed_read_fraction=float(raw["observed_read_fraction"]),
        trace_metadata=raw,
    )
    _CPP_HIT_RATE_CACHE[key] = result
    return result


class DestinyRunner:
    """Run one untouched DESTINY process per cache level and parse its summary."""

    def __init__(self, repo_root: Path, binary: Path, timeout_s: float = 300.0) -> None:
        self.repo_root = repo_root.resolve()
        self.binary = binary.resolve()
        self.timeout_s = timeout_s
        self._cache: Dict[Path, DestinyMetrics] = {}

    def ensure_built(self, auto_build: bool = True) -> None:
        if self.binary.is_file():
            return
        require(auto_build, f"DESTINY binary not found: {self.binary}")
        completed = subprocess.run(
            ["make", "-j4"],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0 or not self.binary.is_file():
            raise ScopeError(f"failed to build DESTINY:\n{completed.stdout}")

    def run(self, config_path: Path) -> DestinyMetrics:
        config_path = config_path.resolve()
        if config_path in self._cache:
            return self._cache[config_path]
        require(config_path.is_file(), f"DESTINY config not found: {config_path}")
        cache_key = hashlib.sha256(
            config_path.read_bytes()
            + str(self.binary.stat().st_mtime_ns).encode("ascii")
            + str(self.binary.stat().st_size).encode("ascii")
        ).hexdigest()
        metrics_dir = self.repo_root / "config" / ".scope-cache" / "metrics"
        metrics_path = metrics_dir / f"{cache_key}.json"
        if metrics_path.is_file():
            try:
                metrics = DestinyMetrics(**json.loads(
                    metrics_path.read_text(encoding="utf-8")
                ))
                self._cache[config_path] = metrics
                return metrics
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        try:
            completed = subprocess.run(
                [str(self.binary), config_path.name],
                cwd=config_path.parent,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ScopeError(
                f"DESTINY timed out after {self.timeout_s:g}s for {config_path.name}"
            ) from exc
        if completed.returncode != 0:
            raise ScopeError(
                f"DESTINY failed for {config_path.name} (exit {completed.returncode}):\n"
                f"{completed.stdout}"
            )
        metrics = self.parse_output(completed.stdout)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._cache[config_path] = metrics
        return metrics

    @staticmethod
    def _number(text: str, label: str, optional: bool = False) -> float:
        match = re.search(re.escape(label) + r"\s*=\s*" + FLOAT_RE, text)
        if match:
            return float(match.group(1))
        if optional:
            return 0.0
        raise ScopeError(f"missing DESTINY output field: {label}")

    @staticmethod
    def parse_output(text: str) -> DestinyMetrics:
        capacity_match = re.search(r"^Capacity\s*:\s*" + FLOAT_RE + r"\s*(B|KB|MB)\s*$",
                                   text, flags=re.MULTILINE)
        line_match = re.search(r"^Cache Line Size:\s*(\d+)\s*Bytes\s*$",
                               text, flags=re.MULTILINE)
        assoc_match = re.search(r"^Cache Associativity:\s*(\d+)\s*Ways\s*$",
                                text, flags=re.MULTILINE)
        require(capacity_match is not None, "missing DESTINY capacity")
        require(line_match is not None, "missing DESTINY cache-line size")
        require(assoc_match is not None, "missing DESTINY associativity")
        scale = {"B": 1, "KB": 1024, "MB": 1024 * 1024}[capacity_match.group(2)]
        capacity_bytes = int(float(capacity_match.group(1)) * scale)
        def detail(pattern: str) -> str:
            match = re.search(pattern, text, flags=re.MULTILINE)
            return match.group(1).strip() if match else ""

        return DestinyMetrics(
            capacity_bytes=capacity_bytes,
            associativity=int(assoc_match.group(1)),
            line_bytes=int(line_match.group(1)),
            hit_latency_ns=DestinyRunner._number(text, "Cache Hit Latency"),
            miss_latency_ns=DestinyRunner._number(text, "Cache Miss Latency"),
            write_latency_ns=DestinyRunner._number(text, "Cache Write Latency"),
            hit_energy_nj=DestinyRunner._number(text, "Cache Hit Dynamic Energy"),
            miss_energy_nj=DestinyRunner._number(text, "Cache Miss Dynamic Energy"),
            write_energy_nj=DestinyRunner._number(text, "Cache Write Dynamic Energy"),
            leakage_power_mw=DestinyRunner._number(text, "Cache Total Leakage Power"),
            data_array_leakage_power_mw=DestinyRunner._number(
                text, "Cache Data Array Leakage Power"
            ),
            tag_array_leakage_power_mw=DestinyRunner._number(
                text, "Cache Tag Array Leakage Power"
            ),
            refresh_latency_us=DestinyRunner._number(
                text, "Cache Refresh Latency", optional=True
            ),
            refresh_energy_nj_per_bank=DestinyRunner._number(
                text, "Cache Refresh Dynamic Energy", optional=True
            ),
            optimization_target=detail(r"^\s*Optimized for:\s*(.+)$"),
            bank_organization=detail(r"^\s*Bank Organization:\s*(.+)$"),
            mat_organization=detail(r"^\s*Mat Organization:\s*(.+)$"),
            subarray_size=detail(r"^\s*-\s*Subarray Size\s*:\s*(.+)$"),
            local_wire=detail(r"^\s*Local Wire[^:]*:\s*(.+)$"),
            global_wire=detail(r"^\s*Global Wire[^:]*:\s*(.+)$"),
            buffer_design=detail(r"^\s*Buffer Design[^:]*:\s*(.+)$"),
            subarray_rows=int(DestinyRunner._number(
                text, "SCOPE Selected Data Subarray Rows", optional=True
            )),
            subarray_columns=int(DestinyRunner._number(
                text, "SCOPE Selected Data Subarray Columns", optional=True
            )),
            rbl_capacitance_ff=DestinyRunner._number(
                text, "SCOPE Selected RBL Capacitance", optional=True
            ),
            rbl_wire_capacitance_ff=DestinyRunner._number(
                text, "SCOPE Selected RBL Wire Capacitance", optional=True
            ),
            rbl_cell_capacitance_ff=DestinyRunner._number(
                text, "SCOPE Selected RBL Cell Capacitance", optional=True
            ),
            rbl_length_um=DestinyRunner._number(
                text, "SCOPE Selected RBL Length", optional=True
            ),
            rwl_length_um=DestinyRunner._number(
                text, "SCOPE Selected RWL Length", optional=True
            ),
            rbl_resistance_ohm=DestinyRunner._number(
                text, "SCOPE Selected RBL Resistance", optional=True
            ),
            rwl_resistance_ohm=DestinyRunner._number(
                text, "SCOPE Selected RWL Resistance", optional=True
            ),
            rbl_delay_ns=DestinyRunner._number(
                text, "SCOPE Selected RBL Delay", optional=True
            ),
            rbl_read_energy_nj=DestinyRunner._number(
                text, "SCOPE Selected RBL Read Energy", optional=True
            ),
            peripheral_read_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Peripheral Read Latency", optional=True
            ),
            peripheral_read_energy_nj=DestinyRunner._number(
                text, "SCOPE Selected Peripheral Read Energy", optional=True
            ),
            data_array_area_mm2=DestinyRunner._number(
                text, "SCOPE Selected Data Array Area", optional=True
            ),
            native_sense_amp_type=detail(
                r"^SCOPE Selected Native Sense Amplifier Type\s*=\s*(.+)$"
            ),
            sense_amp_count=int(DestinyRunner._number(
                text, "SCOPE Selected Sense Amplifier Count", optional=True
            )),
            sense_amp_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Sense Amplifier Latency", optional=True
            ),
            sense_amp_read_energy_nj=DestinyRunner._number(
                text, "SCOPE Selected Sense Amplifier Read Energy", optional=True
            ),
            sense_amp_leakage_mw=DestinyRunner._number(
                text, "SCOPE Selected Sense Amplifier Leakage", optional=True
            ),
            legacy_iv_converter_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Legacy IV Converter Latency", optional=True
            ),
            legacy_iv_converter_read_energy_nj=DestinyRunner._number(
                text, "SCOPE Selected Legacy IV Converter Read Energy", optional=True
            ),
            legacy_iv_converter_leakage_mw=DestinyRunner._number(
                text, "SCOPE Selected Legacy IV Converter Leakage", optional=True
            ),
            tag_lookup_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Tag Lookup Latency", optional=True
            ),
            tag_lookup_energy_nj=DestinyRunner._number(
                text, "SCOPE Selected Tag Lookup Energy", optional=True
            ),
            data_bank_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Data Bank Latency", optional=True
            ),
            data_routing_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Data Routing Latency", optional=True
            ),
            data_predecoder_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Data Predecoder Latency", optional=True
            ),
            data_row_decoder_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Data Row Decoder Latency", optional=True
            ),
            data_mux_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Data Mux Latency", optional=True
            ),
            data_precharge_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Data Precharge Latency", optional=True
            ),
            tag_comparator_latency_ns=DestinyRunner._number(
                text, "SCOPE Selected Tag Comparator Latency", optional=True
            ),
        )


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _destiny_device_family(config_path: Path) -> str:
    """Read the cell family selected by a DESTINY configuration."""
    try:
        config_text = config_path.read_text(encoding="utf-8")
        cell_match = re.search(
            r"^-MemoryCellInputFile:\s*(\S+)\s*$", config_text, re.MULTILINE
        )
        require(cell_match is not None,
                f"{config_path.name}: missing -MemoryCellInputFile")
        cell_path = config_path.parent / cell_match.group(1)
        cell_text = cell_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScopeError(f"failed to inspect DESTINY device family: {exc}") from exc
    type_match = re.search(r"^-MemCellType:\s*(\S+)\s*$", cell_text, re.MULTILINE)
    require(type_match is not None, f"{cell_path.name}: missing -MemCellType")
    family = type_match.group(1)
    return "eDRAM" if family.lower() == "edram" else family


def _power_from_device_library(
    entry: Dict[str, Any], capacity_bits: int, refresh_interval_us: float
) -> Tuple[float, float]:
    """Return device leakage and refresh power in mW for the data bits."""
    refresh = entry["refresh"]
    leakage = entry["leakage"]

    def capacitor_power_mw(model: Dict[str, Any]) -> float:
        interval_us = refresh_interval_us or float(model["reference_retention_us"])
        require(interval_us > 0.0, "capacitor refresh interval must be positive")
        energy_j_per_bit = (
            0.5 * float(model["capacitance_f"]) * float(model["delta_v"]) ** 2
        )
        return energy_j_per_bit * capacity_bits / (interval_us * 1e-6) * 1e3

    refresh_model = str(refresh["model"])
    if refresh_model == "constant":
        refresh_mw = float(refresh["value_aw_per_bit"]) * capacity_bits * 1e-15
    elif refresh_model == "same_as_leakage_reference":
        refresh_mw = capacitor_power_mw(leakage)
    elif refresh_model in {"none", "row_read_write", "bti_row_read_write"}:
        # Row refresh needs the DESTINY-selected Nrow/Ncolumn and is evaluated
        # after the RBL equation.  BTI first derives its own retention time.
        refresh_mw = 0.0
    else:
        raise ScopeError(f"unsupported refresh model: {refresh_model}")

    leakage_model = str(leakage["model"])
    if leakage_model == "constant":
        leakage_mw = float(leakage["value_pw_per_bit"]) * capacity_bits * 1e-9
    elif leakage_model == "capacitor_refresh":
        leakage_mw = capacitor_power_mw(leakage)
    elif leakage_model == "same_as_refresh":
        leakage_mw = refresh_mw
    elif leakage_model == "none":
        leakage_mw = 0.0
    else:
        raise ScopeError(f"unsupported leakage model: {leakage_model}")
    # In the supplied eDRAM table, leakage is defined as Pref (or the same
    # capacitor-retention mechanism used to derive Pref).  Count it once under
    # refresh instead of duplicating it as independent static power.  SRAM is
    # the only library device with a separate static cell-leakage term.
    if str(entry.get("family")) != "SRAM":
        leakage_mw = 0.0
    return leakage_mw, refresh_mw


def _apply_device_library(
    raw_metrics: DestinyMetrics,
    entry: Dict[str, Any],
    capacity_bytes: int,
    line_bytes: int,
    banks: int,
    refresh_interval_us: float,
    retention_time_us: float,
) -> Tuple[
    DestinyMetrics, float, float, Optional[Dict[str, Any]], Dict[str, Any],
    Optional[Dict[str, Any]],
]:
    """Combine the device table with DESTINY-selected circuit geometry."""
    line_bits = line_bytes * 8

    def latency_floor(field: str, baseline: float) -> float:
        model = entry[field]
        if model["model"] == "constant":
            return max(baseline, float(model["value_ns"]))
        if str(model["model"]).startswith("destiny_") or model["model"] == "rbl_equation":
            return baseline
        raise ScopeError(f"unsupported {field} model: {model['model']}")

    def energy_floor(field: str, baseline: float) -> float:
        model = entry[field]
        if model["model"] == "constant":
            device_line_nj = float(model["value_fj_per_bit"]) * line_bits / 1e6
            return max(baseline, device_line_nj)
        if str(model["model"]).startswith("destiny_") or model["model"] == "rbl_equation":
            return baseline
        raise ScopeError(f"unsupported {field} model: {model['model']}")

    device_leakage_mw, device_refresh_mw = _power_from_device_library(
        entry, capacity_bytes * 8, refresh_interval_us
    )
    effective_values = asdict(raw_metrics)
    write_latency_ns = latency_floor("write_latency", raw_metrics.write_latency_ns)
    write_energy_nj = energy_floor("write_energy", raw_metrics.write_energy_nj)
    hit_latency_ns = latency_floor("read_latency", raw_metrics.hit_latency_ns)
    hit_energy_nj = energy_floor("read_energy", raw_metrics.hit_energy_nj)
    edram_read: Optional[Dict[str, Any]] = None
    refresh_result: RefreshResult = no_refresh()
    bti_result: Optional[Dict[str, Any]] = None

    read_model = str(entry["read_latency"]["model"])
    if str(entry.get("family")) == "eDRAM" and read_model == "rbl_equation":
        require(raw_metrics.subarray_rows > 0 and raw_metrics.subarray_columns > 0,
                "DESTINY did not report selected eDRAM subarray geometry")
        require(raw_metrics.rbl_capacitance_ff > 0.0,
                "DESTINY did not report selected eDRAM RBL capacitance")
        circuit = entry["read_circuit"]
        read_result = evaluate_read(
            nrow=raw_metrics.subarray_rows,
            ncolumn=raw_metrics.subarray_columns,
            rbl_capacitance_ff=raw_metrics.rbl_capacitance_ff,
            rbl_wire_capacitance_ff=raw_metrics.rbl_wire_capacitance_ff,
            rbl_cell_capacitance_ff=raw_metrics.rbl_cell_capacitance_ff,
            delta_v=float(circuit["delta_v"]),
            vdd=float(circuit["vdd"]),
            ion_ua_per_um=float(circuit["ion_ua_per_um"]),
            effective_width_um=float(circuit.get("effective_width_um", 1.0)),
            pure_device_path=str(circuit["path"]),
            line_bits=line_bits,
            peripheral_read_latency_ns=raw_metrics.peripheral_read_latency_ns,
            peripheral_read_energy_nj=raw_metrics.peripheral_read_energy_nj,
        )
        edram_read = read_result.to_dict()
        hit_latency_ns = read_result.total_read_latency_ns
        hit_energy_nj = read_result.total_read_energy_nj

        refresh_model = str(entry["refresh"]["model"])
        if refresh_model == "bti_row_read_write":
            nonideal = dict(entry.get("nonideal", {}))
            evaluated_bti = evaluate_bti_retention(
                dict(entry["bti"]),
                read_vgs_v=float(nonideal["read_vgs_v"]),
                initial_vth_v=float(nonideal["read_vth_v"]),
                current_alpha=float(nonideal["current_alpha"]),
            )
            bti_result = evaluated_bti.to_dict()
            guarded_cell_latency_ns = (
                read_result.cell_read_latency_ns
                * evaluated_bti.read_latency_guardband
            )
            edram_read["fresh_cell_read_latency_ns"] = \
                read_result.cell_read_latency_ns
            edram_read["cell_read_latency_ns"] = guarded_cell_latency_ns
            edram_read["bti_read_latency_guardband"] = \
                evaluated_bti.read_latency_guardband
            edram_read["total_read_latency_ns"] = (
                guarded_cell_latency_ns
                + read_result.peripheral_read_latency_ns
            )
            hit_latency_ns = float(edram_read["total_read_latency_ns"])
            refresh_interval_us = evaluated_bti.refresh_interval_s * 1e6
            retention_time_us = evaluated_bti.equivalent_retention_s * 1e6

        if refresh_model in {"row_read_write", "bti_row_read_write"}:
            refresh_result = evaluate_row_refresh(
                capacity_bytes=capacity_bytes,
                banks=banks,
                nrow=raw_metrics.subarray_rows,
                ncolumn=raw_metrics.subarray_columns,
                read_energy_fj_per_bit=read_result.read_energy_fj_per_bit,
                write_energy_fj_per_bit=float(
                    entry["write_energy"]["value_fj_per_bit"]
                ),
                read_latency_ns=hit_latency_ns,
                write_latency_ns=write_latency_ns,
                refresh_interval_us=refresh_interval_us,
                retention_time_us=retention_time_us,
            )
            device_refresh_mw = refresh_result.refresh_power_mw_total

    effective_values.update(
        hit_latency_ns=hit_latency_ns,
        write_latency_ns=write_latency_ns,
        hit_energy_nj=hit_energy_nj,
        write_energy_nj=write_energy_nj,
        # The Device Library is authoritative for cell/static power.  The raw
        # DESTINY tag/data leakage remains in raw_metrics for auditing, but is
        # not added again because that made the v1 static result dominate by
        # hundreds of mW and double-counted an obsolete SRAM-based tag model.
        leakage_power_mw=device_leakage_mw,
        data_array_leakage_power_mw=device_leakage_mw,
        tag_array_leakage_power_mw=0.0,
    )
    effective = DestinyMetrics(**effective_values)
    return (
        effective,
        device_leakage_mw,
        device_refresh_mw,
        edram_read,
        refresh_result.to_dict(),
        bti_result,
    )


def build_layer(raw: Dict[str, Any], raw_metrics: DestinyMetrics, repo_root: Path,
                global_ber_max: float,
                device_library: Dict[str, Dict[str, Any]],
                model_library: Optional[Dict[str, Any]] = None,
                features: Optional[Dict[str, bool]] = None) -> LayerSpec:
    name = str(raw["name"])
    device = str(raw["device"])
    require(device in device_library,
            f"{name}: device must be one of {sorted(device_library)}")
    entry = copy.deepcopy(device_library[device])
    device_family = str(entry["family"])
    destiny_config = _resolve_path(repo_root, str(raw["destiny_config"]))
    require(_destiny_device_family(destiny_config) == device_family,
            f"{name}: {device} requires a {device_family} DESTINY cell file")
    policy = str(raw.get("replacement_policy", "LRU")).upper()
    require(policy in VALID_POLICIES,
            f"{name}: replacement_policy must be one of {sorted(VALID_POLICIES)}")
    capacity_bytes = int(raw["capacity_bytes"])
    associativity = int(raw["associativity"])
    line_bytes = int(raw["line_bytes"])
    destiny_model_capacity_bytes = int(
        raw.get("destiny_model_capacity_bytes", capacity_bytes)
    )
    require(destiny_model_capacity_bytes == raw_metrics.capacity_bytes,
            f"{name}: circuit proxy capacity {destiny_model_capacity_bytes} differs from DESTINY "
            f"capacity {raw_metrics.capacity_bytes}")
    require(associativity == raw_metrics.associativity,
            f"{name}: JSON associativity {associativity} differs from DESTINY "
            f"associativity {raw_metrics.associativity}")
    require(line_bytes == raw_metrics.line_bytes,
            f"{name}: JSON line size {line_bytes} differs from DESTINY "
            f"line size {raw_metrics.line_bytes}")
    banks = int(raw.get("banks", 1))
    require(banks > 0, f"{name}: banks must be positive")
    wear_efficiency = float(raw.get("wear_leveling_efficiency", 1.0))
    require(0.0 < wear_efficiency <= 1.0,
            f"{name}: wear_leveling_efficiency must be in (0, 1]")
    refresh_interval_us = float(raw.get("refresh_interval_us", 0.0))
    retention_time_us = float(raw.get("retention_time_us", 0.0))
    refresh_model = str(entry["refresh"]["model"])
    if refresh_model == "row_read_write":
        require(refresh_interval_us > 0.0,
                f"{name}: Si-eDRAM refresh_interval_us must be positive")
        require(retention_time_us > 0.0,
                f"{name}: Si-eDRAM retention_time_us must be positive")
        require(refresh_interval_us <= retention_time_us,
                f"{name}: Tref must not exceed Tret")
    elif refresh_model == "bti_row_read_write":
        require(device == "OSFET-eDRAM",
                f"{name}: BTI row refresh is supported only for OSFET-eDRAM")
        require(isinstance(entry.get("bti"), dict),
                f"{name}: OSFET BTI model is missing")
    endurance = entry["endurance"].get("writes_per_line")
    if endurance is None:
        require("bti_endurance_writes_per_line" in raw,
                f"{name}: {device} requires bti_endurance_writes_per_line")
        endurance = float(raw["bti_endurance_writes_per_line"])
    (effective_metrics, device_leakage_mw, device_refresh_mw,
     edram_read, refresh_result, bti_result) = _apply_device_library(
        raw_metrics,
        entry,
        capacity_bytes,
        line_bytes,
        banks,
        refresh_interval_us,
        retention_time_us,
    )
    if bti_result is not None:
        refresh_interval_us = float(refresh_result["refresh_interval_us"])
        retention_time_us = float(refresh_result["retention_time_us"])
    has_density_divisor = "density_divisor" in entry
    stacked_tiers = int(raw.get("stacked_tiers", 4 if has_density_divisor else 1))
    require(stacked_tiers > 0, f"{name}: stacked_tiers must be positive")
    models = model_library or {}
    feature_flags = dict(FEATURE_SWITCH_DEFAULTS)
    feature_flags.update(features or {})
    sense_amp_result: Optional[Dict[str, Any]] = None
    m3d_result: Optional[Dict[str, Any]] = None
    nonideal_result: Optional[Dict[str, Any]] = None
    static_components = {"data_cells_mw": device_leakage_mw}
    effective_values = asdict(effective_metrics)
    if (feature_flags["configurable_peripherals"] and
            "sense_amplifier" in entry and
            "sense_amplifier_models" in models):
        try:
            evaluated_sa = evaluate_sense_amp(
                raw_metrics,
                entry,
                dict(models["sense_amplifier_models"]),
                raw.get("sense_amp_type"),
            )
        except ValueError as exc:
            raise ScopeError(f"{name}: {exc}") from exc
        sense_amp_result = evaluated_sa.to_dict()
        sense_amp_result["configuration_enabled"] = True
        effective_values["hit_latency_ns"] = max(
            0.0,
            float(effective_values["hit_latency_ns"])
            + evaluated_sa.hit_latency_delta_ns,
        )
        effective_values["hit_energy_nj"] = max(
            0.0,
            float(effective_values["hit_energy_nj"])
            + evaluated_sa.hit_energy_delta_nj,
        )
        static_components["sense_amplifiers_mw"] = \
            evaluated_sa.selected_leakage_mw
    elif "sense_amplifier" in entry:
        interface = dict(entry["sense_amplifier"])
        native_type = str(interface.get(
            "destiny_native_type", raw_metrics.native_sense_amp_type or "voltage"
        )).lower()
        sense_amp_result = {
            "configuration_enabled": False,
            "selected_type": native_type,
            "destiny_native_type": native_type,
            "read_signal": str(interface.get("read_signal", "unspecified")),
            "supported_types": [native_type],
            "compatible": True,
            "selected_latency_ns": raw_metrics.sense_amp_latency_ns,
            "selected_energy_nj": raw_metrics.sense_amp_read_energy_nj,
            "selected_leakage_mw": raw_metrics.sense_amp_leakage_mw,
            "hit_latency_delta_ns": 0.0,
            "hit_energy_delta_nj": 0.0,
            "comparison_basis": (
                "configurable peripheral overlay disabled; use DESTINY native SA"
            ),
        }
        static_components["sense_amplifiers_mw"] = \
            raw_metrics.sense_amp_leakage_mw

    nonideal_entry = copy.deepcopy(entry)
    if not feature_flags["array_nonideal"]:
        disabled_nonideal = dict(nonideal_entry.get("nonideal", {}))
        disabled_nonideal["enabled"] = False
        nonideal_entry["nonideal"] = disabled_nonideal
    evaluated_nonideal = evaluate_nonideal(
        raw_metrics,
        float(effective_values["hit_latency_ns"]),
        line_bytes * 8,
        nonideal_entry,
        sense_amp_result,
        float(raw["ber"]),
    )
    nonideal_result = evaluated_nonideal.to_dict()
    nonideal_result["feature_enabled"] = feature_flags["array_nonideal"]
    effective_values["hit_latency_ns"] += \
        evaluated_nonideal.read_latency_penalty_ns
    effective_values["hit_energy_nj"] += \
        evaluated_nonideal.read_energy_penalty_nj

    m3d_cfg = dict(raw.get("m3d", {}))
    if not feature_flags["m3d"]:
        m3d_cfg["enabled"] = False
    if m3d_cfg or models.get("m3d_defaults"):
        try:
            evaluated_m3d = evaluate_m3d(
                dict(m3d_cfg),
                dict(models.get("m3d_defaults", {})),
                banks=banks,
                line_bits=line_bytes * 8,
                data_array_area_mm2=(
                    raw_metrics.data_array_area_mm2
                    * capacity_bytes / destiny_model_capacity_bytes
                ),
            )
        except (KeyError, ValueError) as exc:
            raise ScopeError(f"{name}: invalid M3D configuration: {exc}") from exc
        m3d_result = evaluated_m3d.to_dict()
        m3d_result["feature_enabled"] = feature_flags["m3d"]
        if evaluated_m3d.enabled:
            effective_values["hit_latency_ns"] += evaluated_m3d.latency_penalty_ns
            effective_values["write_latency_ns"] += evaluated_m3d.latency_penalty_ns
            effective_values["hit_energy_nj"] += evaluated_m3d.energy_penalty_nj
            effective_values["write_energy_nj"] += evaluated_m3d.energy_penalty_nj

    if models:
        address_bits = int(models.get("tag_address_bits", 49))
        state_bits = int(models.get("tag_state_bits_per_line", 2))
        tag_leak_pw = float(models.get("tag_sram_leakage_pw_per_bit", 27.5))
        sets = capacity_bytes // (line_bytes * associativity)
        index_bits = int(math.ceil(math.log2(max(1, sets))))
        offset_bits = int(math.log2(line_bytes))
        tag_bits_per_line = max(1, address_bits - index_bits - offset_bits + state_bits)
        tag_bits = (capacity_bytes // line_bytes) * tag_bits_per_line
        static_components["sram_tag_array_mw"] = tag_bits * tag_leak_pw * 1e-9
    static_total = sum(static_components.values())
    effective_values.update(
        leakage_power_mw=static_total,
        data_array_leakage_power_mw=device_leakage_mw,
        tag_array_leakage_power_mw=static_components.get("sram_tag_array_mw", 0.0),
    )
    effective_metrics = DestinyMetrics(**effective_values)
    effective_density_f2 = float(entry["density_f2"])
    if has_density_divisor:
        effective_density_f2 /= stacked_tiers
    return LayerSpec(
        name=name,
        device=device,
        device_family=device_family,
        destiny_config=destiny_config,
        capacity_bytes=capacity_bytes,
        destiny_model_capacity_bytes=destiny_model_capacity_bytes,
        associativity=associativity,
        line_bytes=line_bytes,
        replacement_policy=policy,
        banks=banks,
        peripheral_latency_ns=(
            float(raw.get("peripheral_latency_ns", 0.0))
            if feature_flags["configurable_peripherals"] else 0.0
        ),
        peripheral_energy_nj=(
            float(raw.get("peripheral_energy_nj", 0.0))
            if feature_flags["configurable_peripherals"] else 0.0
        ),
        ber=evaluated_nonideal.effective_ber,
        ber_max=float(raw.get("ber_max", global_ber_max)),
        allow_high_variation=bool(raw.get("allow_high_variation", False)),
        endurance_writes_per_line=float(endurance),
        wear_leveling_efficiency=wear_efficiency,
        refresh_interval_us=refresh_interval_us,
        retention_time_us=retention_time_us,
        estimated_writebacks_per_request=float(
            raw.get("estimated_writebacks_per_request", 0.0)
        ),
        device_rows_per_bank=math.ceil(capacity_bytes / (banks * line_bytes)),
        stacked_tiers=stacked_tiers,
        effective_density_f2=effective_density_f2,
        data_cell_area_f2=capacity_bytes * 8 * effective_density_f2,
        device_leakage_power_mw=device_leakage_mw,
        device_refresh_power_mw=device_refresh_mw,
        device_library_entry=entry,
        raw_metrics=raw_metrics,
        metrics=effective_metrics,
        edram_read=edram_read,
        refresh=refresh_result,
        bti=bti_result,
        sense_amp=sense_amp_result,
        m3d=m3d_result,
        nonideal=nonideal_result,
        static_power_components=static_components,
    )


def build_crossbar(raw: Dict[str, Any]) -> CrossbarSpec:
    hops = int(raw.get("hops", 1))
    router_cycles = float(raw.get("router_pipeline_cycles", 0.0))
    link_cycles = float(raw.get("link_traversal_cycles", 0.0))
    detailed = router_cycles > 0.0 or link_cycles > 0.0
    directional_cycles = hops * (router_cycles + link_cycles)
    request_bits = int(raw.get("request_bits", 0))
    response_bits = int(raw.get("response_bits", 0))
    transaction_bits = int(raw.get(
        "transaction_bits", request_bits + response_bits
    ))
    spec = CrossbarSpec(
        name=str(raw["name"]),
        request_cycles=(directional_cycles if detailed else
                        float(raw.get("request_cycles", 1.0))),
        response_cycles=(directional_cycles if detailed else
                         float(raw.get("response_cycles", 1.0))),
        clock_ghz=float(raw.get("clock_ghz", 1.0)),
        energy_pj_per_bit=float(raw.get("energy_pj_per_bit", 0.201)),
        transaction_bits=transaction_bits,
        topology=str(raw.get("topology", "legacy crossbar")),
        hops=hops,
        router_pipeline_cycles=router_cycles,
        link_traversal_cycles=link_cycles,
        request_bits=request_bits,
        response_bits=response_bits,
        router_energy_pj_per_bit=float(raw.get("router_energy_pj_per_bit", 0.0)),
        link_energy_pj_per_bit_per_mm=float(
            raw.get("link_energy_pj_per_bit_per_mm", 0.0)
        ),
        link_length_mm=float(raw.get("link_length_mm", 0.0)),
    )
    require(spec.request_cycles >= 0.0 and spec.response_cycles >= 0.0,
            f"{spec.name}: crossbar cycles must be non-negative")
    require(spec.hops >= 0, f"{spec.name}: hops must be non-negative")
    require(spec.router_pipeline_cycles >= 0.0 and
            spec.link_traversal_cycles >= 0.0,
            f"{spec.name}: detailed NoC cycles must be non-negative")
    require(spec.clock_ghz > 0.0, f"{spec.name}: clock_ghz must be positive")
    require(spec.energy_pj_per_bit >= 0.0,
            f"{spec.name}: energy_pj_per_bit must be non-negative")
    require(spec.transaction_bits >= 0 and spec.request_bits >= 0 and
            spec.response_bits >= 0,
            f"{spec.name}: transaction widths must be non-negative")
    require(spec.router_energy_pj_per_bit >= 0.0 and
            spec.link_energy_pj_per_bit_per_mm >= 0.0 and
            spec.link_length_mm >= 0.0,
            f"{spec.name}: detailed NoC energy parameters must be non-negative")
    require(spec.transaction_bits > 0,
            f"{spec.name}: transaction_bits must be positive")
    require(spec.hops > 0, f"{spec.name}: hops must be positive")
    return spec


def geometry_audit(layer: LayerSpec) -> Dict[str, Any]:
    def dimensions(text: str) -> List[int]:
        return [int(value) for value in re.findall(r"\d+", text)]

    bank = dimensions(layer.raw_metrics.bank_organization)
    mat = dimensions(layer.raw_metrics.mat_organization)
    subarray = [layer.raw_metrics.subarray_rows,
                layer.raw_metrics.subarray_columns]
    factors = [value for value in bank + mat + subarray if value > 0]
    reconstructed = math.prod(factors) if factors else 0
    expected = layer.capacity_bytes * 8
    return {
        "bank_dimensions": bank,
        "mat_dimensions": mat,
        "subarray_dimensions": subarray,
        "reconstructed_physical_bits": reconstructed,
        "configured_capacity_bits": expected,
        "capacity_reconstruction_ratio": reconstructed / expected if expected else 0.0,
        "interpretation": (
            "bank dimensions count mats; mat dimensions count subarrays per mat; "
            "only subarray dimensions are physical cell rows/columns. A small mat "
            "count shortens local word/bit lines while the bank-level mat count "
            "reconstructs the full capacity."
        ),
    }


def _parse_address(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    raise ScopeError(f"invalid trace address: {value!r}")


def _trace_requests(path: Path, line_bytes: int) -> Iterator[Tuple[str, int]]:
    require(path.is_file(), f"trace file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
                op = str(item["op"]).lower()
                address = _parse_address(item["address"])
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                raise ScopeError(f"invalid trace line {line_number}: {line}") from exc
            require(op in {"load", "store"},
                    f"trace line {line_number}: op must be load or store")
            yield op, address // line_bytes


def _zipf_sampler(working_set_lines: int, alpha: float,
                  rng: random.Random) -> Iterator[int]:
    require(working_set_lines > 0, "working_set_bytes must cover at least one line")
    require(working_set_lines <= 2_000_000,
            "synthetic working set is too large for the built-in sampler")
    if alpha == 0.0:
        while True:
            yield rng.randrange(working_set_lines)
    require(alpha > 0.0, "zipf_alpha must be non-negative")
    cumulative: List[float] = []
    total = 0.0
    for rank in range(1, working_set_lines + 1):
        total += 1.0 / (rank ** alpha)
        cumulative.append(total)
    while True:
        draw = rng.random() * total
        yield bisect.bisect_left(cumulative, draw)


def estimate_hit_rates(layers: Sequence[LayerSpec], workload: Dict[str, Any],
                       repo_root: Path) -> HitRateResult:
    model = dict(workload.get("hit_rate_model", {}))
    mode = str(model.get("mode", "synthetic")).lower()
    if mode == "cpp_openvla_trace":
        return _cpp_openvla_hit_rates(layers, model, repo_root)

    read_fraction = float(workload.get("read_fraction", 0.5))
    require(0.0 <= read_fraction <= 1.0, "read_fraction must be in [0, 1]")

    if mode == "fixed":
        rates = tuple(float(item["hit_rate"]) for item in workload["fixed_hit_rates"])
        require(len(rates) == len(layers), "fixed_hit_rates must contain one rate per layer")
        require(all(0.0 <= value <= 1.0 for value in rates),
                "fixed hit rates must be in [0, 1]")
        writebacks = tuple(layer.estimated_writebacks_per_request for layer in layers)
        return HitRateResult(
            hit_rates=rates,
            accesses=tuple(0 for _ in layers),
            hits=tuple(0 for _ in layers),
            writebacks_per_request=writebacks,
            offchip_writebacks_per_request=0.0,
            observed_read_fraction=read_fraction,
        )

    if mode == "operator":
        windows = model.get("reuse_window_bytes")
        locality = model.get("locality_factor", 2.0)
        reference_associativity = model.get("reference_associativity", 8)
        require(isinstance(windows, (dict, list)),
                "operator reuse_window_bytes must be an object or list")

        def per_layer(value: Any, index: int, name: str) -> float:
            if isinstance(value, dict):
                require(name in value, f"operator model missing {name}")
                return float(value[name])
            if isinstance(value, list):
                require(len(value) == len(layers),
                        "operator list must contain one value per layer")
                return float(value[index])
            return float(value)

        policy_factors = {"LRU": 1.0, "FIFO": 0.96, "RANDOM": 0.90}
        maximum = float(model.get("max_hit_rate", 0.95))
        minimum = float(model.get("min_hit_rate", 0.0))
        require(0.0 <= minimum <= maximum <= 1.0,
                "operator hit-rate bounds must satisfy 0 <= min <= max <= 1")
        rates: List[float] = []
        for index, layer in enumerate(layers):
            window = per_layer(windows, index, layer.name)
            factor = per_layer(locality, index, layer.name)
            reference = per_layer(
                reference_associativity, index, layer.name
            )
            require(window > 0.0 and factor >= 0.0 and reference > 0.0,
                    f"{layer.name}: invalid operator locality parameters")
            associativity_factor = clamp(
                1.0 + 0.08 * math.log2(layer.associativity / reference),
                0.75,
                1.25,
            )
            exponent = (
                factor
                * layer.capacity_bytes / window
                * associativity_factor
                * policy_factors[layer.replacement_policy]
            )
            rates.append(clamp(1.0 - math.exp(-exponent), minimum, maximum))
        writebacks = tuple(layer.estimated_writebacks_per_request for layer in layers)
        return HitRateResult(
            hit_rates=tuple(rates),
            accesses=tuple(0 for _ in layers),
            hits=tuple(0 for _ in layers),
            writebacks_per_request=writebacks,
            offchip_writebacks_per_request=float(
                model.get("offchip_writebacks_per_request", 0.0)
            ),
            observed_read_fraction=read_fraction,
        )

    require(mode in {"synthetic", "trace", "openvla_operator_trace"},
            "hit_rate_model.mode must be fixed, operator, synthetic, trace, "
            "openvla_operator_trace, or cpp_openvla_trace")
    line_sizes = {layer.line_bytes for layer in layers}
    require(len(line_sizes) == 1,
            "WB+WA hit-rate simulation requires one cache-line size across all levels")
    line_bytes = next(iter(line_sizes))
    seed = int(model.get("seed", 7))
    rng = random.Random(seed)
    caches = [
        SetAssociativeCache(
            layer.capacity_bytes,
            layer.line_bytes,
            layer.associativity,
            layer.replacement_policy,
            random.Random(seed + index + 1),
        )
        for index, layer in enumerate(layers)
    ]
    accesses = [0] * len(layers)
    hits = [0] * len(layers)
    writebacks = [0] * len(layers)
    offchip_writebacks = 0
    sampled_requests = 0
    sampled_reads = 0

    def writeback(level: int, line: int, record: bool) -> None:
        nonlocal offchip_writebacks
        if level >= len(caches):
            if record:
                offchip_writebacks += 1
            return
        if record:
            writebacks[level] += 1
        evicted = caches[level].insert(line, dirty=True)
        if evicted is not None and evicted.dirty:
            writeback(level + 1, evicted.line, record)

    def process(op: str, line: int, record: bool) -> None:
        nonlocal sampled_requests, sampled_reads
        if record:
            sampled_requests += 1
            sampled_reads += int(op == "load")
        missed: List[int] = []
        hit_level: Optional[int] = None
        for index, cache in enumerate(caches):
            if record:
                accesses[index] += 1
            # A store miss is an RFO read below L1; only an L1 store hit is dirty here.
            hit = cache.probe(line, mark_dirty=(op == "store" and index == 0))
            if hit:
                if record:
                    hits[index] += 1
                hit_level = index
                break
            missed.append(index)

        for index in reversed(missed):
            dirty = op == "store" and index == 0
            evicted = caches[index].insert(line, dirty=dirty)
            if evicted is not None and evicted.dirty:
                writeback(index + 1, evicted.line, record)

        # hit_level is intentionally unused after fills; retaining it makes the
        # demand path explicit and documents that no lower level is accessed.
        _ = hit_level

    trace_metadata: Optional[Dict[str, Any]] = None
    if mode == "openvla_operator_trace":
        operator_trace = build_trace(model, line_bytes)
        warmup_repetitions = int(model.get("warmup_repetitions", 1))
        sample_repetitions = int(model.get("sample_repetitions", 1))
        require(warmup_repetitions >= 0 and sample_repetitions > 0,
                "OpenVLA trace repetitions are invalid")
        warmup = len(operator_trace.events) * warmup_repetitions
        samples = len(operator_trace.events) * sample_repetitions
        requests = repeated(
            operator_trace.events, warmup_repetitions + sample_repetitions
        )
        trace_metadata = operator_trace.summary()
        read_fraction = operator_trace.read_fraction
    else:
        warmup = int(model.get("warmup_accesses", 10_000))
        samples = int(model.get("sample_accesses", 100_000))
    require(warmup >= 0 and samples > 0,
            "warmup_accesses must be non-negative and sample_accesses positive")

    if mode == "trace":
        trace_path = _resolve_path(repo_root, str(model["path"]))
        requests = _trace_requests(trace_path, line_bytes)
    elif mode == "synthetic":
        working_set_bytes = int(model["working_set_bytes"])
        working_set_lines = math.ceil(working_set_bytes / line_bytes)
        alpha = float(model.get("zipf_alpha", 1.0))
        addresses = _zipf_sampler(working_set_lines, alpha, rng)

        def generated() -> Iterator[Tuple[str, int]]:
            for line in addresses:
                yield ("load" if rng.random() < read_fraction else "store"), line

        requests = generated()

    for ordinal, (op, line) in enumerate(requests):
        if ordinal >= warmup + samples:
            break
        process(op, line, record=ordinal >= warmup)

    require(sampled_requests > 0,
            "trace contains no sampled requests after warmup")
    rates = tuple(
        hits[index] / accesses[index] if accesses[index] else 0.0
        for index in range(len(layers))
    )
    return HitRateResult(
        hit_rates=rates,
        accesses=tuple(accesses),
        hits=tuple(hits),
        writebacks_per_request=tuple(value / sampled_requests for value in writebacks),
        offchip_writebacks_per_request=offchip_writebacks / sampled_requests,
        observed_read_fraction=sampled_reads / sampled_requests,
        trace_metadata=trace_metadata,
    )


class ScopeModel:
    def __init__(self, layers: Sequence[LayerSpec], crossbars: Sequence[CrossbarSpec],
                 offchip: OffChipSpec, workload: Dict[str, Any],
                 hit_rates: HitRateResult, refill_on_critical_path: bool = False,
                 core_to_l1_latency_ns: float = 0.0) -> None:
        require(len(layers) == 3, "SCOPE requires exactly three cache levels")
        require(len(crossbars) == 2, "SCOPE requires L1-L2 and L2-L3 crossbars")
        self.layers = tuple(layers)
        self.crossbars = tuple(crossbars)
        self.offchip = offchip
        self.workload = workload
        self.hit_rates = hit_rates
        self.refill_on_critical_path = refill_on_critical_path
        self.core_to_l1_latency_ns = core_to_l1_latency_ns

    @property
    def read_fraction(self) -> float:
        return self.hit_rates.observed_read_fraction

    @property
    def write_fraction(self) -> float:
        return 1.0 - self.read_fraction

    def _access_latency(self, layer: LayerSpec) -> float:
        base = (
            self.read_fraction * layer.metrics.hit_latency_ns
            + self.write_fraction * layer.metrics.write_latency_ns
            + layer.peripheral_latency_ns
        )
        return self._refresh_adjusted_latency(layer, base)

    @staticmethod
    def _refresh_adjusted_latency(layer: LayerSpec, base: float) -> float:
        refresh = layer.refresh or {}
        occupancy = float(refresh.get("bandwidth_occupancy", 0.0))
        if not refresh.get("enabled", False):
            return base
        # Periodic refresh removes this share of service bandwidth.  At or
        # above 100% the cache is unschedulable; retain a finite audit value.
        return base / max(1e-6, 1.0 - occupancy)

    def _access_energy(self, layer: LayerSpec) -> float:
        return (
            self.read_fraction * layer.metrics.hit_energy_nj
            + self.write_fraction * layer.metrics.write_energy_nj
            + layer.peripheral_energy_nj
        )

    def _refresh_power(self, layer: LayerSpec) -> float:
        if layer.device_family != "eDRAM":
            return 0.0
        return layer.device_refresh_power_mw

    def average(self) -> Dict[str, Any]:
        reach = 1.0
        latency_total = self.core_to_l1_latency_ns
        expected_dynamic_energy = 0.0
        latency_breakdown: List[Dict[str, Any]] = []
        dynamic_breakdown: List[Dict[str, Any]] = []
        reaches: List[float] = []

        if self.core_to_l1_latency_ns:
            latency_breakdown.append({
                "component": "core->L1",
                "reach_probability": 1.0,
                "raw_latency_ns": self.core_to_l1_latency_ns,
                "weighted_latency_ns": self.core_to_l1_latency_ns,
            })

        for index, layer in enumerate(self.layers):
            reaches.append(reach)
            access_latency = self._access_latency(layer)
            access_energy = self._access_energy(layer)
            noc_latency = 0.0 if index == 0 else self.crossbars[index - 1].latency_ns
            noc_energy = 0.0 if index == 0 else self.crossbars[index - 1].energy_nj
            weighted_latency = reach * (noc_latency + access_latency)
            weighted_energy = reach * (noc_energy + access_energy)
            latency_total += weighted_latency
            expected_dynamic_energy += weighted_energy
            latency_breakdown.append({
                "component": f"{layer.name} access" if index == 0 else
                             f"{self.crossbars[index - 1].name} + {layer.name} access",
                "reach_probability": reach,
                "raw_latency_ns": noc_latency + access_latency,
                "weighted_latency_ns": weighted_latency,
            })
            dynamic_breakdown.append({
                "component": f"{layer.name} access" if index == 0 else
                             f"{self.crossbars[index - 1].name} + {layer.name} access",
                "reach_probability": reach,
                "raw_energy_nj": noc_energy + access_energy,
                "weighted_energy_nj": weighted_energy,
            })
            reach *= 1.0 - self.hit_rates.hit_rates[index]

        offchip_latency = reach * self.offchip.latency_ns
        offchip_energy = reach * self.offchip.energy_nj
        latency_total += offchip_latency
        expected_dynamic_energy += offchip_energy
        latency_breakdown.append({
            "component": "off-chip memory",
            "reach_probability": reach,
            "raw_latency_ns": self.offchip.latency_ns,
            "weighted_latency_ns": offchip_latency,
        })
        dynamic_breakdown.append({
            "component": "off-chip memory",
            "reach_probability": reach,
            "raw_energy_nj": self.offchip.energy_nj,
            "weighted_energy_nj": offchip_energy,
        })

        # Buffered dirty evictions are not placed on the demand critical path,
        # but their link and destination-write energy must contribute to Eavg.
        for index, writebacks_per_request in enumerate(
            self.hit_rates.writebacks_per_request
        ):
            if index == 0 or writebacks_per_request <= 0.0:
                continue
            layer = self.layers[index]
            link_energy = self.crossbars[index - 1].energy_nj
            raw_energy = (
                link_energy
                + layer.metrics.write_energy_nj
                + layer.peripheral_energy_nj
            )
            weighted_energy = writebacks_per_request * raw_energy
            expected_dynamic_energy += weighted_energy
            dynamic_breakdown.append({
                "component": f"WB to {layer.name}",
                "reach_probability": writebacks_per_request,
                "raw_energy_nj": raw_energy,
                "weighted_energy_nj": weighted_energy,
                "off_critical_path": True,
            })
        if self.hit_rates.offchip_writebacks_per_request > 0.0:
            weighted_energy = (
                self.hit_rates.offchip_writebacks_per_request * self.offchip.energy_nj
            )
            expected_dynamic_energy += weighted_energy
            dynamic_breakdown.append({
                "component": "WB to off-chip memory",
                "reach_probability": self.hit_rates.offchip_writebacks_per_request,
                "raw_energy_nj": self.offchip.energy_nj,
                "weighted_energy_nj": weighted_energy,
                "off_critical_path": True,
            })

        access_rate = float(self.workload["memory_access_rate_per_s"])
        dynamic_power_mw = access_rate * expected_dynamic_energy * 1e-6
        for item in dynamic_breakdown:
            item["power_mw"] = access_rate * item["weighted_energy_nj"] * 1e-6

        static_breakdown = [
            {"component": layer.name, "power_mw": layer.metrics.leakage_power_mw}
            for layer in self.layers
        ]
        refresh_breakdown = [
            {"component": layer.name, "power_mw": self._refresh_power(layer)}
            for layer in self.layers
        ]
        static_power_mw = sum(item["power_mw"] for item in static_breakdown)
        refresh_power_mw = sum(item["power_mw"] for item in refresh_breakdown)
        total_power_mw = dynamic_power_mw + static_power_mw + refresh_power_mw
        fom = 1.0 / (latency_total * total_power_mw) if total_power_mw > 0.0 else math.inf

        constraints = self._constraints(tuple(reaches))
        feasible = all(item["pass"] for item in constraints)
        access_equations = []
        for layer in self.layers:
            refresh = layer.refresh or {}
            base_latency = (
                self.read_fraction * layer.metrics.hit_latency_ns
                + self.write_fraction * layer.metrics.write_latency_ns
                + layer.peripheral_latency_ns
            )
            access_equations.append({
                "layer": layer.name,
                "rho_r": self.read_fraction,
                "rho_w": self.write_fraction,
                "read_latency_ns": layer.metrics.hit_latency_ns,
                "write_latency_ns": layer.metrics.write_latency_ns,
                "peripheral_latency_ns": layer.peripheral_latency_ns,
                "base_access_latency_ns": base_latency,
                "refresh_bandwidth_occupancy": float(
                    refresh.get("bandwidth_occupancy", 0.0)
                ),
                "effective_access_latency_ns": self._access_latency(layer),
                "read_energy_nj": layer.metrics.hit_energy_nj,
                "write_energy_nj": layer.metrics.write_energy_nj,
                "peripheral_energy_nj": layer.peripheral_energy_nj,
                "access_energy_nj": self._access_energy(layer),
            })
        return {
            "hit_rates": list(self.hit_rates.hit_rates),
            "conditional_accesses": list(self.hit_rates.accesses),
            "conditional_hits": list(self.hit_rates.hits),
            "average_latency_ns": latency_total,
            "latency_breakdown": latency_breakdown,
            "expected_dynamic_energy_nj_per_request": expected_dynamic_energy,
            "dynamic_power_mw": dynamic_power_mw,
            "dynamic_power_breakdown": dynamic_breakdown,
            "static_power_mw": static_power_mw,
            "static_power_breakdown": static_breakdown,
            "refresh_power_mw": refresh_power_mw,
            "refresh_power_breakdown": refresh_breakdown,
            "average_power_mw": total_power_mw,
            "fom_per_ns_mw": fom,
            "feasible": feasible,
            "constraints": constraints,
            "offchip_reach_probability": reach,
            "workload_access_mix": {
                "rho_r": self.read_fraction,
                "rho_w": self.write_fraction,
                "source": "observed load/store counts from the selected trace",
            },
            "per_layer_access_equations": access_equations,
        }

    def _constraints(self, reaches: Tuple[float, ...]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        access_rate = float(self.workload["memory_access_rate_per_s"])
        lifetime_seconds = float(self.workload["lifetime_seconds"])
        for index, layer in enumerate(self.layers):
            results.append({
                "layer": layer.name,
                "constraint": "BER",
                "value": layer.ber,
                "limit": layer.ber_max,
                "pass": layer.ber <= layer.ber_max,
            })
            nonideal = layer.nonideal or {}
            disturb_target = float(
                nonideal.get("read_disturb_target_per_read", 0.0)
            )
            if disturb_target > 0.0:
                disturb_value = float(
                    nonideal.get("read_disturb_probability_per_read", 0.0)
                )
                results.append({
                    "layer": layer.name,
                    "constraint": "STT read-disturb probability/read",
                    "value": disturb_value,
                    "limit": disturb_target,
                    "pass": disturb_value <= disturb_target * (1.0 + 1e-9),
                    "detail": (
                        "The limiter reduces read current when necessary; write "
                        "current and write latency are not altered."
                    ),
                })
            high_variation = "high" in str(
                layer.device_library_entry["variation"]
            ).lower()
            results.append({
                "layer": layer.name,
                "constraint": "high_variation",
                "value": int(high_variation),
                "limit": int(layer.allow_high_variation),
                "pass": not high_variation or layer.allow_high_variation,
                "detail": layer.device_library_entry["variation"],
            })
            if index == 0:
                writes_per_request = self.write_fraction
            else:
                estimated = self.hit_rates.writebacks_per_request[index]
                writes_per_request = (
                    estimated if estimated > 0.0
                    else layer.estimated_writebacks_per_request
                )
            writes_life = (
                access_rate
                * lifetime_seconds
                * writes_per_request
                / max(1, layer.lines)
                / layer.wear_leveling_efficiency
            )
            results.append({
                "layer": layer.name,
                "constraint": "endurance",
                "value": writes_life,
                "limit": layer.endurance_writes_per_line,
                "pass": writes_life <= layer.endurance_writes_per_line,
                "writes_per_request": writes_per_request,
            })
            refresh = layer.refresh or {}
            if refresh.get("enabled", False):
                results.append({
                    "layer": layer.name,
                    "constraint": "refresh<=retention",
                    "value": layer.refresh_interval_us,
                    "limit": layer.retention_time_us,
                    "pass": 0.0 < layer.refresh_interval_us <= layer.retention_time_us,
                })
                occupancy = float(refresh["bandwidth_occupancy"])
                results.append({
                    "layer": layer.name,
                    "constraint": "refresh bandwidth schedulable",
                    "value": occupancy,
                    "limit": 1.0,
                    "pass": occupancy < 1.0,
                    "detail": "occupancy>=1 means refresh alone exceeds bank bandwidth",
                })
            else:
                results.append({
                    "layer": layer.name,
                    "constraint": "refresh<=retention",
                    "value": None,
                    "limit": None,
                    "pass": True,
                    "not_applicable": True,
                })
                results.append({
                    "layer": layer.name,
                    "constraint": "refresh bandwidth schedulable",
                    "value": None,
                    "limit": None,
                    "pass": True,
                    "not_applicable": True,
                })
        return results

    @staticmethod
    def _scaled_path(
        layer: LayerSpec, total_latency_ns: float, total_energy_nj: float,
        kind: str,
    ) -> List[Dict[str, Any]]:
        raw = layer.raw_metrics
        if kind == "tag":
            latency_weights = [
                max(0.0, raw.tag_lookup_latency_ns - raw.tag_comparator_latency_ns),
                max(0.0, raw.tag_comparator_latency_ns),
            ]
            energy_weights = [0.75 * raw.tag_lookup_energy_nj,
                              0.25 * raw.tag_lookup_energy_nj]
            labels = [
                f"{layer.name} tag decoder + tag SRAM",
                f"{layer.name} tag comparator",
            ]
        elif kind == "write":
            m3d = layer.m3d or {}
            latency_weights = [
                max(0.0, raw.tag_lookup_latency_ns),
                max(0.0, raw.data_routing_latency_ns),
                max(0.0, raw.data_predecoder_latency_ns
                    + raw.data_row_decoder_latency_ns),
                max(0.0, raw.write_latency_ns - raw.tag_lookup_latency_ns),
                max(0.0, float(m3d.get("latency_penalty_ns", 0.0))),
            ]
            energy_weights = [
                max(0.0, raw.tag_lookup_energy_nj),
                max(0.0, raw.write_energy_nj * 0.10),
                max(0.0, raw.write_energy_nj * 0.15),
                max(0.0, raw.write_energy_nj * 0.75),
                max(0.0, float(m3d.get("energy_penalty_nj", 0.0))),
            ]
            labels = [
                f"{layer.name} tag lookup + comparator",
                f"{layer.name} bank routing",
                f"{layer.name} predecoder + row decoder/wordline",
                f"{layer.name} cell write + bitline restore",
                f"{layer.name} M3D vertical links",
            ]
        else:
            sa = layer.sense_amp or {}
            m3d = layer.m3d or {}
            nonideal = layer.nonideal or {}
            edram = layer.edram_read or {}
            selected_sa = str(sa.get("selected_type", raw.native_sense_amp_type
                                     or "voltage"))
            cell_latency = float(edram.get(
                "cell_read_latency_ns", raw.rbl_delay_ns
            ))
            latency_weights = [
                max(0.0, raw.tag_lookup_latency_ns
                    - raw.tag_comparator_latency_ns),
                max(0.0, raw.tag_comparator_latency_ns),
                max(0.0, raw.data_routing_latency_ns),
                max(0.0, raw.data_predecoder_latency_ns
                    + raw.data_row_decoder_latency_ns),
                max(0.0, cell_latency),
                max(0.0, float(nonideal.get(
                    "read_latency_penalty_ns", 0.0
                ))),
                max(0.0, float(sa.get(
                    "selected_latency_ns", raw.sense_amp_latency_ns
                ))),
                max(0.0, raw.data_mux_latency_ns + raw.data_precharge_latency_ns),
                max(0.0, float(m3d.get("latency_penalty_ns", 0.0))),
            ]
            peripheral_energy = max(
                0.0, raw.peripheral_read_energy_nj
                - raw.sense_amp_read_energy_nj
            )
            energy_weights = [
                max(0.0, 0.75 * raw.tag_lookup_energy_nj),
                max(0.0, 0.25 * raw.tag_lookup_energy_nj),
                0.20 * peripheral_energy,
                0.25 * peripheral_energy,
                max(0.0, raw.rbl_read_energy_nj),
                max(0.0, float(nonideal.get(
                    "read_energy_penalty_nj", 0.0
                ))),
                max(0.0, float(sa.get(
                    "selected_energy_nj", raw.sense_amp_read_energy_nj
                ))),
                0.55 * peripheral_energy,
                max(0.0, float(m3d.get("energy_penalty_nj", 0.0))),
            ]
            labels = [
                f"{layer.name} tag decoder + tag SRAM",
                f"{layer.name} tag comparator",
                f"{layer.name} bank routing",
                f"{layer.name} predecoder + row decoder/wordline",
                f"{layer.name} cell discharge + RBL",
                f"{layer.name} array IR/coupling/crosstalk penalty",
                f"{layer.name} {selected_sa} sense amplifier",
                f"{layer.name} column mux + precharge/output",
                f"{layer.name} M3D vertical links",
            ]

        latency_sum = sum(latency_weights)
        energy_sum = sum(energy_weights)
        if latency_sum <= 0.0:
            latency_weights = [1.0] * len(labels)
            latency_sum = float(len(labels))
        if energy_sum <= 0.0:
            energy_weights = [1.0] * len(labels)
            energy_sum = float(len(labels))
        return [
            {
                "component": label,
                "latency_ns": total_latency_ns * latency / latency_sum,
                "energy_nj": total_energy_nj * energy / energy_sum,
                "raw_model_latency_weight_ns": latency,
                "raw_model_energy_weight_nj": energy,
            }
            for label, latency, energy in zip(
                labels, latency_weights, energy_weights
            )
            if latency > 0.0 or energy > 0.0
        ]

    def instruction(self, op: str, hit_level: str) -> Dict[str, Any]:
        op = op.lower()
        hit_level = hit_level.upper()
        require(op in {"load", "store"}, "instruction op must be load or store")
        valid_targets = {layer.name.upper() for layer in self.layers} | {"OFF", "MEM"}
        require(hit_level in valid_targets,
                f"hit_level must be one of {sorted(valid_targets)}")
        if hit_level == "MEM":
            hit_level = "OFF"
        target = len(self.layers) if hit_level == "OFF" else next(
            index for index, layer in enumerate(self.layers)
            if layer.name.upper() == hit_level
        )
        breakdown: List[Dict[str, Any]] = []

        def add(component: str, latency_ns: float, energy_nj: float,
                critical: bool = True) -> None:
            breakdown.append({
                "component": component,
                "latency_ns": latency_ns if critical else 0.0,
                "physical_latency_ns": latency_ns,
                "energy_nj": energy_nj,
                "on_critical_path": critical,
            })

        def add_path(layer: LayerSpec, kind: str, latency_ns: float,
                     energy_nj: float, critical: bool = True) -> None:
            for item in self._scaled_path(layer, latency_ns, energy_nj, kind):
                add(item["component"], item["latency_ns"], item["energy_nj"],
                    critical=critical)
                breakdown[-1]["raw_model_latency_weight_ns"] = \
                    item["raw_model_latency_weight_ns"]
                breakdown[-1]["raw_model_energy_weight_nj"] = \
                    item["raw_model_energy_weight_nj"]

        if self.core_to_l1_latency_ns:
            add("core->L1", self.core_to_l1_latency_ns, 0.0)

        if op == "store" and target == 0:
            layer = self.layers[0]
            add_path(layer, "write", self._refresh_adjusted_latency(
                layer, layer.metrics.write_latency_ns),
                layer.metrics.write_energy_nj)
        else:
            missed = list(range(min(target, len(self.layers))))
            for index in missed:
                layer = self.layers[index]
                add_path(layer, "tag", self._refresh_adjusted_latency(
                    layer, layer.metrics.miss_latency_ns),
                    layer.metrics.miss_energy_nj)
                if index < len(self.crossbars):
                    link = self.crossbars[index]
                    add(link.name, link.latency_ns, link.energy_nj)

            if target < len(self.layers):
                layer = self.layers[target]
                add_path(layer, "read", self._refresh_adjusted_latency(
                    layer, layer.metrics.hit_latency_ns),
                    layer.metrics.hit_energy_nj)
            else:
                add("off-chip read", self.offchip.latency_ns, self.offchip.energy_nj)

            for index in reversed(missed):
                layer = self.layers[index]
                store_commit = op == "store" and index == 0
                critical = self.refill_on_critical_path or store_commit
                label = f"{layer.name} allocate + store" if store_commit else \
                        f"{layer.name} refill"
                add_path(layer, "write", self._refresh_adjusted_latency(
                    layer, layer.metrics.write_latency_ns),
                    layer.metrics.write_energy_nj, critical=critical)

        latency_ns = sum(item["latency_ns"] for item in breakdown)
        dynamic_energy_nj = sum(item["energy_nj"] for item in breakdown)
        static_power_mw = sum(layer.metrics.leakage_power_mw for layer in self.layers)
        refresh_power_mw = sum(self._refresh_power(layer) for layer in self.layers)
        dynamic_power_mw = (
            1000.0 * dynamic_energy_nj / latency_ns if latency_ns > 0.0 else 0.0
        )
        return {
            "op": op,
            "hit_level": hit_level,
            "write_policy": "write-back + write-allocate",
            "latency_ns": latency_ns,
            "dynamic_energy_nj": dynamic_energy_nj,
            "serialized_dynamic_power_mw": dynamic_power_mw,
            "static_power_mw": static_power_mw,
            "refresh_power_mw": refresh_power_mw,
            "serialized_total_power_mw": dynamic_power_mw + static_power_mw + refresh_power_mw,
            "breakdown": breakdown,
        }


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeError(f"failed to load JSON config {path}: {exc}") from exc
    require(isinstance(data, dict), "top-level SCOPE config must be an object")
    return data


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_model_library(path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load a device library and recursively apply an optional base overlay."""
    resolved = path.resolve()
    raw = load_json(resolved)
    base_name = raw.pop("extends", None)
    if base_name is None:
        return raw
    root = repo_root or resolved.parent
    base_path = Path(str(base_name))
    if not base_path.is_absolute():
        candidate = resolved.parent / base_path
        base_path = candidate if candidate.exists() else root / base_path
    return _deep_merge(load_model_library(base_path, root), raw)


def select_workload(raw: Dict[str, Any], requested: Optional[str]) -> Dict[str, Any]:
    """Resolve a named application workload and its default Guidance."""
    resolved = copy.deepcopy(raw)
    profiles = resolved.get("workloads")
    if profiles is None:
        require("workload" in resolved, "config must contain workload or workloads")
        resolved.setdefault("guidance", {
            "name": "latency_power_product",
            "weights": {"latency": 1.0, "power": 1.0},
        })
        return resolved
    require(isinstance(profiles, dict) and profiles,
            "workloads must be a non-empty object")
    selected = requested or str(resolved.get("selected_workload", "attention"))
    require(selected in profiles,
            f"workload must be one of {sorted(profiles)}")
    profile = copy.deepcopy(profiles[selected])
    require(isinstance(profile, dict), f"workload {selected} must be an object")
    profile_guidance = profile.pop("guidance", None)
    resolved["workload"] = profile
    resolved["selected_workload"] = selected
    if "guidance" not in resolved:
        require(isinstance(profile_guidance, dict),
                f"workload {selected} must provide guidance")
        resolved["guidance"] = profile_guidance
    return resolved


def _canonical_guidance(guidance: Dict[str, Any]) -> Dict[str, Any]:
    weights_raw = guidance.get("weights", {"latency": 1.0, "power": 1.0})
    normalizers_raw = guidance.get("normalizers", {})
    limits_raw = guidance.get("limits", {})
    require(isinstance(weights_raw, dict) and weights_raw,
            "guidance.weights must be a non-empty object")
    weights: Dict[str, float] = {}
    normalizers: Dict[str, float] = {}
    limits: Dict[str, float] = {}
    for alias, value in weights_raw.items():
        require(alias in GUIDANCE_METRICS, f"unsupported Guidance metric: {alias}")
        metric = GUIDANCE_METRICS[alias]
        weight = float(value)
        require(weight >= 0.0, f"Guidance weight for {alias} must be non-negative")
        weights[metric] = weights.get(metric, 0.0) + weight
    require(any(value > 0.0 for value in weights.values()),
            "at least one Guidance weight must be positive")
    require(isinstance(normalizers_raw, dict), "guidance.normalizers must be an object")
    for alias, value in normalizers_raw.items():
        require(alias in GUIDANCE_METRICS,
                f"unsupported Guidance normalizer: {alias}")
        normalizer = float(value)
        require(normalizer > 0.0,
                f"Guidance normalizer for {alias} must be positive")
        normalizers[GUIDANCE_METRICS[alias]] = normalizer
    require(isinstance(limits_raw, dict), "guidance.limits must be an object")
    for alias, value in limits_raw.items():
        require(alias in GUIDANCE_METRICS, f"unsupported Guidance limit: {alias}")
        limit = float(value)
        require(limit >= 0.0, f"Guidance limit for {alias} must be non-negative")
        limits[GUIDANCE_METRICS[alias]] = limit
    return {
        "name": str(guidance.get("name", "custom")),
        "weights": weights,
        "normalizers": normalizers,
        "limits": limits,
        "destiny_optimization_target": guidance.get("destiny_optimization_target"),
    }


def guidance_score(report: Dict[str, Any], guidance: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a normalized multiplicative FoM; higher score is better."""
    canonical = _canonical_guidance(guidance)
    cost = 1.0
    terms: Dict[str, Dict[str, float]] = {}
    for metric, weight in canonical["weights"].items():
        if weight == 0.0:
            continue
        value = float(report[metric])
        normalizer = float(canonical["normalizers"].get(metric, 1.0))
        require(value > 0.0, f"Guidance metric {metric} must be positive")
        normalized = value / normalizer
        cost *= normalized ** weight
        terms[metric] = {
            "value": value,
            "normalizer": normalizer,
            "normalized": normalized,
            "weight": weight,
        }
    score = 1.0 / cost
    limit_checks = [
        {
            "metric": metric,
            "value": float(report[metric]),
            "limit": limit,
            "pass": float(report[metric]) <= limit,
        }
        for metric, limit in canonical["limits"].items()
    ]
    canonical["terms"] = terms
    canonical["cost"] = cost
    canonical["score"] = score
    canonical["limit_checks"] = limit_checks
    return canonical


def _guided_destiny_target(guidance: Dict[str, Any], read_fraction: float) -> str:
    canonical = _canonical_guidance(guidance)
    explicit = canonical.get("destiny_optimization_target")
    valid = {
        "ReadLatency", "WriteLatency", "ReadDynamicEnergy",
        "WriteDynamicEnergy", "ReadEDP", "WriteEDP", "LeakagePower", "Area",
    }
    if explicit is not None:
        require(str(explicit) in valid,
                f"unsupported DESTINY optimization target: {explicit}")
        return str(explicit)
    prefix = "Read" if read_fraction >= 0.5 else "Write"
    latency_weight = canonical["weights"].get("average_latency_ns", 0.0)
    power_weight = (
        canonical["weights"].get("average_power_mw", 0.0)
        + canonical["weights"].get(
            "expected_dynamic_energy_nj_per_request", 0.0
        )
    )
    if latency_weight > 1.5 * power_weight:
        return prefix + "Latency"
    if power_weight > 1.5 * latency_weight:
        return prefix + "DynamicEnergy"
    return prefix + "EDP"


def _set_config_field(text: str, field: str, value: Any) -> str:
    pattern = rf"^{re.escape(field)}:\s*.*$"
    require(re.search(pattern, text, flags=re.MULTILINE) is not None,
            f"DESTINY base config missing {field}")
    return re.sub(pattern, f"{field}: {value}", text, flags=re.MULTILINE)


def _generate_destiny_config(
    repo_root: Path,
    layer: Dict[str, Any],
    entry: Dict[str, Any],
    target: str,
) -> Path:
    base_path = _resolve_path(repo_root, str(entry["destiny_cfg"]))
    require(base_path.is_file(), f"DESTINY device config not found: {base_path}")
    text = base_path.read_text(encoding="utf-8")
    cell_match = re.search(r"^-MemoryCellInputFile:\s*(\S+)\s*$", text, re.MULTILINE)
    require(cell_match is not None,
            f"{base_path.name}: missing -MemoryCellInputFile")
    text = _set_config_field(
        text, "-MemoryCellInputFile", f"../devices/{cell_match.group(1)}"
    )
    text = _set_config_field(
        text, "-Capacity (KB)",
        int(layer.get("destiny_model_capacity_bytes", layer["capacity_bytes"])) // 1024
    )
    text = _set_config_field(
        text, "-Associativity (for cache only)", int(layer["associativity"])
    )
    text = _set_config_field(
        text, "-WordWidth (bit)", int(layer["line_bytes"]) * 8
    )
    text = _set_config_field(text, "-OptimizationTarget", target)
    text = _set_config_field(text, "-PrintLevel", 1)
    text = _set_config_field(
        text, "-StackedDieCount",
        1 if bool(layer.get("m3d", {}).get("enabled", False))
        else int(layer.get("stacked_tiers", 1))
    )
    array_limits = layer.get("array_limits", {})
    require(isinstance(array_limits, dict),
            f"{layer['name']}: array_limits must be an object")
    if "max_subarray_rows" in array_limits:
        text += (
            f"\n-MaxPhysicalSubarrayRows: "
            f"{int(array_limits['max_subarray_rows'])}\n"
        )
    if "max_subarray_columns" in array_limits:
        text += (
            f"-MaxPhysicalSubarrayColumns: "
            f"{int(array_limits['max_subarray_columns'])}\n"
        )
    generated_dir = repo_root / "config" / ".scope-cache"
    generated_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", str(layer["device"]).lower()).strip("_")
    filename = (
        f"{str(layer['name']).lower()}_{slug}_"
        f"{int(layer.get('destiny_model_capacity_bytes', layer['capacity_bytes'])) // 1024}k_"
        f"{int(layer['associativity'])}w_{target.lower()}.cfg"
    )
    output = generated_dir / filename
    if not output.exists() or output.read_text(encoding="utf-8") != text:
        output.write_text(text, encoding="utf-8")
    return output


def _effective_density(entry: Dict[str, Any], stacked_tiers: int) -> float:
    density = float(entry["density_f2"])
    if "density_divisor" in entry:
        density /= stacked_tiers
    require(density > 0.0, "effective device density must be positive")
    return density


def _capacity_for_device(
    raw: Dict[str, Any], layer: Dict[str, Any], entry: Dict[str, Any],
    stacked_tiers: int,
) -> Tuple[int, int, float, int]:
    capacity_cfg = raw.get("capacity", {})
    require(isinstance(capacity_cfg, dict), "capacity must be an object")
    mode = str(layer.get("capacity_mode", capacity_cfg.get("mode", "fixed")))
    if mode == "fixed":
        capacity = int(layer["capacity_bytes"])
        baseline = capacity
        scale = 1.0
        ideal_capacity = capacity
    else:
        require(mode == "density_scaled",
                "capacity mode must be fixed or density_scaled")
        baselines = capacity_cfg.get("sram_baseline_bytes", {})
        require(isinstance(baselines, dict) and layer["name"] in baselines,
                f"missing SRAM baseline capacity for {layer['name']}")
        baseline = int(baselines[layer["name"]])
        reference_density = float(capacity_cfg.get("sram_density_f2", 140.0))
        scale = reference_density / _effective_density(entry, stacked_tiers)
        capacity = int(round(baseline * scale))
        ideal_capacity = capacity
        if bool(capacity_cfg.get("destiny_power_of_two", True)):
            capacity = 1 << int(round(math.log2(capacity)))
    quantum = math.lcm(1024, int(layer["line_bytes"]) * int(layer["associativity"]))
    capacity = max(quantum, int(round(capacity / quantum)) * quantum)
    return capacity, baseline, scale, ideal_capacity


def design_variants(
    raw: Dict[str, Any], explore: bool, repo_root: Path,
    device_library: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    layers = raw.get("layers")
    require(isinstance(layers, list) and len(layers) == 3,
            "config must contain exactly three layers")
    automatic = raw.get("auto_selection", {})
    require(isinstance(automatic, dict), "auto_selection must be an object")
    auto_enabled = bool(automatic.get("enabled", False))
    feature_flags = resolve_feature_switches(raw)
    all_devices = automatic.get("devices", list(device_library))
    require(isinstance(all_devices, list) and all_devices,
            "auto_selection.devices must be a non-empty list")
    target = _guided_destiny_target(
        raw.get("guidance", {}),
        float(raw["workload"].get("read_fraction_hint", 0.8)),
    )
    choices: List[List[Dict[str, Any]]] = []
    for layer in layers:
        require(isinstance(layer, dict), "each layer must be an object")
        device_overrides = layer.get("device_overrides", {})
        require(isinstance(device_overrides, dict),
                f"{layer.get('name', 'layer')}: device_overrides must be an object")
        base = {
            key: value for key, value in layer.items()
            if key not in {"candidates", "devices", "device_overrides"}
        }
        if auto_enabled:
            devices = layer.get("devices", all_devices)
            require(isinstance(devices, list) and devices,
                    f"{base.get('name', 'layer')}: devices must be non-empty")
            candidates = [{"device": device} for device in devices]
        else:
            candidates = layer.get("candidates", [{}]) if explore else [{}]
        require(isinstance(candidates, list) and candidates,
                f"{base.get('name', 'layer')}: candidates must be a non-empty list")
        expanded: List[Dict[str, Any]] = []
        for candidate in candidates:
            require(isinstance(candidate, dict), "each candidate must be an object")
            merged = copy.deepcopy(base)
            merged.update(copy.deepcopy(candidate))
            if auto_enabled:
                device = str(merged["device"])
                require(device in device_library,
                        f"unknown automatic device: {device}")
                override = device_overrides.get(device, {})
                require(isinstance(override, dict),
                        f"{base.get('name', 'layer')}: invalid override for {device}")
                merged.update(copy.deepcopy(override))
                entry = device_library[device]
                stacked_tiers = int(merged.get(
                    "stacked_tiers", 4 if "density_divisor" in entry else 1
                ))
                capacity, baseline, scale, ideal_capacity = _capacity_for_device(
                    raw, merged, entry, stacked_tiers
                )
                merged["stacked_tiers"] = stacked_tiers
                merged["capacity_bytes"] = capacity
                if bool(raw.get("capacity", {}).get(
                    "destiny_proxy_power_of_two", False
                )) and capacity & (capacity - 1):
                    merged["destiny_model_capacity_bytes"] = \
                        1 << int(round(math.log2(capacity)))
                else:
                    merged["destiny_model_capacity_bytes"] = capacity
                merged["sram_baseline_capacity_bytes"] = baseline
                merged["density_capacity_scale"] = scale
                merged["ideal_density_capacity_bytes"] = ideal_capacity
                merged["actual_capacity_scale"] = capacity / baseline
                if str(entry["family"]) == "eDRAM":
                    merged.setdefault(
                        "refresh_interval_us",
                        float(automatic.get("refresh_interval_us", 10.0)),
                    )
                    merged.setdefault(
                        "retention_time_us",
                        float(automatic.get("retention_time_us", 10.0)),
                    )
                if entry["endurance"].get("writes_per_line") is None:
                    merged.setdefault(
                        "bti_endurance_writes_per_line",
                        float(automatic["bti_endurance_writes_per_line"]),
                    )
                generated = _generate_destiny_config(
                    repo_root, merged, entry, target
                )
                merged["destiny_config"] = str(generated.relative_to(repo_root))
                merged["destiny_optimization_target"] = target
            entry = device_library[str(merged["device"])]
            interface = entry.get("sense_amplifier", {})
            native_sa = str(interface.get("destiny_native_type", "voltage"))
            default_sa = str(interface.get("default_type", native_sa))
            if not feature_flags["configurable_peripherals"]:
                sa_choices = [native_sa]
            elif explore and interface:
                sa_choices = merged.get(
                    "sense_amp_types", interface.get("supported_types", [default_sa])
                )
            else:
                sa_choices = [merged.get("sense_amp_type", default_sa)]
            require(isinstance(sa_choices, list) and sa_choices,
                    f"{merged['name']}: sense_amp_types must be non-empty")
            for sense_amp_type in sa_choices:
                sa_variant = copy.deepcopy(merged)
                sa_variant["sense_amp_type"] = str(sense_amp_type)
                expanded.append(sa_variant)
        choices.append(expanded)

    variants: List[Dict[str, Any]] = []
    for combination in itertools.product(*choices):
        variant = copy.deepcopy(raw)
        variant["layers"] = list(combination)
        variants.append(variant)
    return variants


def evaluate_config(raw: Dict[str, Any], repo_root: Path, runner: DestinyRunner,
                    auto_build: bool,
                    device_library: Dict[str, Dict[str, Any]],
                    model_library: Optional[Dict[str, Any]] = None
                    ) -> Tuple[ScopeModel, Dict[str, Any]]:
    runner.ensure_built(auto_build=auto_build)
    feature_flags = resolve_feature_switches(raw)
    global_ber_max = float(raw.get("constraints", {}).get("ber_max", 1.0))
    layer_specs: List[LayerSpec] = []
    for layer_raw in raw["layers"]:
        cfg_path = _resolve_path(repo_root, str(layer_raw["destiny_config"]))
        metrics = runner.run(cfg_path)
        layer_specs.append(build_layer(
            layer_raw, metrics, repo_root, global_ber_max, device_library,
            model_library, feature_flags,
        ))
    require([layer.name for layer in layer_specs] == ["L1", "L2", "L3"],
            "layers must be ordered and named L1, L2, L3")
    crossbars_raw = raw.get("crossbars")
    require(isinstance(crossbars_raw, list) and len(crossbars_raw) == 2,
            "config must contain exactly two crossbars")
    crossbars = [build_crossbar(item) for item in crossbars_raw]
    off_raw = raw["off_chip"]
    transaction_bytes = int(off_raw.get("transaction_bytes", 64))
    energy_pj_per_bit = float(off_raw.get("energy_pj_per_bit", 0.0))
    energy_nj = float(off_raw.get(
        "energy_nj", energy_pj_per_bit * transaction_bytes * 8 / 1000.0
    ))
    offchip = OffChipSpec(
        latency_ns=float(off_raw["latency_ns"]),
        energy_nj=energy_nj,
        standard=str(off_raw.get("standard", "unspecified")),
        bandwidth_gbps=float(off_raw.get("bandwidth_gbps", 0.0)),
        bus_width_bits=int(off_raw.get("bus_width_bits", 0)),
        data_rate_mtps=float(off_raw.get("data_rate_mtps", 0.0)),
        transaction_bytes=transaction_bytes,
        energy_pj_per_bit=energy_pj_per_bit,
        timing_basis=str(off_raw.get("timing_basis", "")),
    )
    require(offchip.latency_ns >= 0.0 and offchip.energy_nj >= 0.0,
            "off-chip latency and energy must be non-negative")
    workload = dict(raw["workload"])
    hit_rates = estimate_hit_rates(layer_specs, workload, repo_root)
    workload["read_fraction"] = hit_rates.observed_read_fraction
    if hit_rates.trace_metadata is not None:
        workload["memory_access_rate_per_s"] = float(
            hit_rates.trace_metadata["memory_access_rate_per_s"]
        )
    model = ScopeModel(
        layers=layer_specs,
        crossbars=crossbars,
        offchip=offchip,
        workload=workload,
        hit_rates=hit_rates,
        refill_on_critical_path=bool(raw.get("refill_on_critical_path", False)),
        core_to_l1_latency_ns=float(raw.get("core_to_l1_latency_ns", 0.0)),
    )
    report = model.average()
    report["name"] = str(raw.get("name", "SCOPE"))
    report["selected_workload"] = str(raw.get("selected_workload", "custom"))
    report["features"] = feature_flags
    guidance = guidance_score(report, raw.get("guidance", {}))
    report["guidance"] = guidance
    report["guidance_score"] = guidance["score"]
    for check in guidance["limit_checks"]:
        report["constraints"].append({
            "layer": "SYSTEM",
            "constraint": f"Guidance limit: {check['metric']}",
            "value": check["value"],
            "limit": check["limit"],
            "pass": check["pass"],
        })
    report["feasible"] = all(item["pass"] for item in report["constraints"])
    report["layers"] = [
        {
            "name": layer.name,
            "device": layer.device,
            "device_family": layer.device_family,
            "destiny_config": str(layer.destiny_config.relative_to(repo_root)),
            "capacity_bytes": layer.capacity_bytes,
            "destiny_model_capacity_bytes": layer.destiny_model_capacity_bytes,
            "associativity": layer.associativity,
            "line_bytes": layer.line_bytes,
            "replacement_policy": layer.replacement_policy,
            "banks": layer.banks,
            "effective_ber": layer.ber,
            "ber_limit": layer.ber_max,
            "sram_baseline_capacity_bytes": layer_raw.get(
                "sram_baseline_capacity_bytes", layer.capacity_bytes
            ),
            "density_capacity_scale": layer_raw.get("density_capacity_scale", 1.0),
            "ideal_density_capacity_bytes": layer_raw.get(
                "ideal_density_capacity_bytes", layer.capacity_bytes
            ),
            "actual_capacity_scale": layer_raw.get("actual_capacity_scale", 1.0),
            "destiny_optimization_target": layer_raw.get(
                "destiny_optimization_target", layer.raw_metrics.optimization_target
            ),
            "device_rows_per_bank": layer.device_rows_per_bank,
            "device_library_values": layer.device_library_entry,
            "device_derived": {
                "leakage_power_mw": layer.device_leakage_power_mw,
                "refresh_power_mw": layer.device_refresh_power_mw,
                "endurance_writes_per_line": layer.endurance_writes_per_line,
                "stacked_tiers": layer.stacked_tiers,
                "effective_density_f2": layer.effective_density_f2,
                "data_cell_area_f2": layer.data_cell_area_f2
            },
            "edram_read_equation": layer.edram_read,
            "refresh_equation": layer.refresh,
            "bti_retention": layer.bti,
            "sense_amplifier": layer.sense_amp,
            "m3d": layer.m3d,
            "array_nonideal": layer.nonideal,
            "static_power_components": layer.static_power_components,
            "geometry_audit": geometry_audit(layer),
            "latency_audit": {
                "raw_destiny_hit_latency_ns": layer.raw_metrics.hit_latency_ns,
                "raw_destiny_rbl_delay_ns": layer.raw_metrics.rbl_delay_ns,
                "destiny_peripheral_read_latency_ns":
                    layer.raw_metrics.peripheral_read_latency_ns,
                "equation_cell_read_latency_ns": (
                    layer.edram_read or {}
                ).get("cell_read_latency_ns"),
                "sense_amp_delta_ns": (
                    layer.sense_amp or {}
                ).get("hit_latency_delta_ns", 0.0),
                "array_nonideal_penalty_ns": (
                    layer.nonideal or {}
                ).get("read_latency_penalty_ns", 0.0),
                "m3d_penalty_ns": (
                    layer.m3d or {}
                ).get("latency_penalty_ns", 0.0),
                "bti_read_latency_guardband": (
                    layer.bti or {}
                ).get("read_latency_guardband", 1.0),
                "effective_hit_latency_ns": layer.metrics.hit_latency_ns,
                "interpretation": (
                    "For equation-based eDRAM, SCOPE replaces only DESTINY's "
                    "selected RBL delay with C_RBL*deltaV/Ion and retains "
                    "peripheral/tag timing, then applies BTI, SA, array-nonideal, "
                    "and M3D deltas."
                ),
            },
            "raw_destiny_metrics": asdict(layer.raw_metrics),
            "effective_metrics": asdict(layer.metrics),
            "combination_rule": (
                "DESTINY selects bank/mat/subarray geometry and supplies peripheral/tag "
                "terms. eDRAM replaces the selected RBL term with C_RBL*deltaV/Ion and "
                "0.5*C_RBL*(Vdd^2-(Vdd-deltaV)^2); Si-eDRAM additionally uses row "
                "read+write refresh power and bandwidth occupancy. OSFET-eDRAM derives "
                "a BTI retention limit from measured Delta-Vth(t), then uses the same "
                "row maintenance equation. Other values come from the Device Library."
            ),
        }
        for layer, layer_raw in zip(layer_specs, raw["layers"])
    ]
    report["crossbars"] = [
        {
            "name": item.name,
            "latency_ns": item.latency_ns,
            "energy_nj": item.energy_nj,
            "request_cycles": item.request_cycles,
            "response_cycles": item.response_cycles,
            "clock_ghz": item.clock_ghz,
            "energy_pj_per_bit": item.energy_pj_per_bit,
            "transaction_bits": item.transaction_bits,
            "topology": item.topology,
            "hops": item.hops,
            "router_pipeline_cycles": item.router_pipeline_cycles,
            "link_traversal_cycles": item.link_traversal_cycles,
            "request_bits": item.request_bits,
            "response_bits": item.response_bits,
            "request_latency_ns": item.request_latency_ns,
            "response_latency_ns": item.response_latency_ns,
            "router_energy_pj_per_bit": item.router_energy_pj_per_bit,
            "link_energy_pj_per_bit_per_mm": item.link_energy_pj_per_bit_per_mm,
            "link_length_mm": item.link_length_mm,
            "model_basis": (
                "BookSim-style hop/cycle latency plus ORION-style router and "
                "wire dynamic-energy decomposition; no queueing is modeled in "
                "this single-request behavior demo."
            ),
        }
        for item in crossbars
    ]
    report["hit_rate_model"] = {
        "mode": workload.get("hit_rate_model", {}).get("mode", "synthetic"),
        "parameters": workload.get("hit_rate_model", {}),
        "observed_read_fraction": hit_rates.observed_read_fraction,
        "writebacks_per_request": list(hit_rates.writebacks_per_request),
        "offchip_writebacks_per_request": hit_rates.offchip_writebacks_per_request,
        "trace_metadata": hit_rates.trace_metadata,
    }
    trace = hit_rates.trace_metadata or {}
    compulsory = trace.get("compulsory_misses", [0] * len(layer_specs))
    noncompulsory = trace.get("noncompulsory_misses", [0] * len(layer_specs))
    report["hit_rate_model"]["per_level_explanation"] = [
        {
            "layer": layer.name,
            "conditional_accesses": hit_rates.accesses[index],
            "conditional_hits": hit_rates.hits[index],
            "conditional_hit_rate": hit_rates.hit_rates[index],
            "compulsory_misses": int(compulsory[index]),
            "noncompulsory_misses": int(noncompulsory[index]),
            "reason": (
                "This is a conditional hit rate after all upper-level hits were "
                "filtered. A low middle-level rate can therefore mean the remaining "
                "stream is dominated by first touches or conflicts, not that its "
                "absolute capacity is ineffective."
            ),
        }
        for index, layer in enumerate(layer_specs)
    ]
    report["workload"] = workload
    report["off_chip"] = asdict(offchip)
    return model, report


def _exploration_record(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "devices": [layer["device"] for layer in report["layers"]],
        "sense_amplifiers": [
            (layer.get("sense_amplifier") or {}).get("selected_type")
            for layer in report["layers"]
        ],
        "average_latency_ns": report["average_latency_ns"],
        "average_power_mw": report["average_power_mw"],
        "hit_rates": report["hit_rates"],
        "offchip_reach_probability": report["offchip_reach_probability"],
        "fom_per_ns_mw": report["fom_per_ns_mw"],
        "guidance_score": report["guidance_score"],
        "capacities_bytes": [
            layer["capacity_bytes"] for layer in report["layers"]
        ],
        "effective_ber": [
            layer["effective_ber"] for layer in report["layers"]
        ],
        "destiny_optimization_targets": [
            layer["destiny_optimization_target"] for layer in report["layers"]
        ],
        "feasible": report["feasible"],
    }


def evaluate_design(
    raw: Dict[str, Any], repo_root: Path, runner: DestinyRunner, *,
    auto_build: bool, device_library: Dict[str, Dict[str, Any]],
    model_library: Dict[str, Any], library_path: Path, explore: bool,
    instruction_override: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[ScopeModel, Dict[str, Any]]:
    """Evaluate and select the best feasible circuit variant for one design."""
    variants = design_variants(
        raw, explore=explore, repo_root=repo_root,
        device_library=device_library,
    )
    evaluations = [
        evaluate_config(
            variant, repo_root, runner, auto_build=auto_build,
            device_library=device_library, model_library=model_library,
        )
        for variant in variants
    ]
    feasible = [item for item in evaluations if item[1]["feasible"]]
    pool = feasible if feasible else evaluations
    model, report = max(pool, key=lambda item: item[1]["guidance_score"])
    report["exploration"] = {
        "evaluated_designs": len(evaluations),
        "feasible_designs": len(feasible),
        "selected_highest_feasible_fom": bool(feasible),
        "designs": [_exploration_record(item[1]) for item in evaluations],
    }
    report["device_library"] = {
        "path": str(library_path.relative_to(repo_root)),
        "schema_version": model_library.get("schema_version"),
        "source": model_library.get("source"),
        "semantics": model_library.get("semantics"),
    }
    report["auto_selection"] = raw.get("auto_selection", {"enabled": False})
    cases = list(instruction_override) if instruction_override is not None else \
        raw.get("instruction_cases", [])
    report["instructions"] = [
        model.instruction(str(case["op"]), str(case["hit_level"]))
        for case in cases
    ]
    return model, report


def build_evaluation_cases(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand v7 architecture comparisons and one-at-a-time ablations."""
    suite = raw.get("evaluation_suite")
    require(isinstance(suite, dict),
            "evaluation_suite must be provided for --compare")
    architectures = suite.get("architectures")
    require(isinstance(architectures, list) and architectures,
            "evaluation_suite.architectures must be a non-empty list")
    names: set[str] = set()
    cases: List[Dict[str, Any]] = []
    base_features = resolve_feature_switches(raw)
    for architecture in architectures:
        require(isinstance(architecture, dict),
                "each evaluation architecture must be an object")
        name = str(architecture.get("name", ""))
        devices = architecture.get("devices")
        require(name and name not in names,
                "evaluation architecture names must be unique and non-empty")
        require(isinstance(devices, list) and len(devices) == 3,
                f"evaluation architecture {name} must list three devices")
        names.add(name)
        case_raw = copy.deepcopy(raw)
        case_raw["name"] = f"{raw.get('name', 'SCOPE')} / {name}"
        case_raw["features"] = dict(base_features)
        for layer, device in zip(case_raw["layers"], devices):
            layer["devices"] = [str(device)]
        cases.append({
            "id": name,
            "group": "architecture",
            "architecture": name,
            "config": case_raw,
        })

    optimized = str(suite.get("optimized_architecture", "optimized"))
    require(optimized in names,
            "evaluation_suite.optimized_architecture is not defined")
    optimized_devices = next(
        item["devices"] for item in architectures
        if str(item["name"]) == optimized
    )
    ablations = suite.get(
        "feature_ablations", list(FEATURE_SWITCH_DEFAULTS)
    )
    require(isinstance(ablations, list) and ablations,
            "evaluation_suite.feature_ablations must be a non-empty list")
    for feature in ablations:
        feature_name = str(feature)
        require(feature_name in FEATURE_SWITCH_DEFAULTS,
                f"unknown feature ablation: {feature_name}")
        case_raw = copy.deepcopy(raw)
        case_raw["name"] = (
            f"{raw.get('name', 'SCOPE')} / {optimized} / {feature_name}=off"
        )
        case_raw["features"] = dict(base_features)
        case_raw["features"][feature_name] = False
        for layer, device in zip(case_raw["layers"], optimized_devices):
            layer["devices"] = [str(device)]
        cases.append({
            "id": f"{optimized}_{feature_name}_off",
            "group": "feature_ablation",
            "architecture": optimized,
            "disabled_feature": feature_name,
            "config": case_raw,
        })
    return cases


def comparison_summary(case_id: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the architecture comparison surface compact and machine-readable."""
    return {
        "case": case_id,
        "features": report["features"],
        "devices": [layer["device"] for layer in report["layers"]],
        "sense_amplifiers": [
            (layer.get("sense_amplifier") or {}).get("selected_type")
            for layer in report["layers"]
        ],
        "average_latency_ns": report["average_latency_ns"],
        "average_power_mw": report["average_power_mw"],
        "fom_per_ns_mw": report["fom_per_ns_mw"],
        "capacities_bytes": [
            layer["capacity_bytes"] for layer in report["layers"]
        ],
        "conditional_hit_rates": list(report["hit_rates"]),
        "effective_ber": [
            layer["effective_ber"] for layer in report["layers"]
        ],
        "offchip_reach_probability": report["offchip_reach_probability"],
        "refresh_power_mw": report["refresh_power_mw"],
        "refresh_bandwidth_occupancy": [
            float((layer.get("refresh_equation") or {}).get(
                "bandwidth_occupancy", 0.0
            ))
            for layer in report["layers"]
        ],
        "bti_retention_s": [
            (layer.get("bti_retention") or {}).get("equivalent_retention_s")
            for layer in report["layers"]
        ],
        "feasible": report["feasible"],
    }


def evaluate_comparison_suite(
    raw: Dict[str, Any], repo_root: Path, runner: DestinyRunner, *,
    auto_build: bool, device_library: Dict[str, Dict[str, Any]],
    model_library: Dict[str, Any], library_path: Path,
) -> Dict[str, Any]:
    """Run architecture and feature-ablation comparisons."""
    case_specs = build_evaluation_cases(raw)
    case_reports: Dict[str, Dict[str, Any]] = {}
    architecture_summaries: List[Dict[str, Any]] = []
    for case in case_specs:
        _, report = evaluate_design(
            case["config"], repo_root, runner,
            auto_build=auto_build, device_library=device_library,
            model_library=model_library, library_path=library_path,
            explore=True,
        )
        case_reports[case["id"]] = report
        summary = comparison_summary(case["id"], report)
        if case["group"] == "architecture":
            architecture_summaries.append(summary)

    suite = raw["evaluation_suite"]
    optimized = str(suite.get("optimized_architecture", "optimized"))
    baseline = comparison_summary(optimized, case_reports[optimized])
    feature_summaries = [dict(baseline, comparison="all_features_on")]
    for case in case_specs:
        if case["group"] != "feature_ablation":
            continue
        summary = comparison_summary(case["id"], case_reports[case["id"]])
        summary["comparison"] = f"{case['disabled_feature']}_off"
        summary["delta_vs_all_features_on"] = {
            "average_latency_ns": (
                summary["average_latency_ns"] - baseline["average_latency_ns"]
            ),
            "average_power_mw": (
                summary["average_power_mw"] - baseline["average_power_mw"]
            ),
            "fom_per_ns_mw": (
                summary["fom_per_ns_mw"] - baseline["fom_per_ns_mw"]
            ),
        }
        feature_summaries.append(summary)

    feasible_architectures = [
        item for item in architecture_summaries if item["feasible"]
    ]
    ranking_pool = feasible_architectures or architecture_summaries
    best = max(ranking_pool, key=lambda item: item["fom_per_ns_mw"])
    return {
        "schema_version": int(raw.get("schema_version", 7)),
        "name": str(raw.get("name", "SCOPE evaluation suite")),
        "selected_workload": str(raw.get("selected_workload", "custom")),
        "comparison_method": (
            "same OpenVLA trace, cache policy, line size, NoC and LPDDR model; "
            "feature ablations change one overlay at a time"
        ),
        "optimized_architecture": optimized,
        "best_evaluated_architecture": best["case"],
        "architecture_comparison": architecture_summaries,
        "feature_comparison": feature_summaries,
        "case_reports": case_reports,
    }


def print_comparison(report: Dict[str, Any]) -> None:
    print(
        f"SCOPE comparison [workload={report['selected_workload']}]"
    )
    print("=" * 120)
    print("Architecture comparison:")
    for item in report["architecture_comparison"]:
        capacities = "/".join(str(value // 1024) for value in item["capacities_bytes"])
        hits = "/".join(f"{value:.4f}" for value in item["conditional_hit_rates"])
        ber = "/".join(f"{value:.2e}" for value in item["effective_ber"])
        print(
            f"  {item['case']:<18} devices={'/'.join(item['devices']):<44} "
            f"lat={item['average_latency_ns']:.6f} ns  "
            f"power={item['average_power_mw']:.6f} mW  "
            f"FoM={item['fom_per_ns_mw']:.9g}  "
            f"refresh={item['refresh_power_mw']:.6g} mW  "
            f"capacity(KiB)={capacities}  hit={hits}  BER={ber}  "
            f"feasible={str(item['feasible']).lower()}"
        )
    print("Feature ablation on the optimized architecture:")
    for item in report["feature_comparison"]:
        print(
            f"  {item['comparison']:<30} "
            f"lat={item['average_latency_ns']:.6f} ns  "
            f"power={item['average_power_mw']:.6f} mW  "
            f"FoM={item['fom_per_ns_mw']:.9g}"
        )


def print_report(report: Dict[str, Any], instructions: Sequence[Dict[str, Any]]) -> None:
    print(
        f"SCOPE evaluation: {report['name']} "
        f"[workload={report.get('selected_workload', 'custom')}]"
    )
    print("=" * 72)
    print("Layers (one DESTINY instance each):")
    for index, layer in enumerate(report["layers"]):
        metrics = layer["effective_metrics"]
        print(
            f"  {layer['name']}: {layer['device']}, {layer['capacity_bytes'] // 1024} KiB, "
            f"{layer['associativity']}-way, R={report['hit_rates'][index]:.6f}, "
            f"read/write={metrics['hit_latency_ns']:.3f}/{metrics['write_latency_ns']:.3f} ns"
        )
        print(
            f"      DESTINY {layer['destiny_optimization_target']}; "
            f"bank={metrics['bank_organization'] or 'n/a'}, "
            f"mat={metrics['mat_organization'] or 'n/a'}, "
            f"subarray={metrics['subarray_size'] or 'n/a'}"
        )
    print("\nAverage latency breakdown (provided FoM equation):")
    for item in report["latency_breakdown"]:
        print(
            f"  {item['component']:<30} reach={item['reach_probability']:.6f} "
            f"weighted={item['weighted_latency_ns']:.6f} ns"
        )
    print(f"  {'TOTAL':<30} {report['average_latency_ns']:.6f} ns")

    print("\nAverage power breakdown:")
    for item in report["dynamic_power_breakdown"]:
        print(f"  dynamic {item['component']:<22} {item['power_mw']:.9g} mW")
    for item in report["static_power_breakdown"]:
        print(f"  static  {item['component']:<22} {item['power_mw']:.9g} mW")
    for item in report["refresh_power_breakdown"]:
        print(f"  refresh {item['component']:<22} {item['power_mw']:.9g} mW")
    print(f"  {'DYNAMIC TOTAL':<31} {report['dynamic_power_mw']:.9g} mW")
    print(f"  {'STATIC TOTAL':<31} {report['static_power_mw']:.9g} mW")
    print(f"  {'REFRESH TOTAL':<31} {report['refresh_power_mw']:.9g} mW")
    print(f"  {'POWER TOTAL':<31} {report['average_power_mw']:.9g} mW")
    print(f"\nClassic latency*power FoM = {report['fom_per_ns_mw']:.12g} 1/(ns*mW)")
    print(
        f"Guidance '{report['guidance']['name']}' score = "
        f"{report['guidance_score']:.12g}"
    )
    print(f"Feasible = {str(report['feasible']).lower()}")
    print("Constraints:")
    for item in report["constraints"]:
        value = "n/a" if item["value"] is None else f"{item['value']:.6g}"
        limit = "n/a" if item["limit"] is None else f"{item['limit']:.6g}"
        print(
            f"  {item['layer']} {item['constraint']:<20} value={value:<12} "
            f"limit={limit:<12} {'PASS' if item['pass'] else 'FAIL'}"
        )

    for instruction in instructions:
        print("\n" + "-" * 72)
        print(
            f"Instruction: {instruction['op']} (data from {instruction['hit_level']})\n"
            f"Policy: {instruction['write_policy']}"
        )
        for item in instruction["breakdown"]:
            critical = "critical" if item["on_critical_path"] else "off-path"
            print(
                f"  {item['component']:<28} {item['latency_ns']:.6f} ns, "
                f"{item['energy_nj']:.6f} nJ ({critical})"
            )
        print(f"  latency total              {instruction['latency_ns']:.6f} ns")
        print(f"  dynamic energy total       {instruction['dynamic_energy_nj']:.6f} nJ")
        print(f"  serialized dynamic power   {instruction['serialized_dynamic_power_mw']:.6f} mW")
        print(f"  static power               {instruction['static_power_mw']:.9g} mW")
        print(f"  refresh power              {instruction['refresh_power_mw']:.9g} mW")
        print(f"  serialized total power     {instruction['serialized_total_power_mw']:.9g} mW")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose three DESTINY cache instances into the SCOPE hierarchy"
    )
    parser.add_argument("config", type=Path, help="SCOPE JSON configuration")
    parser.add_argument("--op", choices=("load", "store"),
                        help="evaluate one explicit instruction path")
    parser.add_argument("--hit-level", choices=("L1", "L2", "L3", "OFF", "MEM"),
                        help="where the explicit instruction obtains its cache line")
    parser.add_argument("--explore", action="store_true",
                        help="evaluate the Cartesian product of per-layer candidates")
    parser.add_argument("--compare", action="store_true",
                        help="run the configured v7 architecture and feature suite")
    parser.add_argument("--workload",
                        help="select a named workload profile from config.workloads")
    parser.add_argument("--json-output", type=Path,
                        help="also write the complete machine-readable report")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="timeout for each DESTINY instance (seconds)")
    parser.add_argument("--no-build", action="store_true",
                        help="fail instead of building a missing DESTINY binary")
    args = parser.parse_args(argv)
    if bool(args.op) != bool(args.hit_level):
        parser.error("--op and --hit-level must be provided together")
    if args.compare and (args.op or args.explore):
        parser.error("--compare cannot be combined with --op/--hit-level or --explore")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    try:
        raw = select_workload(load_json(config_path.resolve()), args.workload)
        library_path = _resolve_path(
            repo_root, str(raw.get("device_library", "config/device_library.json"))
        )
        library_raw = load_model_library(library_path, repo_root)
        device_library = library_raw.get("devices")
        require(isinstance(device_library, dict) and device_library,
                "device library must contain a non-empty devices object")
        binary = _resolve_path(repo_root, str(raw.get("destiny_binary", "destiny")))
        runner = DestinyRunner(repo_root, binary, timeout_s=args.timeout)
        if args.compare:
            report = evaluate_comparison_suite(
                raw, repo_root, runner, auto_build=not args.no_build,
                device_library=device_library, model_library=library_raw,
                library_path=library_path,
            )
            print_comparison(report)
        else:
            instruction_override = (
                [{"op": args.op, "hit_level": args.hit_level}]
                if args.op else None
            )
            _, report = evaluate_design(
                raw, repo_root, runner, auto_build=not args.no_build,
                device_library=device_library, model_library=library_raw,
                library_path=library_path, explore=args.explore,
                instruction_override=instruction_override,
            )
            print_report(report, report["instructions"])
        if args.json_output:
            output = args.json_output if args.json_output.is_absolute() else \
                Path.cwd() / args.json_output
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
        return 0
    except (KeyError, TypeError, ValueError, ScopeError) as exc:
        print(f"SCOPE error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
