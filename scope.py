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


FLOAT_RE = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
VALID_POLICIES = {"LRU", "FIFO", "RANDOM"}


class ScopeError(RuntimeError):
    """Raised for invalid configurations or failed DESTINY runs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScopeError(message)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


@dataclass(frozen=True)
class LayerSpec:
    name: str
    device: str
    device_family: str
    destiny_config: Path
    capacity_bytes: int
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

    @property
    def latency_ns(self) -> float:
        return (self.request_cycles + self.response_cycles) / self.clock_ghz

    @property
    def energy_nj(self) -> float:
        return self.energy_pj_per_bit * self.transaction_bits / 1000.0


@dataclass(frozen=True)
class OffChipSpec:
    latency_ns: float
    energy_nj: float


@dataclass(frozen=True)
class HitRateResult:
    hit_rates: Tuple[float, ...]
    accesses: Tuple[int, ...]
    hits: Tuple[int, ...]
    writebacks_per_request: Tuple[float, ...]
    offchip_writebacks_per_request: float
    observed_read_fraction: float


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
    else:
        raise ScopeError(f"unsupported refresh model: {refresh_model}")

    leakage_model = str(leakage["model"])
    if leakage_model == "constant":
        leakage_mw = float(leakage["value_pw_per_bit"]) * capacity_bits * 1e-9
    elif leakage_model == "capacitor_refresh":
        leakage_mw = capacitor_power_mw(leakage)
    elif leakage_model == "same_as_refresh":
        leakage_mw = refresh_mw
    else:
        raise ScopeError(f"unsupported leakage model: {leakage_model}")
    return leakage_mw, refresh_mw


def _apply_device_library(
    raw_metrics: DestinyMetrics,
    entry: Dict[str, Any],
    capacity_bytes: int,
    line_bytes: int,
    refresh_interval_us: float,
) -> Tuple[DestinyMetrics, float, float]:
    """Apply screenshot values as device floors over DESTINY hierarchy results."""
    line_bits = line_bytes * 8

    def latency_floor(field: str, baseline: float) -> float:
        model = entry[field]
        if model["model"] == "constant":
            return max(baseline, float(model["value_ns"]))
        if str(model["model"]).startswith("destiny_"):
            return baseline
        raise ScopeError(f"unsupported {field} model: {model['model']}")

    def energy_floor(field: str, baseline: float) -> float:
        model = entry[field]
        if model["model"] == "constant":
            device_line_nj = float(model["value_fj_per_bit"]) * line_bits / 1e6
            return max(baseline, device_line_nj)
        if str(model["model"]).startswith("destiny_"):
            return baseline
        raise ScopeError(f"unsupported {field} model: {model['model']}")

    device_leakage_mw, device_refresh_mw = _power_from_device_library(
        entry, capacity_bytes * 8, refresh_interval_us
    )
    effective = DestinyMetrics(
        capacity_bytes=raw_metrics.capacity_bytes,
        associativity=raw_metrics.associativity,
        line_bytes=raw_metrics.line_bytes,
        hit_latency_ns=latency_floor("read_latency", raw_metrics.hit_latency_ns),
        miss_latency_ns=raw_metrics.miss_latency_ns,
        write_latency_ns=latency_floor("write_latency", raw_metrics.write_latency_ns),
        hit_energy_nj=energy_floor("read_energy", raw_metrics.hit_energy_nj),
        miss_energy_nj=raw_metrics.miss_energy_nj,
        write_energy_nj=energy_floor("write_energy", raw_metrics.write_energy_nj),
        leakage_power_mw=raw_metrics.tag_array_leakage_power_mw + device_leakage_mw,
        data_array_leakage_power_mw=device_leakage_mw,
        tag_array_leakage_power_mw=raw_metrics.tag_array_leakage_power_mw,
        refresh_latency_us=raw_metrics.refresh_latency_us,
        refresh_energy_nj_per_bank=raw_metrics.refresh_energy_nj_per_bank,
    )
    return effective, device_leakage_mw, device_refresh_mw


def build_layer(raw: Dict[str, Any], raw_metrics: DestinyMetrics, repo_root: Path,
                global_ber_max: float,
                device_library: Dict[str, Dict[str, Any]]) -> LayerSpec:
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
    require(capacity_bytes == raw_metrics.capacity_bytes,
            f"{name}: JSON capacity {capacity_bytes} differs from DESTINY "
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
    if device_family == "eDRAM":
        require(refresh_interval_us > 0.0,
                f"{name}: eDRAM refresh_interval_us must be positive")
        require(retention_time_us > 0.0,
                f"{name}: eDRAM retention_time_us must be positive")
    endurance = entry["endurance"].get("writes_per_line")
    if endurance is None:
        require("bti_endurance_writes_per_line" in raw,
                f"{name}: {device} requires bti_endurance_writes_per_line")
        endurance = float(raw["bti_endurance_writes_per_line"])
    effective_metrics, device_leakage_mw, device_refresh_mw = _apply_device_library(
        raw_metrics,
        entry,
        capacity_bytes,
        line_bytes,
        refresh_interval_us,
    )
    has_density_divisor = "density_divisor" in entry
    stacked_tiers = int(raw.get("stacked_tiers", 4 if has_density_divisor else 1))
    require(stacked_tiers > 0, f"{name}: stacked_tiers must be positive")
    effective_density_f2 = float(entry["density_f2"])
    if has_density_divisor:
        effective_density_f2 /= stacked_tiers
    return LayerSpec(
        name=name,
        device=device,
        device_family=device_family,
        destiny_config=destiny_config,
        capacity_bytes=capacity_bytes,
        associativity=associativity,
        line_bytes=line_bytes,
        replacement_policy=policy,
        banks=banks,
        peripheral_latency_ns=float(raw.get("peripheral_latency_ns", 0.0)),
        peripheral_energy_nj=float(raw.get("peripheral_energy_nj", 0.0)),
        ber=float(raw["ber"]),
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
    )


def build_crossbar(raw: Dict[str, Any]) -> CrossbarSpec:
    spec = CrossbarSpec(
        name=str(raw["name"]),
        request_cycles=float(raw.get("request_cycles", 1.0)),
        response_cycles=float(raw.get("response_cycles", 1.0)),
        clock_ghz=float(raw.get("clock_ghz", 1.0)),
        energy_pj_per_bit=float(raw.get("energy_pj_per_bit", 0.201)),
        transaction_bits=int(raw["transaction_bits"]),
    )
    require(spec.request_cycles >= 0.0 and spec.response_cycles >= 0.0,
            f"{spec.name}: crossbar cycles must be non-negative")
    require(spec.clock_ghz > 0.0, f"{spec.name}: clock_ghz must be positive")
    require(spec.energy_pj_per_bit >= 0.0,
            f"{spec.name}: energy_pj_per_bit must be non-negative")
    require(spec.transaction_bits > 0,
            f"{spec.name}: transaction_bits must be positive")
    return spec


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
    read_fraction = float(workload["read_fraction"])
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

    require(mode in {"synthetic", "trace"},
            "hit_rate_model.mode must be fixed, synthetic, or trace")
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

    warmup = int(model.get("warmup_accesses", 10_000))
    samples = int(model.get("sample_accesses", 100_000))
    require(warmup >= 0 and samples > 0,
            "warmup_accesses must be non-negative and sample_accesses positive")

    if mode == "trace":
        trace_path = _resolve_path(repo_root, str(model["path"]))
        requests: Iterable[Tuple[str, int]] = _trace_requests(trace_path, line_bytes)
    else:
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
        return float(self.workload["read_fraction"])

    @property
    def write_fraction(self) -> float:
        return 1.0 - self.read_fraction

    def _access_latency(self, layer: LayerSpec) -> float:
        return (
            self.read_fraction * layer.metrics.hit_latency_ns
            + self.write_fraction * layer.metrics.write_latency_ns
            + layer.peripheral_latency_ns
        )

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
            if layer.device_family == "eDRAM":
                results.append({
                    "layer": layer.name,
                    "constraint": "refresh<=retention",
                    "value": layer.refresh_interval_us,
                    "limit": layer.retention_time_us,
                    "pass": 0.0 < layer.refresh_interval_us <= layer.retention_time_us,
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
        return results

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

        if self.core_to_l1_latency_ns:
            add("core->L1", self.core_to_l1_latency_ns, 0.0)

        if op == "store" and target == 0:
            layer = self.layers[0]
            add("L1 store hit", layer.metrics.write_latency_ns,
                layer.metrics.write_energy_nj)
        else:
            missed = list(range(min(target, len(self.layers))))
            for index in missed:
                layer = self.layers[index]
                add(f"{layer.name} tag miss", layer.metrics.miss_latency_ns,
                    layer.metrics.miss_energy_nj)
                if index < len(self.crossbars):
                    link = self.crossbars[index]
                    add(link.name, link.latency_ns, link.energy_nj)

            if target < len(self.layers):
                layer = self.layers[target]
                add(f"{layer.name} read hit", layer.metrics.hit_latency_ns,
                    layer.metrics.hit_energy_nj)
            else:
                add("off-chip read", self.offchip.latency_ns, self.offchip.energy_nj)

            for index in reversed(missed):
                layer = self.layers[index]
                store_commit = op == "store" and index == 0
                critical = self.refill_on_critical_path or store_commit
                label = f"{layer.name} allocate + store" if store_commit else \
                        f"{layer.name} refill"
                add(label, layer.metrics.write_latency_ns,
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


def design_variants(raw: Dict[str, Any], explore: bool) -> List[Dict[str, Any]]:
    layers = raw.get("layers")
    require(isinstance(layers, list) and len(layers) == 3,
            "config must contain exactly three layers")
    choices: List[List[Dict[str, Any]]] = []
    for layer in layers:
        require(isinstance(layer, dict), "each layer must be an object")
        base = {key: value for key, value in layer.items() if key != "candidates"}
        candidates = layer.get("candidates", [{}]) if explore else [{}]
        require(isinstance(candidates, list) and candidates,
                f"{base.get('name', 'layer')}: candidates must be a non-empty list")
        expanded: List[Dict[str, Any]] = []
        for candidate in candidates:
            require(isinstance(candidate, dict), "each candidate must be an object")
            merged = copy.deepcopy(base)
            merged.update(copy.deepcopy(candidate))
            expanded.append(merged)
        choices.append(expanded)

    variants: List[Dict[str, Any]] = []
    for combination in itertools.product(*choices):
        variant = copy.deepcopy(raw)
        variant["layers"] = list(combination)
        variants.append(variant)
    return variants


def evaluate_config(raw: Dict[str, Any], repo_root: Path, runner: DestinyRunner,
                    auto_build: bool,
                    device_library: Dict[str, Dict[str, Any]]) -> Tuple[ScopeModel, Dict[str, Any]]:
    runner.ensure_built(auto_build=auto_build)
    global_ber_max = float(raw.get("constraints", {}).get("ber_max", 1.0))
    layer_specs: List[LayerSpec] = []
    for layer_raw in raw["layers"]:
        cfg_path = _resolve_path(repo_root, str(layer_raw["destiny_config"]))
        metrics = runner.run(cfg_path)
        layer_specs.append(build_layer(
            layer_raw, metrics, repo_root, global_ber_max, device_library
        ))
    require([layer.name for layer in layer_specs] == ["L1", "L2", "L3"],
            "layers must be ordered and named L1, L2, L3")
    crossbars_raw = raw.get("crossbars")
    require(isinstance(crossbars_raw, list) and len(crossbars_raw) == 2,
            "config must contain exactly two crossbars")
    crossbars = [build_crossbar(item) for item in crossbars_raw]
    off_raw = raw["off_chip"]
    offchip = OffChipSpec(
        latency_ns=float(off_raw["latency_ns"]),
        energy_nj=float(off_raw["energy_nj"]),
    )
    require(offchip.latency_ns >= 0.0 and offchip.energy_nj >= 0.0,
            "off-chip latency and energy must be non-negative")
    workload = dict(raw["workload"])
    hit_rates = estimate_hit_rates(layer_specs, workload, repo_root)
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
    report["layers"] = [
        {
            "name": layer.name,
            "device": layer.device,
            "device_family": layer.device_family,
            "destiny_config": str(layer.destiny_config.relative_to(repo_root)),
            "capacity_bytes": layer.capacity_bytes,
            "associativity": layer.associativity,
            "line_bytes": layer.line_bytes,
            "replacement_policy": layer.replacement_policy,
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
            "raw_destiny_metrics": asdict(layer.raw_metrics),
            "effective_metrics": asdict(layer.metrics),
            "combination_rule": (
                "dynamic latency/energy = max(DESTINY hierarchy result, Device Library "
                "data-bit floor); eDRAM read functions use the matching DESTINY result "
                "at Nrow; data-array leakage is replaced by Device Library and DESTINY "
                "tag-array leakage is retained; refresh/endurance/density/variation/M3D "
                "come from Device Library"
            ),
        }
        for layer in layer_specs
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
        }
        for item in crossbars
    ]
    report["hit_rate_model"] = {
        "mode": workload.get("hit_rate_model", {}).get("mode", "synthetic"),
        "observed_read_fraction": hit_rates.observed_read_fraction,
        "writebacks_per_request": list(hit_rates.writebacks_per_request),
        "offchip_writebacks_per_request": hit_rates.offchip_writebacks_per_request,
    }
    report["workload"] = workload
    report["off_chip"] = asdict(offchip)
    return model, report


def print_report(report: Dict[str, Any], instructions: Sequence[Dict[str, Any]]) -> None:
    print(f"SCOPE evaluation: {report['name']}")
    print("=" * 72)
    print("Layers (one DESTINY instance each):")
    for index, layer in enumerate(report["layers"]):
        metrics = layer["effective_metrics"]
        print(
            f"  {layer['name']}: {layer['device']}, {layer['capacity_bytes'] // 1024} KiB, "
            f"{layer['associativity']}-way, R={report['hit_rates'][index]:.6f}, "
            f"read/write={metrics['hit_latency_ns']:.3f}/{metrics['write_latency_ns']:.3f} ns"
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
    print(f"\nFoM = {report['fom_per_ns_mw']:.12g} 1/(ns*mW)")
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
    parser.add_argument("--json-output", type=Path,
                        help="also write the complete machine-readable report")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="timeout for each DESTINY instance (seconds)")
    parser.add_argument("--no-build", action="store_true",
                        help="fail instead of building a missing DESTINY binary")
    args = parser.parse_args(argv)
    if bool(args.op) != bool(args.hit_level):
        parser.error("--op and --hit-level must be provided together")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    try:
        raw = load_json(config_path.resolve())
        library_path = _resolve_path(
            repo_root, str(raw.get("device_library", "config/device_library.json"))
        )
        library_raw = load_json(library_path)
        device_library = library_raw.get("devices")
        require(isinstance(device_library, dict) and device_library,
                "device library must contain a non-empty devices object")
        binary = _resolve_path(repo_root, str(raw.get("destiny_binary", "destiny")))
        runner = DestinyRunner(repo_root, binary, timeout_s=args.timeout)
        variants = design_variants(raw, explore=args.explore)
        evaluations: List[Tuple[ScopeModel, Dict[str, Any]]] = []
        for variant in variants:
            evaluations.append(
                evaluate_config(
                    variant,
                    repo_root,
                    runner,
                    auto_build=not args.no_build,
                    device_library=device_library,
                )
            )
        feasible = [item for item in evaluations if item[1]["feasible"]]
        pool = feasible if feasible else evaluations
        model, report = max(pool, key=lambda item: item[1]["fom_per_ns_mw"])
        report["exploration"] = {
            "evaluated_designs": len(evaluations),
            "feasible_designs": len(feasible),
            "selected_highest_feasible_fom": bool(feasible),
            "designs": [
                {
                    "devices": [layer["device"] for layer in item[1]["layers"]],
                    "average_latency_ns": item[1]["average_latency_ns"],
                    "average_power_mw": item[1]["average_power_mw"],
                    "fom_per_ns_mw": item[1]["fom_per_ns_mw"],
                    "feasible": item[1]["feasible"],
                }
                for item in evaluations
            ],
        }
        report["device_library"] = {
            "path": str(library_path.relative_to(repo_root)),
            "schema_version": library_raw.get("schema_version"),
            "source": library_raw.get("source"),
            "semantics": library_raw.get("semantics"),
        }
        if args.op:
            cases = [{"op": args.op, "hit_level": args.hit_level}]
        else:
            cases = raw.get("instruction_cases", [])
        instructions = [
            model.instruction(str(case["op"]), str(case["hit_level"]))
            for case in cases
        ]
        report["instructions"] = instructions
        print_report(report, instructions)
        if args.json_output:
            output = args.json_output if args.json_output.is_absolute() else Path.cwd() / args.json_output
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
