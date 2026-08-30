"""Behavior-level selectable sense-amplifier overlay for SCOPE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SenseAmpResult:
    selected_type: str
    destiny_native_type: str
    read_signal: str
    supported_types: tuple[str, ...]
    compatible: bool
    base_voltage_latency_ns: float
    base_voltage_energy_nj: float
    base_voltage_leakage_mw: float
    selected_latency_ns: float
    selected_energy_nj: float
    selected_leakage_mw: float
    destiny_reported_latency_ns: float
    destiny_reported_energy_nj: float
    destiny_reported_leakage_mw: float
    legacy_iv_converter_latency_ns: float
    legacy_iv_converter_energy_nj: float
    legacy_iv_converter_leakage_mw: float
    hit_latency_delta_ns: float
    hit_energy_delta_nj: float
    model_factors: Dict[str, float]
    comparison_basis: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_sense_amp(
    raw_metrics: Any,
    device_entry: Dict[str, Any],
    models: Dict[str, Any],
    requested_type: str | None,
) -> SenseAmpResult:
    """Replace DESTINY's fixed read-mode choice with a compatible SA variant.

    DESTINY's current mode is a voltage latch preceded by a tabulated I-V
    converter.  SCOPE first removes that converter to recover a common voltage
    baseline, then applies configurable current/voltage topology factors.
    """
    interface = dict(device_entry["sense_amplifier"])
    native = str(interface["destiny_native_type"]).lower()
    supported = tuple(str(value).lower() for value in interface["supported_types"])
    selected = str(requested_type or interface.get("default_type", native)).lower()
    compatible = selected in supported
    if not compatible:
        raise ValueError(
            f"sense amplifier {selected} is incompatible with "
            f"{interface['read_signal']}; supported={supported}"
        )
    factors = dict(models[selected])
    raw_latency = float(raw_metrics.sense_amp_latency_ns)
    raw_energy = float(raw_metrics.sense_amp_read_energy_nj)
    raw_leakage = float(raw_metrics.sense_amp_leakage_mw)
    iv_latency = float(raw_metrics.legacy_iv_converter_latency_ns)
    iv_energy = float(raw_metrics.legacy_iv_converter_read_energy_nj)
    iv_leakage = float(raw_metrics.legacy_iv_converter_leakage_mw)
    base_latency = max(0.0, raw_latency - iv_latency)
    base_energy = max(0.0, raw_energy - iv_energy)
    base_leakage = max(0.0, raw_leakage - iv_leakage)
    selected_latency = base_latency * float(factors["latency_factor"])
    selected_energy = base_energy * float(factors["energy_factor"])
    selected_leakage = base_leakage * float(factors["leakage_factor"])
    return SenseAmpResult(
        selected_type=selected,
        destiny_native_type=native,
        read_signal=str(interface["read_signal"]),
        supported_types=supported,
        compatible=compatible,
        base_voltage_latency_ns=base_latency,
        base_voltage_energy_nj=base_energy,
        base_voltage_leakage_mw=base_leakage,
        selected_latency_ns=selected_latency,
        selected_energy_nj=selected_energy,
        selected_leakage_mw=selected_leakage,
        destiny_reported_latency_ns=raw_latency,
        destiny_reported_energy_nj=raw_energy,
        destiny_reported_leakage_mw=raw_leakage,
        legacy_iv_converter_latency_ns=iv_latency,
        legacy_iv_converter_energy_nj=iv_energy,
        legacy_iv_converter_leakage_mw=iv_leakage,
        hit_latency_delta_ns=selected_latency - raw_latency,
        hit_energy_delta_nj=selected_energy - raw_energy,
        model_factors={key: float(value) for key, value in factors.items()
                       if key.endswith("factor")},
        comparison_basis=(
            "DESTINY current mode = tabulated I-V converter + the same voltage "
            "latch equation; SCOPE v5 instead compares explicit compatible "
            "current/voltage topology variants against a converter-free baseline"
        ),
    )
