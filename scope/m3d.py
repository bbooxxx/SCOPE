"""Generic monolithic-3D inter-tier interconnect overlay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class M3DResult:
    enabled: bool
    tiers: int
    average_tier_hops: float
    vertical_data_bits: int
    vias_per_interface: int
    via_count_total: int
    via_resistance_ohm: float
    via_capacitance_ff: float
    via_pitch_um: float
    intrinsic_rc_latency_ns: float
    driven_hop_latency_ns: float
    interface_energy_fj_per_bit: float
    via_switching_energy_fj_per_bit: float
    latency_penalty_ns: float
    energy_penalty_nj: float
    footprint_mm2: float
    footprint_percent_of_data_array: float
    model: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_m3d(
    config: Dict[str, Any], defaults: Dict[str, Any], *, banks: int,
    line_bits: int, data_array_area_mm2: float,
) -> M3DResult:
    values = dict(defaults)
    values.update(config)
    enabled = bool(values.get("enabled", False))
    tiers = int(values.get("tiers", 1)) if enabled else 1
    if tiers < 1:
        raise ValueError("M3D tiers must be positive")
    r_via = float(values["via_resistance_ohm"])
    c_via_ff = float(values["via_capacitance_ff"])
    pitch_um = float(values["via_pitch_um"])
    vdd = float(values["vdd"])
    driver_r = float(values["driver_resistance_ohm"])
    address_bits = int(values.get("address_bits", 49))
    control_bits = int(values.get("control_bits", 8))
    vertical_data_bits = int(values.get("vertical_data_bits", line_bits))
    via_count_per_interface = banks * (vertical_data_bits + address_bits + control_bits)
    interfaces = max(0, tiers - 1)
    via_count_total = via_count_per_interface * interfaces
    average_hops = interfaces / 2.0
    hop_delay_s = 0.69 * (driver_r + 0.5 * r_via) * c_via_ff * 1e-15
    intrinsic_ns = average_hops * hop_delay_s * 1e9
    driven_hop_ns = float(values.get("driven_hop_latency_ps", 0.0)) * 1e-3
    latency_ns = average_hops * max(hop_delay_s * 1e9, driven_hop_ns)
    interface_energy_fj = float(values.get(
        "interface_energy_fj_per_bit", 0.0
    ))
    via_energy_fj = c_via_ff * vdd ** 2
    energy_nj = (
        average_hops * vertical_data_bits
        * (via_energy_fj + interface_energy_fj) / 1e6
    )
    footprint_mm2 = via_count_total * pitch_um ** 2 / 1e6
    footprint_percent = (
        100.0 * footprint_mm2 / data_array_area_mm2
        if data_array_area_mm2 > 0.0 else 0.0
    )
    return M3DResult(
        enabled=enabled,
        tiers=tiers,
        average_tier_hops=average_hops,
        vertical_data_bits=vertical_data_bits,
        vias_per_interface=via_count_per_interface,
        via_count_total=via_count_total,
        via_resistance_ohm=r_via,
        via_capacitance_ff=c_via_ff,
        via_pitch_um=pitch_um,
        intrinsic_rc_latency_ns=intrinsic_ns,
        driven_hop_latency_ns=driven_hop_ns,
        interface_energy_fj_per_bit=interface_energy_fj,
        via_switching_energy_fj_per_bit=via_energy_fj,
        latency_penalty_ns=latency_ns,
        energy_penalty_nj=energy_nj,
        footprint_mm2=footprint_mm2,
        footprint_percent_of_data_array=footprint_percent,
        model=(
            "uniform tier access; reports intrinsic Elmore RC separately, but "
            "critical latency uses max(intrinsic RC, measured driven-hop delay); "
            "energy includes Cvia*Vdd^2 plus configurable interface energy; "
            "footprint counts dedicated via landing/keep-out pitch"
        ),
    )
