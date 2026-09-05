"""Generic monolithic-3D inter-tier interconnect overlay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
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
    write_energy_penalty_nj: float | None = None
    per_tier: tuple = ()
    array_height_um: float = 0.0
    array_width_um: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_m3d(
    config: Dict[str, Any], defaults: Dict[str, Any], *, banks: int,
    line_bits: int, data_array_area_mm2: float,
) -> M3DResult:
    values = dict(defaults)
    values.update(config)
    if values.get("model") == "elmore_staircase":
        return _evaluate_elmore(values, banks, line_bits, data_array_area_mm2)
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


def _evaluate_elmore(values, banks, line_bits, area_mm2):
    """Per-path RC ladder; energy is integrated before access-rate scaling."""
    enabled = bool(values.get("enabled", False))
    tiers = int(values.get("tiers", 1)) if enabled else 1
    if tiers < 1 or banks < 1 or area_mm2 < 0:
        raise ValueError("invalid tier count, bank count or array area")
    def positive(name, default=0.0):
        result = float(values.get(name, default))
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return result
    r = positive("via_resistance_ohm")
    c = positive("via_capacitance_ff")
    pitch = positive("via_pitch_um")
    count = int(values.get("array_count", banks))
    if count < 1:
        raise ValueError("array_count must be positive")
    side = math.sqrt(area_mm2 * 1e6 / count)
    height = positive("array_height_um", side)
    width = positive("array_width_um", side)
    # Tier 0 shares the driver plane; optional bottom hop models a separate logic tier.
    bottom = int(values.get("bottom_tier_hops", 0))
    if bottom < 0:
        raise ValueError("bottom_tier_hops must be nonnegative")
    hops = [bottom + i for i in range(tiers)] if enabled else [0]
    maximum = max(hops)
    def segments(name, scalar):
        result = [float(x) for x in values.get(name, [scalar] * maximum)]
        if len(result) < maximum or any(not math.isfinite(x) or x < 0 for x in result):
            raise ValueError(f"invalid {name}")
        return result
    rs = segments("segment_resistance_ohm", r)
    cs = segments("segment_capacitance_ff", c)
    probabilities = values.get("tier_probabilities", [1.0 / tiers] * tiers) if enabled else [1.0]
    if (len(probabilities) != tiers or
        any(not math.isfinite(p) or p < 0 for p in probabilities) or
        not math.isclose(sum(probabilities), 1.0)):
        raise ValueError("tier_probabilities must be nonnegative and sum to one")
    energies = {}
    for operation in ("read", "write"):
        swing = positive(f"{operation}_swing_v", values.get("vdd", 1.2))
        current = positive(f"{operation}_current_ua") * 1e-6
        pulse = positive(f"{operation}_pulse_ns") * 1e-9
        if current and not pulse:
            raise ValueError("nonzero conduction current requires a pulse duration")
        active = positive(f"{operation}_active_lines", line_bits + 1)
        energies[operation] = [
            active * (0.5 * sum(cs[:h]) * 1e-15 * swing**2
                      + current**2 * sum(rs[:h]) * pulse) * 1e9
            for h in hops
        ]
    delays = [0.69 * sum(sum(rs[:i+1]) * cs[i] * 1e-15
                          for i in range(h)) * 1e9 for h in hops]
    per_tier = tuple({"tier": i + 1, "hops": h, "probability": probabilities[i],
                      "latency_ns": delays[i], "read_energy_nj": energies["read"][i],
                      "write_energy_nj": energies["write"][i]}
                     for i, h in enumerate(hops))
    average = lambda numbers: sum(p * x for p, x in zip(probabilities, numbers))
    margin = 2 * tiers * pitch if enabled and tiers > 1 else 0.0
    extra = count * ((height + margin) * (width + margin) - height * width) / 1e6
    return M3DResult(
        enabled=enabled, tiers=tiers, average_tier_hops=average(hops),
        vertical_data_bits=line_bits, vias_per_interface=0, via_count_total=0,
        via_resistance_ohm=r, via_capacitance_ff=c, via_pitch_um=pitch,
        intrinsic_rc_latency_ns=average(delays), driven_hop_latency_ns=0.0,
        interface_energy_fj_per_bit=0.0,
        via_switching_energy_fj_per_bit=0.5*c*positive("read_swing_v", 1.2)**2,
        latency_penalty_ns=average(delays), energy_penalty_nj=average(energies["read"]),
        write_energy_penalty_nj=average(energies["write"]),
        footprint_mm2=extra, footprint_percent_of_data_array=100*extra/area_mm2 if area_mm2 else 0,
        model="Elmore RC ladder; 0.5*C*swing^2 plus independent conduction I^2*R*t; staircase footprint; via counts unspecified",
        per_tier=per_tier, array_height_um=height, array_width_um=width,
    )
