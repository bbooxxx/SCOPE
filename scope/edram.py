"""Circuit-level equations used by the SCOPE v3 eDRAM device library."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class EdramReadResult:
    nrow: int
    ncolumn: int
    rbl_capacitance_ff: float
    rbl_wire_capacitance_ff: float
    rbl_cell_capacitance_ff: float
    delta_v: float
    vdd: float
    ion_ua_per_um: float
    effective_width_um: float
    pure_device_path: str
    cell_read_latency_ns: float
    peripheral_read_latency_ns: float
    total_read_latency_ns: float
    read_energy_fj_per_bit: float
    read_energy_nj_per_line: float
    peripheral_read_energy_nj: float
    total_read_energy_nj: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RefreshResult:
    enabled: bool
    retention_time_us: float
    refresh_interval_us: float
    rows_per_subarray: int
    columns_per_subarray: int
    subarray_count: int
    refresh_energy_nj_per_row: float
    refresh_power_mw_per_subarray: float
    refresh_power_mw_total: float
    refresh_row_latency_ns: float
    refresh_rows_per_bank: int
    refresh_busy_time_ns_per_interval: float
    bandwidth_occupancy: float
    availability: float
    schedulable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_read(
    *,
    nrow: int,
    ncolumn: int,
    rbl_capacitance_ff: float,
    rbl_wire_capacitance_ff: float,
    rbl_cell_capacitance_ff: float,
    delta_v: float,
    vdd: float,
    ion_ua_per_um: float,
    effective_width_um: float,
    pure_device_path: str,
    line_bits: int,
    peripheral_read_latency_ns: float,
    peripheral_read_energy_nj: float,
) -> EdramReadResult:
    """Evaluate Lr=C*ΔV/Ion and Er=1/2*C*(Vdd²-(Vdd-ΔV)²)."""
    if min(nrow, ncolumn, line_bits) <= 0:
        raise ValueError("eDRAM geometry and cache-line width must be positive")
    if rbl_capacitance_ff <= 0.0:
        raise ValueError("RBL capacitance must be positive")
    if not (0.0 < delta_v < vdd):
        raise ValueError("delta_v must be between zero and Vdd")
    if ion_ua_per_um <= 0.0 or effective_width_um <= 0.0:
        raise ValueError("Ion and effective transistor width must be positive")

    capacitance_f = rbl_capacitance_ff * 1e-15
    current_a = ion_ua_per_um * effective_width_um * 1e-6
    cell_latency_ns = capacitance_f * delta_v / current_a * 1e9
    energy_j_per_bit = 0.5 * capacitance_f * (
        vdd * vdd - (vdd - delta_v) * (vdd - delta_v)
    )
    energy_fj_per_bit = energy_j_per_bit * 1e15
    energy_nj_per_line = energy_j_per_bit * line_bits * 1e9
    return EdramReadResult(
        nrow=nrow,
        ncolumn=ncolumn,
        rbl_capacitance_ff=rbl_capacitance_ff,
        rbl_wire_capacitance_ff=rbl_wire_capacitance_ff,
        rbl_cell_capacitance_ff=rbl_cell_capacitance_ff,
        delta_v=delta_v,
        vdd=vdd,
        ion_ua_per_um=ion_ua_per_um,
        effective_width_um=effective_width_um,
        pure_device_path=pure_device_path,
        cell_read_latency_ns=cell_latency_ns,
        peripheral_read_latency_ns=max(0.0, peripheral_read_latency_ns),
        total_read_latency_ns=cell_latency_ns + max(0.0, peripheral_read_latency_ns),
        read_energy_fj_per_bit=energy_fj_per_bit,
        read_energy_nj_per_line=energy_nj_per_line,
        peripheral_read_energy_nj=max(0.0, peripheral_read_energy_nj),
        total_read_energy_nj=energy_nj_per_line + max(0.0, peripheral_read_energy_nj),
    )


def evaluate_row_refresh(
    *,
    capacity_bytes: int,
    banks: int,
    nrow: int,
    ncolumn: int,
    read_energy_fj_per_bit: float,
    write_energy_fj_per_bit: float,
    read_latency_ns: float,
    write_latency_ns: float,
    refresh_interval_us: float,
    retention_time_us: float,
) -> RefreshResult:
    """Evaluate periodic row read+write power and serialized bank occupancy."""
    if min(capacity_bytes, banks, nrow, ncolumn) <= 0:
        raise ValueError("cache and subarray geometry must be positive")
    if refresh_interval_us <= 0.0 or retention_time_us <= 0.0:
        raise ValueError("refresh and retention intervals must be positive")
    if refresh_interval_us > retention_time_us:
        raise ValueError("Tref must not exceed Tret")

    capacity_bits = capacity_bytes * 8
    subarray_bits = nrow * ncolumn
    subarrays = math.ceil(capacity_bits / subarray_bits)
    row_energy_nj = ncolumn * (
        read_energy_fj_per_bit + write_energy_fj_per_bit
    ) / 1e6
    # nJ/us is mW. This is exactly Nrow*Eref,row/Tref per subarray.
    power_per_subarray_mw = nrow * row_energy_nj / refresh_interval_us
    total_power_mw = subarrays * power_per_subarray_mw

    row_latency_ns = read_latency_ns + write_latency_ns
    subarrays_per_bank = math.ceil(subarrays / banks)
    rows_per_bank = subarrays_per_bank * nrow
    busy_ns = rows_per_bank * row_latency_ns
    interval_ns = refresh_interval_us * 1000.0
    occupancy = busy_ns / interval_ns
    availability = max(0.0, 1.0 - occupancy)
    return RefreshResult(
        enabled=True,
        retention_time_us=retention_time_us,
        refresh_interval_us=refresh_interval_us,
        rows_per_subarray=nrow,
        columns_per_subarray=ncolumn,
        subarray_count=subarrays,
        refresh_energy_nj_per_row=row_energy_nj,
        refresh_power_mw_per_subarray=power_per_subarray_mw,
        refresh_power_mw_total=total_power_mw,
        refresh_row_latency_ns=row_latency_ns,
        refresh_rows_per_bank=rows_per_bank,
        refresh_busy_time_ns_per_interval=busy_ns,
        bandwidth_occupancy=occupancy,
        availability=availability,
        schedulable=occupancy < 1.0,
    )


def evaluate_si_refresh(**kwargs: Any) -> RefreshResult:
    """Compatibility alias for the original Si-eDRAM refresh API."""
    return evaluate_row_refresh(**kwargs)


def no_refresh() -> RefreshResult:
    return RefreshResult(
        enabled=False,
        retention_time_us=0.0,
        refresh_interval_us=0.0,
        rows_per_subarray=0,
        columns_per_subarray=0,
        subarray_count=0,
        refresh_energy_nj_per_row=0.0,
        refresh_power_mw_per_subarray=0.0,
        refresh_power_mw_total=0.0,
        refresh_row_latency_ns=0.0,
        refresh_rows_per_bank=0,
        refresh_busy_time_ns_per_interval=0.0,
        bandwidth_occupancy=0.0,
        availability=1.0,
        schedulable=True,
    )
