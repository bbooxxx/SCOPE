"""Behavior-level array non-ideality overlay for SCOPE."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class NonidealResult:
    enabled: bool
    mechanism: str
    r0_ohm_per_cell: float
    r1_ohm_per_cell: float
    ideal_selected_current_ua: float
    unselected_current_ua_per_cell: float
    real_selected_current_ua: float
    rwl_ir_drop_mv: float
    rbl_ir_drop_mv: float
    ir_power_uw_per_active_column: float
    ir_energy_nj_per_line: float
    coupling_sn_rwl_mv: float
    coupling_sn_rbl_mv: float
    coupling_sn_wwl_mv: float
    coupling_net_sn_mv: float
    effective_signal_mv: float
    read_latency_penalty_ns: float
    read_energy_penalty_nj: float
    read_disturb_probability_per_read: float
    read_disturb_target_per_read: float
    read_disturb_current_limited: bool
    nominal_ber: float
    effective_ber: float
    reliability_model: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _gaussian_tail(signal_v: float, sigma_v: float) -> float:
    if sigma_v <= 0.0:
        return 0.0 if signal_v > 0.0 else 0.5
    return 0.5 * math.erfc(max(0.0, signal_v) / (math.sqrt(2.0) * sigma_v))


def _stt_disturb_probability(
    pulse_s: float, attempt_s: float, thermal_stability: float,
    read_current_a: float, critical_current_a: float,
) -> float:
    if pulse_s <= 0.0 or critical_current_a <= 0.0:
        return 0.0
    exponent = -thermal_stability * (1.0 - read_current_a / critical_current_a)
    hazard = pulse_s / attempt_s * math.exp(min(50.0, exponent))
    return -math.expm1(-hazard)


def evaluate_nonideal(
    raw_metrics: Any,
    effective_hit_latency_ns: float,
    line_bits: int,
    device_entry: Dict[str, Any],
    sense_amp: Dict[str, Any] | None,
    nominal_ber: float,
) -> NonidealResult:
    """Evaluate worst-case selected cell and return incremental penalties.

    The model is intentionally array-level: DESTINY supplies the chosen
    physical line resistance and subarray dimensions; the device library
    supplies device-current, coupling, and reliability parameters.
    """
    cfg = dict(device_entry.get("nonideal", {}))
    family = str(device_entry.get("family", ""))
    enabled = bool(cfg.get("enabled", False))
    nrow = max(1, int(raw_metrics.subarray_rows))
    ncol = max(1, int(raw_metrics.subarray_columns))
    rwl_total = max(0.0, float(raw_metrics.rwl_resistance_ohm))
    rbl_total = max(0.0, float(raw_metrics.rbl_resistance_ohm))
    r0 = rwl_total / ncol
    r1 = rbl_total / nrow
    sa = dict(sense_amp or {})
    noise_sigma_v = float(sa.get("input_referred_noise_mv", 8.0)) * 1e-3

    if not enabled:
        signal_v = float(cfg.get("sense_signal_mv", 100.0)) * 1e-3
        return NonidealResult(
            enabled=False, mechanism="not enabled", r0_ohm_per_cell=r0,
            r1_ohm_per_cell=r1, ideal_selected_current_ua=0.0,
            unselected_current_ua_per_cell=0.0, real_selected_current_ua=0.0,
            rwl_ir_drop_mv=0.0, rbl_ir_drop_mv=0.0,
            ir_power_uw_per_active_column=0.0, ir_energy_nj_per_line=0.0,
            coupling_sn_rwl_mv=0.0, coupling_sn_rbl_mv=0.0,
            coupling_sn_wwl_mv=0.0, coupling_net_sn_mv=0.0,
            effective_signal_mv=signal_v * 1e3,
            read_latency_penalty_ns=0.0, read_energy_penalty_nj=0.0,
            read_disturb_probability_per_read=0.0,
            read_disturb_target_per_read=0.0,
            read_disturb_current_limited=False, nominal_ber=nominal_ber,
            effective_ber=max(nominal_ber, _gaussian_tail(signal_v, noise_sigma_v)),
            reliability_model="nominal library BER plus configurable SA noise",
        )

    if family == "eDRAM":
        circuit = dict(device_entry["read_circuit"])
        ideal_a = (
            float(circuit["ion_ua_per_um"])
            * float(circuit.get("effective_width_um", 1.0)) * 1e-6
        )
        unsel_a = float(cfg.get("unselected_current_na", 0.0)) * 1e-9
        after_crosstalk_a = max(
            ideal_a * float(cfg.get("minimum_current_fraction", 0.05)),
            ideal_a - (nrow - 1) * unsel_a,
        )
        rwl_path = 0.5 * rwl_total
        rbl_path = rbl_total
        rwl_drop_v = after_crosstalk_a * rwl_path

        coupling = dict(cfg.get("coupling", {}))
        cgsw = float(coupling.get("cgsw_ff", 0.0))
        cgsr = float(coupling.get("cgsr_ff", 0.0))
        cgdr = float(coupling.get("cgdr_ff", 0.0))
        cpara = float(coupling.get("cpara_ff", 0.0))
        denominator = max(1e-30, cgsw + cgsr + cgdr + cpara)
        sn_rwl_v = float(coupling.get("rwl_transition_v", 0.0)) * cgsr / denominator
        sn_rbl_v = float(coupling.get("rbl_transition_v", 0.0)) * cgdr / denominator
        sn_wwl_v = float(coupling.get("wwl_transition_v", 0.0)) * cgsw / denominator
        signed_net_v = sn_rwl_v + sn_rbl_v + sn_wwl_v
        if bool(coupling.get("complementary_polarity", False)):
            signed_net_v *= float(coupling.get("residual_mismatch_fraction", 0.1))
        adverse_sn_v = abs(signed_net_v)

        vgs = float(cfg.get("read_vgs_v", circuit.get("vdd", 1.2)))
        vth = float(cfg.get("read_vth_v", 0.3))
        alpha = float(cfg.get("current_alpha", 1.3))
        overdrive = max(1e-3, vgs - vth)
        effective_overdrive = max(
            overdrive * float(cfg.get("minimum_current_fraction", 0.05)),
            overdrive - rwl_drop_v - adverse_sn_v,
        )
        real_a = after_crosstalk_a * (effective_overdrive / overdrive) ** alpha
        rbl_drop_v = real_a * rbl_path
        sense_delta_v = float(circuit["delta_v"])
        capacitance_f = float(raw_metrics.rbl_capacitance_ff) * 1e-15
        ideal_cell_ns = capacitance_f * sense_delta_v / ideal_a * 1e9
        real_cell_ns = capacitance_f * (sense_delta_v + rbl_drop_v) / real_a * 1e9
        latency_penalty_ns = max(0.0, real_cell_ns - ideal_cell_ns)
        ir_power_w = real_a * real_a * (rwl_path + rbl_path)
        ir_energy_nj = ir_power_w * real_cell_ns * 1e-9 * line_bits * 1e9
        current_ratio = min(1.0, real_a / ideal_a)
        effective_signal_v = max(0.0, sense_delta_v * current_ratio - adverse_sn_v)
        effective_ber = max(nominal_ber, _gaussian_tail(
            effective_signal_v, noise_sigma_v
        ))
        return NonidealResult(
            enabled=True,
            mechanism="gain-cell RWL/RBL IR drop + 2T0C coupling/read crosstalk",
            r0_ohm_per_cell=r0, r1_ohm_per_cell=r1,
            ideal_selected_current_ua=ideal_a * 1e6,
            unselected_current_ua_per_cell=unsel_a * 1e6,
            real_selected_current_ua=real_a * 1e6,
            rwl_ir_drop_mv=rwl_drop_v * 1e3,
            rbl_ir_drop_mv=rbl_drop_v * 1e3,
            ir_power_uw_per_active_column=ir_power_w * 1e6,
            ir_energy_nj_per_line=ir_energy_nj,
            coupling_sn_rwl_mv=sn_rwl_v * 1e3,
            coupling_sn_rbl_mv=sn_rbl_v * 1e3,
            coupling_sn_wwl_mv=sn_wwl_v * 1e3,
            coupling_net_sn_mv=signed_net_v * 1e3,
            effective_signal_mv=effective_signal_v * 1e3,
            read_latency_penalty_ns=latency_penalty_ns,
            read_energy_penalty_nj=ir_energy_nj,
            read_disturb_probability_per_read=0.0,
            read_disturb_target_per_read=0.0,
            read_disturb_current_limited=False,
            nominal_ber=nominal_ber, effective_ber=effective_ber,
            reliability_model=(
                "Ireal=Ion-(Nrow-1)Iunsel, then alpha-power Vgs derating; "
                "BER=Q(effective voltage signal / SA input-noise sigma)"
            ),
        )

    if family == "MRAM":
        ideal_a = float(cfg.get("ideal_read_current_ua", 20.0)) * 1e-6
        read_v = float(cfg.get("read_voltage_v", 0.2))
        cell_path_r = read_v / ideal_a
        wire_r = rwl_total + rbl_total
        delivered_a = read_v / (cell_path_r + wire_r)
        disturb_probability = 0.0
        target_probability = 0.0
        current_limited = False
        real_a = delivered_a
        if bool(cfg.get("stt_read_disturb", False)):
            thermal = float(cfg.get("thermal_stability", 60.0))
            critical_a = float(cfg.get("critical_current_ua", 200.0)) * 1e-6
            attempt_s = float(cfg.get("attempt_time_ns", 1.0)) * 1e-9
            target_probability = float(cfg.get("max_disturb_probability_per_read", 1e-12))
            pulse_s = max(1e-15, effective_hit_latency_ns * 1e-9)
            target_hazard = -math.log1p(-target_probability)
            ratio_limit = 1.0 + math.log(
                max(1e-300, target_hazard * attempt_s / pulse_s)
            ) / thermal
            allowed_a = critical_a * max(0.01, min(0.99, ratio_limit))
            if real_a > allowed_a:
                real_a = allowed_a
                current_limited = True
            disturb_probability = _stt_disturb_probability(
                pulse_s, attempt_s, thermal, real_a, critical_a
            )
        current_ratio = max(1e-9, real_a / ideal_a)
        latency_penalty_ns = max(
            0.0, effective_hit_latency_ns / current_ratio - effective_hit_latency_ns
        )
        pulse_ns = effective_hit_latency_ns + latency_penalty_ns
        ir_power_w = real_a * real_a * wire_r
        ir_energy_nj = ir_power_w * pulse_ns * 1e-9 * line_bits * 1e9
        signal_v = float(cfg.get("sense_signal_mv", 100.0)) * 1e-3 * current_ratio
        ber_model = str(cfg.get("ber_model", "sa_gaussian_tail"))
        if ber_model == "nominal":
            effective_ber = nominal_ber
        elif ber_model == "sa_gaussian_tail":
            effective_ber = max(
                nominal_ber, _gaussian_tail(signal_v, noise_sigma_v)
            )
        else:
            raise ValueError(f"unsupported MRAM BER model: {ber_model}")
        return NonidealResult(
            enabled=True, mechanism="MRAM word/bit-line IR drop and STT read disturb",
            r0_ohm_per_cell=r0, r1_ohm_per_cell=r1,
            ideal_selected_current_ua=ideal_a * 1e6,
            unselected_current_ua_per_cell=0.0,
            real_selected_current_ua=real_a * 1e6,
            rwl_ir_drop_mv=real_a * rwl_total * 1e3,
            rbl_ir_drop_mv=real_a * rbl_total * 1e3,
            ir_power_uw_per_active_column=ir_power_w * 1e6,
            ir_energy_nj_per_line=ir_energy_nj,
            coupling_sn_rwl_mv=0.0, coupling_sn_rbl_mv=0.0,
            coupling_sn_wwl_mv=0.0, coupling_net_sn_mv=0.0,
            effective_signal_mv=signal_v * 1e3,
            read_latency_penalty_ns=latency_penalty_ns,
            read_energy_penalty_nj=ir_energy_nj,
            read_disturb_probability_per_read=disturb_probability,
            read_disturb_target_per_read=target_probability,
            read_disturb_current_limited=current_limited,
            nominal_ber=nominal_ber, effective_ber=effective_ber,
            reliability_model=(
                "MRAM BER uses the configured nominal value; series-line "
                "current derating still affects latency/energy, and the STT "
                "thermal-activation model independently limits read current"
                if ber_model == "nominal" else
                "series-line current derating; STT thermal-activation switching "
                "probability limits read current (write current is unchanged)"
            ),
        )

    return evaluate_nonideal(
        raw_metrics, effective_hit_latency_ns, line_bits,
        {**device_entry, "nonideal": {"enabled": False}}, sense_amp, nominal_ber,
    )
