"""Behavior-level OSFET bias-temperature-instability retention model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class BtiRetentionResult:
    enabled: bool
    model: str
    vth_max_mv: float
    anchor_shift_mv: float
    anchor_time_s: float
    stretch_exponent: float
    trapping_time_constant_s: float
    fitted_asymptotic_shift_mv: float
    reference_temperature_k: float
    operating_temperature_k: float
    activation_energy_ev: float
    temperature_acceleration: float
    reference_field_mv_per_cm: float
    operating_field_mv_per_cm: float
    field_acceleration_coefficient_cm_per_mv: float
    field_acceleration: float
    total_acceleration: float
    stress_duty_cycle: float
    reference_equivalent_retention_s: float
    equivalent_retention_s: float
    refresh_safety_factor: float
    refresh_interval_s: float
    shift_at_refresh_mv: float
    read_latency_guardband: float
    source_anchor: str
    source_time_law: str
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_bti_retention(
    model: Dict[str, Any], *, read_vgs_v: float, initial_vth_v: float,
    current_alpha: float,
) -> BtiRetentionResult:
    """Fit ΔVth(t), then solve the wall time at the configured drift limit.

    The fit uses ΔVth=A*[1-exp(-(d*t/tau)^beta)], where d is the stress
    duty cycle.  A is calibrated from the measured anchor instead of being
    guessed independently.
    """
    vth_max_mv = float(model["vth_max_mv"])
    anchor_shift_mv = float(model["anchor_shift_mv"])
    anchor_time_s = float(model["anchor_time_s"])
    beta = float(model["stretch_exponent"])
    tau_s = float(model["trapping_time_constant_s"])
    reference_temperature_k = float(model.get("reference_temperature_k", 300.0))
    operating_temperature_k = float(
        model.get("operating_temperature_k", reference_temperature_k)
    )
    activation_energy_ev = float(model.get("activation_energy_ev", 0.0))
    reference_field = float(model.get("reference_field_mv_per_cm", 0.0))
    operating_field = float(model.get(
        "operating_field_mv_per_cm", reference_field
    ))
    field_coefficient = float(model.get(
        "field_acceleration_coefficient_cm_per_mv", 0.0
    ))
    duty_cycle = float(model.get("stress_duty_cycle", 1.0))
    safety = float(model.get("refresh_safety_factor", 0.8))
    if min(vth_max_mv, anchor_shift_mv, anchor_time_s, beta, tau_s) <= 0.0:
        raise ValueError("BTI fit parameters must be positive")
    if min(reference_temperature_k, operating_temperature_k) <= 0.0:
        raise ValueError("BTI temperatures must be positive")
    if min(activation_energy_ev, reference_field, operating_field,
           field_coefficient) < 0.0:
        raise ValueError("BTI acceleration parameters must be non-negative")
    if not 0.0 < duty_cycle <= 1.0:
        raise ValueError("BTI stress duty cycle must be in (0, 1]")
    if not 0.0 < safety <= 1.0:
        raise ValueError("BTI refresh safety factor must be in (0, 1]")
    if read_vgs_v <= initial_vth_v or current_alpha <= 0.0:
        raise ValueError("BTI current guardband needs Vgs>Vth and alpha>0")

    anchor_fraction = 1.0 - math.exp(-(
        duty_cycle * anchor_time_s / tau_s
    ) ** beta)
    asymptotic_mv = anchor_shift_mv / anchor_fraction
    if vth_max_mv >= asymptotic_mv:
        raise ValueError("configured Vth limit is not reached by the fitted BTI law")

    reference_active_retention_s = tau_s * (
        -math.log1p(-vth_max_mv / asymptotic_mv)
    ) ** (1.0 / beta)
    boltzmann_ev_per_k = 8.617333262e-5
    temperature_acceleration = math.exp(
        activation_energy_ev / boltzmann_ev_per_k
        * (1.0 / reference_temperature_k - 1.0 / operating_temperature_k)
    )
    field_acceleration = math.exp(
        field_coefficient * (operating_field - reference_field)
    )
    total_acceleration = temperature_acceleration * field_acceleration
    reference_equivalent_retention_s = reference_active_retention_s / duty_cycle
    equivalent_retention_s = reference_equivalent_retention_s / total_acceleration
    refresh_interval_s = safety * equivalent_retention_s
    shift_at_refresh_mv = asymptotic_mv * (1.0 - math.exp(-(
        duty_cycle * total_acceleration * refresh_interval_s / tau_s
    ) ** beta))

    fresh_overdrive_v = read_vgs_v - initial_vth_v
    guarded_overdrive_v = fresh_overdrive_v - shift_at_refresh_mv / 1000.0
    if guarded_overdrive_v <= 0.0:
        raise ValueError("BTI guardband consumes the full read overdrive")
    latency_guardband = (
        fresh_overdrive_v / guarded_overdrive_v
    ) ** current_alpha

    return BtiRetentionResult(
        enabled=True,
        model="calibrated_stretched_exponential",
        vth_max_mv=vth_max_mv,
        anchor_shift_mv=anchor_shift_mv,
        anchor_time_s=anchor_time_s,
        stretch_exponent=beta,
        trapping_time_constant_s=tau_s,
        fitted_asymptotic_shift_mv=asymptotic_mv,
        reference_temperature_k=reference_temperature_k,
        operating_temperature_k=operating_temperature_k,
        activation_energy_ev=activation_energy_ev,
        temperature_acceleration=temperature_acceleration,
        reference_field_mv_per_cm=reference_field,
        operating_field_mv_per_cm=operating_field,
        field_acceleration_coefficient_cm_per_mv=field_coefficient,
        field_acceleration=field_acceleration,
        total_acceleration=total_acceleration,
        stress_duty_cycle=duty_cycle,
        reference_equivalent_retention_s=reference_equivalent_retention_s,
        equivalent_retention_s=equivalent_retention_s,
        refresh_safety_factor=safety,
        refresh_interval_s=refresh_interval_s,
        shift_at_refresh_mv=shift_at_refresh_mv,
        read_latency_guardband=latency_guardband,
        source_anchor=str(model["source_anchor"]),
        source_time_law=str(model["source_time_law"]),
        interpretation=(
            "Cross-device behavior-level fit: the measured ITO:F point anchors "
            "the voltage scale, while the recent oxide-TFT stretched-exponential "
            "range supplies beta and tau. Temperature uses an Arrhenius law; "
            "the field factor is a disclosed SCOPE stress-corner calibration. "
            "This is not a direct lifetime measurement of the modeled cache cell."
        ),
    )
