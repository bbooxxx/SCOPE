"""Explicit power units and the original v9 analytical activity budget."""

import math


def legacy_v9_activity_counts(operator: str, shape: dict, *,
                              bytes_per_element: int = 2, isa_bytes: int = 16,
                              transaction_bytes: int = 128) -> dict:
    """Original c87e22d count model: weights once, not emitted tiled traffic."""
    s, h, inner, tile_m = (shape[k] for k in
                           ("sequence_tokens", "hidden_size", "intermediate_size", "tile_m"))
    values = (s, h, inner, tile_m, bytes_per_element, isa_bytes, transaction_bytes)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("shape and byte counts must be positive integers")
    if transaction_bytes < isa_bytes or transaction_bytes % isa_bytes:
        raise ValueError("transaction_bytes must be a multiple of isa_bytes")
    if operator == "attention":
        q_tiles = (s + tile_m - 1) // tile_m
        load_elements = 4 * h * h + (3 + 2 * q_tiles) * s * h
        store_elements = 5 * s * h
    elif operator == "ffn":
        load_elements = 3 * h * inner + 2 * s * h + 2 * s * inner
        store_elements = 2 * s * inner + s * h
    else:
        raise ValueError("operator must be attention or ffn")
    elements_per_isa = max(1, isa_bytes // bytes_per_element)
    isa_per_transaction = transaction_bytes // isa_bytes
    def transactions(elements):
        isa_count = (elements + elements_per_isa - 1) // elements_per_isa
        return (isa_count + isa_per_transaction - 1) // isa_per_transaction
    loads, stores = transactions(load_elements), transactions(store_elements)
    return {"loads": loads, "stores": stores, "total": loads + stores}


def access_power_uw(energy_fj: float, duration_ns: float) -> float:
    if not math.isfinite(energy_fj) or energy_fj < 0:
        raise ValueError("energy_fj must be finite and non-negative")
    if not math.isfinite(duration_ns) or duration_ns <= 0:
        raise ValueError("duration_ns must be finite and positive")
    return energy_fj / duration_ns


def access_power_mw(energy_nj: float, duration_ns: float) -> float:
    return access_power_uw(energy_nj * 1e6, duration_ns) / 1000.0
