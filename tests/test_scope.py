import json
import unittest
from pathlib import Path

import scope


ROOT = Path(__file__).resolve().parents[1]


def metrics(latency: float, energy: float, leakage: float = 0.01) -> scope.DestinyMetrics:
    return scope.DestinyMetrics(
        capacity_bytes=4096,
        associativity=1,
        line_bytes=64,
        hit_latency_ns=latency,
        miss_latency_ns=latency,
        write_latency_ns=latency,
        hit_energy_nj=energy,
        miss_energy_nj=energy,
        write_energy_nj=energy,
        leakage_power_mw=leakage,
        data_array_leakage_power_mw=0.0,
        tag_array_leakage_power_mw=leakage,
    )


def layer(name: str, latency: float, energy: float) -> scope.LayerSpec:
    value = metrics(latency, energy)
    return scope.LayerSpec(
        name=name,
        device="SRAM",
        device_family="SRAM",
        destiny_config=ROOT / "config/scope_l1_sram.cfg",
        capacity_bytes=4096,
        associativity=1,
        line_bytes=64,
        replacement_policy="LRU",
        banks=1,
        peripheral_latency_ns=0.0,
        peripheral_energy_nj=0.0,
        ber=0.0,
        ber_max=1.0,
        allow_high_variation=False,
        endurance_writes_per_line=1e30,
        wear_leveling_efficiency=1.0,
        refresh_interval_us=0.0,
        retention_time_us=0.0,
        estimated_writebacks_per_request=0.0,
        device_rows_per_bank=64,
        stacked_tiers=1,
        effective_density_f2=140.0,
        data_cell_area_f2=4096 * 8 * 140.0,
        device_leakage_power_mw=0.0,
        device_refresh_power_mw=0.0,
        device_library_entry={"variation": "low"},
        raw_metrics=value,
        metrics=value,
    )


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "config/device_library.json").open(encoding="utf-8") as stream:
            cls.library = json.load(stream)["devices"]

    def test_device_library_contains_all_screenshot_columns(self) -> None:
        self.assertEqual(
            set(self.library),
            {
                "STT-MRAM", "SOT-MRAM", "SRAM", "Si-eDRAM",
                "TFET-eDRAM", "2D-eDRAM", "OSFET-eDRAM",
            },
        )
        required = {
            "read_latency", "read_energy", "write_latency", "write_energy",
            "leakage", "refresh", "endurance", "variation", "density_f2",
            "m3d_scalability",
        }
        for device in self.library.values():
            self.assertTrue(required.issubset(device))

    def test_sram_table_leakage_is_applied_per_data_bit(self) -> None:
        leakage, refresh = scope._power_from_device_library(
            self.library["SRAM"], 32 * 1024 * 8, 0.0
        )
        self.assertAlmostEqual(leakage, 0.00720896)
        self.assertEqual(refresh, 0.0)

    def test_supplied_average_equation(self) -> None:
        links = [
            scope.CrossbarSpec("x12", 2.0, 2.0, 1.0, 400.0, 1),
            scope.CrossbarSpec("x23", 2.5, 2.5, 1.0, 500.0, 1),
        ]
        model = scope.ScopeModel(
            layers=[layer("L1", 1.0, 0.1), layer("L2", 2.0, 0.2),
                    layer("L3", 3.0, 0.3)],
            crossbars=links,
            offchip=scope.OffChipSpec(10.0, 1.0),
            workload={
                "read_fraction": 1.0,
                "memory_access_rate_per_s": 2_000_000,
                "lifetime_seconds": 1.0,
            },
            hit_rates=scope.HitRateResult(
                hit_rates=(0.5, 0.5, 0.5),
                accesses=(0, 0, 0),
                hits=(0, 0, 0),
                writebacks_per_request=(0.0, 0.0, 0.0),
                offchip_writebacks_per_request=0.0,
                observed_read_fraction=1.0,
            ),
        )
        report = model.average()
        self.assertAlmostEqual(report["average_latency_ns"], 7.25)
        self.assertAlmostEqual(report["expected_dynamic_energy_nj_per_request"], 0.725)
        self.assertAlmostEqual(report["dynamic_power_mw"], 1.45)

    def test_writeback_energy_is_included_but_not_latency(self) -> None:
        links = [
            scope.CrossbarSpec("x12", 0.0, 0.0, 1.0, 100.0, 1),
            scope.CrossbarSpec("x23", 0.0, 0.0, 1.0, 100.0, 1),
        ]
        model = scope.ScopeModel(
            layers=[layer("L1", 1.0, 0.1), layer("L2", 2.0, 0.2),
                    layer("L3", 3.0, 0.3)],
            crossbars=links,
            offchip=scope.OffChipSpec(10.0, 1.0),
            workload={
                "read_fraction": 1.0,
                "memory_access_rate_per_s": 1_000_000,
                "lifetime_seconds": 1.0,
            },
            hit_rates=scope.HitRateResult(
                hit_rates=(1.0, 0.0, 0.0),
                accesses=(0, 0, 0),
                hits=(0, 0, 0),
                writebacks_per_request=(0.0, 0.25, 0.0),
                offchip_writebacks_per_request=0.0,
                observed_read_fraction=1.0,
            ),
        )
        report = model.average()
        self.assertAlmostEqual(report["average_latency_ns"], 1.0)
        self.assertAlmostEqual(report["expected_dynamic_energy_nj_per_request"], 0.175)


if __name__ == "__main__":
    unittest.main()
