import json
import unittest
from dataclasses import replace
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
        destiny_config=ROOT / "config/devices/sram.cfg",
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
        with (ROOT / "config/device_library_v3.json").open(encoding="utf-8") as stream:
            cls.library_v3 = json.load(stream)["devices"]

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
            "m3d_scalability", "destiny_cfg",
        }
        for device in self.library.values():
            self.assertTrue(required.issubset(device))

    def test_seven_cell_and_cfg_files_contain_all_device_metrics(self) -> None:
        for entry in self.library.values():
            cfg = ROOT / entry["destiny_cfg"]
            self.assertTrue(cfg.is_file())
            text = cfg.read_text(encoding="utf-8")
            cell_name = next(
                line.split(":", 1)[1].strip()
                for line in text.splitlines()
                if line.startswith("-MemoryCellInputFile:")
            )
            cell_text = (cfg.parent / cell_name).read_text(encoding="utf-8")
            for marker in (
                "ScopeDevice", "ScopeRead", "ScopeWrite", "ScopeLeakage",
                "ScopeRefresh", "ScopeEndurance", "ScopeVariation",
                "ScopeDensity", "ScopeM3DScalability",
            ):
                self.assertIn(marker, text)
                self.assertIn(marker, cell_text)

    def test_sram_table_leakage_is_applied_per_data_bit(self) -> None:
        leakage, refresh = scope._power_from_device_library(
            self.library["SRAM"], 32 * 1024 * 8, 0.0
        )
        self.assertAlmostEqual(leakage, 0.00720896)
        self.assertEqual(refresh, 0.0)

    def test_effective_static_power_does_not_reuse_raw_tag_leakage(self) -> None:
        raw = metrics(1.0, 0.1, leakage=250.0)
        effective, device_leakage, _, _, _ = scope._apply_device_library(
            raw, self.library["SOT-MRAM"], 4096, 64, 1, 0.0, 0.0
        )
        self.assertEqual(device_leakage, 0.0)
        self.assertEqual(effective.leakage_power_mw, 0.0)
        self.assertEqual(effective.tag_array_leakage_power_mw, 0.0)
        self.assertEqual(raw.tag_array_leakage_power_mw, 250.0)
        edram_leakage, edram_refresh = scope._power_from_device_library(
            self.library["TFET-eDRAM"], 4 * 1024 * 1024 * 8, 10.0
        )
        self.assertEqual(edram_leakage, 0.0)
        self.assertGreater(edram_refresh, 0.0)

    def test_v3_edram_read_uses_rbl_equation_and_requested_pure_paths(self) -> None:
        raw = replace(
            metrics(1.0, 0.1),
            subarray_rows=256,
            subarray_columns=512,
            rbl_capacitance_ff=40.0,
            rbl_wire_capacitance_ff=30.0,
            rbl_cell_capacitance_ff=10.0,
            peripheral_read_latency_ns=0.5,
            peripheral_read_energy_nj=0.01,
        )
        effective, _, _, read, refresh = scope._apply_device_library(
            raw, self.library_v3["Si-eDRAM"], 4096, 64, 1, 10.0, 10.0
        )
        self.assertAlmostEqual(read["cell_read_latency_ns"], 40e-15 * 0.1 / 33.1e-6 * 1e9)
        self.assertAlmostEqual(read["read_energy_fj_per_bit"], 4.6)
        self.assertAlmostEqual(effective.hit_latency_ns,
                               read["cell_read_latency_ns"] + 0.5)
        self.assertTrue(refresh["enabled"])
        self.assertEqual(self.library_v3["2D-eDRAM"]["read_latency"]["selected_path"],
                         "pure")
        self.assertEqual(self.library_v3["OSFET-eDRAM"]["read_latency"]["selected_path"],
                         "pure")
        self.assertEqual(self.library_v3["2D-eDRAM"]["read_circuit"]["path"],
                         "pure 2D-FET (d)")
        self.assertEqual(self.library_v3["OSFET-eDRAM"]["read_circuit"]["path"],
                         "pure a-IGZO OSFET (o)")

    def test_si_refresh_can_consume_more_than_all_bank_bandwidth(self) -> None:
        result = scope.evaluate_si_refresh(
            capacity_bytes=32 * 1024 * 1024,
            banks=4,
            nrow=256,
            ncolumn=512,
            read_energy_fj_per_bit=4.6,
            write_energy_fj_per_bit=2.5,
            read_latency_ns=0.62,
            write_latency_ns=2.0,
            refresh_interval_us=10.0,
            retention_time_us=10.0,
        )
        self.assertGreater(result.bandwidth_occupancy, 1.0)
        self.assertFalse(result.schedulable)
        half_busy = replace(layer("L1", 1.0, 0.1), refresh={
            "enabled": True, "bandwidth_occupancy": 0.5,
        })
        self.assertEqual(scope.ScopeModel._refresh_adjusted_latency(half_busy, 2.0),
                         4.0)

    def test_openvla_trace_derives_rho_from_attention_and_ffn_events(self) -> None:
        config = json.loads((ROOT / "config/scope_v3.json").read_text())
        attention = scope.build_trace(
            config["workloads"]["attention"]["hit_rate_model"], 64
        )
        ffn = scope.build_trace(config["workloads"]["ffn"]["hit_rate_model"], 64)
        self.assertEqual(attention.reads + attention.writes, len(attention.events))
        self.assertEqual(ffn.reads + ffn.writes, len(ffn.events))
        self.assertAlmostEqual(attention.read_fraction,
                               attention.reads / len(attention.events))
        self.assertAlmostEqual(ffn.read_fraction, ffn.reads / len(ffn.events))
        self.assertGreater(attention.read_fraction, 0.5)
        self.assertGreater(ffn.read_fraction, 0.5)
        self.assertFalse(attention.metadata["hardware_trace_available"])

    def test_osfet_endurance_is_not_limiting_in_v3(self) -> None:
        endurance = self.library_v3["OSFET-eDRAM"]["endurance"]
        self.assertEqual(endurance["model"], "not_limiting")
        self.assertEqual(endurance["writes_per_line"], 1e30)
        for stem in ("2d_edram", "osfet_edram"):
            for suffix in ("cfg", "cell"):
                text = (ROOT / f"config/devices/{stem}.{suffix}").read_text()
                self.assertIn("-ScopeSelectedReadPath: pure", text)
        for stem in ("tfet_edram", "2d_edram", "osfet_edram"):
            for suffix in ("cfg", "cell"):
                text = (ROOT / f"config/devices/{stem}.{suffix}").read_text()
                self.assertIn("-ScopeRefreshPower: not_applicable_in_v3", text)

    def test_operator_hit_rates_are_capacity_aware_and_reasonable(self) -> None:
        layers = [
            replace(layer("L1", 1.0, 0.1), capacity_bytes=32768,
                    associativity=8),
            replace(layer("L2", 1.0, 0.1), capacity_bytes=1048576,
                    associativity=8),
            replace(layer("L3", 1.0, 0.1), capacity_bytes=4194304,
                    associativity=16),
        ]
        result = scope.estimate_hit_rates(
            layers,
            {
                "read_fraction": 0.82,
                "hit_rate_model": {
                    "mode": "operator",
                    "reuse_window_bytes": {
                        "L1": 65536, "L2": 2097152, "L3": 8388608,
                    },
                    "locality_factor": 2.0,
                    "reference_associativity": {
                        "L1": 8, "L2": 8, "L3": 16,
                    },
                },
            },
            ROOT,
        )
        for rate in result.hit_rates:
            self.assertGreaterEqual(rate, 0.60)
            self.assertLessEqual(rate, 0.80)
        larger = replace(layers[1], capacity_bytes=4 * 1048576)
        larger_result = scope.estimate_hit_rates(
            [layers[0], larger, layers[2]],
            {
                "read_fraction": 0.82,
                "hit_rate_model": {
                    "mode": "operator",
                    "reuse_window_bytes": [65536, 2097152, 8388608],
                    "locality_factor": 2.0,
                    "reference_associativity": [8, 8, 16],
                },
            },
            ROOT,
        )
        self.assertGreater(larger_result.hit_rates[1], result.hit_rates[1])

    def test_guidance_and_density_scaled_requested_mapping(self) -> None:
        report = {
            "average_latency_ns": 10.0,
            "average_power_mw": 20.0,
            "expected_dynamic_energy_nj_per_request": 2.0,
        }
        balanced = scope.guidance_score(
            report, {"weights": {"latency": 1, "power": 1}}
        )
        latency_heavy = scope.guidance_score(
            report, {"weights": {"latency": 2, "power": 1}}
        )
        self.assertAlmostEqual(balanced["score"], 1 / 200)
        self.assertAlmostEqual(latency_heavy["score"], 1 / 2000)
        self.assertEqual(
            scope._guided_destiny_target(
                {"weights": {"latency": 2, "power": 1}}, 0.82
            ),
            "ReadLatency",
        )
        raw = scope.select_workload(
            scope.load_json(ROOT / "config/scope_v2_requested.json"), None
        )
        variants = scope.design_variants(raw, False, ROOT, self.library)
        self.assertEqual(len(variants), 1)
        self.assertEqual(
            [item["capacity_bytes"] for item in variants[0]["layers"]],
            [32768, 4 * 1048576, 16 * 4194304],
        )

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
