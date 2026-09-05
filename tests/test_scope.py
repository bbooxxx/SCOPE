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
        destiny_model_capacity_bytes=4096,
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
        with (ROOT / "config/device_library_v5.json").open(encoding="utf-8") as stream:
            cls.library_v5_raw = json.load(stream)
            cls.library_v5 = cls.library_v5_raw["devices"]
        cls.library_v6_raw = scope.load_model_library(
            ROOT / "config/device_library_v6.json", ROOT
        )
        cls.library_v6 = cls.library_v6_raw["devices"]
        cls.library_v7_raw = scope.load_model_library(
            ROOT / "config/device_library_v7.json", ROOT
        )
        cls.library_v7 = cls.library_v7_raw["devices"]
        cls.library_v8_raw = scope.load_model_library(
            ROOT / "config/device_library_v8.json", ROOT
        )
        cls.library_v8 = cls.library_v8_raw["devices"]

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
        effective, device_leakage, _, _, _, _ = scope._apply_device_library(
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
        effective, _, _, read, refresh, _ = scope._apply_device_library(
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

    def test_cpp_trace_measures_shape_derived_mix_and_capacity_sensitive_l3(self) -> None:
        base_layers = [
            replace(layer("L1", 1.0, 0.1), capacity_bytes=32768,
                    associativity=8),
            replace(layer("L2", 1.0, 0.1), capacity_bytes=262144,
                    associativity=8),
            replace(layer("L3", 1.0, 0.1), capacity_bytes=1048576,
                    associativity=16),
        ]
        hit_model = {
            "mode": "cpp_openvla_trace",
            "operator": "attention",
            "sampled_working_set_bytes": 2097152,
            "seed": 20260829,
            "isa_access_bytes": 16,
            "working_set_stride_bytes": 64,
            "operator_shape": {
                "sequence_tokens": 295,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "head_dim": 128,
                "tile_m": 16,
                "tile_n": 64,
                "tile_k": 32,
            },
        }
        small = scope.estimate_hit_rates(
            base_layers,
            {"read_fraction": 0.75, "hit_rate_model": hit_model},
            ROOT,
        )
        large = scope.estimate_hit_rates(
            [base_layers[0], base_layers[1],
             replace(base_layers[2], capacity_bytes=8388608)],
            {"read_fraction": 0.75, "hit_rate_model": hit_model},
            ROOT,
        )
        self.assertAlmostEqual(
            small.observed_read_fraction,
            small.trace_metadata["analytical_read_fraction"],
            delta=0.002,
        )
        self.assertAlmostEqual(
            large.observed_read_fraction,
            large.trace_metadata["analytical_read_fraction"],
            delta=0.002,
        )
        self.assertNotEqual(small.observed_read_fraction, 0.75)
        self.assertGreater(large.hit_rates[2], small.hit_rates[2])
        self.assertEqual(large.trace_metadata["isa_access_bytes"], 16)
        self.assertEqual(large.trace_metadata["cache_line_bytes"], 64)
        self.assertEqual(large.trace_metadata["vector_accesses_per_cache_line"], 4)
        self.assertIn("cache-line bursts", large.trace_metadata["trace_layout"])
        self.assertIn("compulsory_misses", large.trace_metadata)
        self.assertIn("representative_access", large.trace_metadata)
        reuse_model = dict(hit_model)
        reuse_model["cross_frame_l3_reuse_fraction"] = 0.9
        reused = scope.estimate_hit_rates(
            base_layers,
            {"read_fraction": 0.75, "hit_rate_model": reuse_model},
            ROOT,
        )
        self.assertGreater(reused.hit_rates[2], small.hit_rates[2])
        self.assertEqual(
            reused.trace_metadata["cross_frame_l3_reuse_fraction"], 0.9
        )

    def test_v5_full_layer_working_sets_exceed_l2(self) -> None:
        config = scope.select_workload(
            scope.load_json(ROOT / "config/scope_v5.json"), "attention"
        )
        model = dict(config["workload"]["hit_rate_model"])
        model.update({
            "trace_cycle_accesses": 4096,
            "warmup_accesses": 1024,
            "sample_accesses": 1024,
        })
        levels = [
            replace(layer("L1", 1.0, 0.1), capacity_bytes=256 * 1024,
                    associativity=8),
            replace(layer("L2", 1.0, 0.1), capacity_bytes=32 * 1024 * 1024,
                    associativity=16),
            replace(layer("L3", 1.0, 0.1), capacity_bytes=384 * 1024 * 1024,
                    associativity=16),
        ]
        result = scope.estimate_hit_rates(
            levels, {"read_fraction": 0.5, "hit_rate_model": model}, ROOT
        )
        working_set = result.trace_metadata["analytical_working_set_bytes"]
        self.assertEqual(working_set, 148717568)
        self.assertGreater(working_set, levels[1].capacity_bytes)
        self.assertEqual(
            result.trace_metadata["sampled_tensor_bytes"], working_set
        )
        transactions = (
            -(-result.trace_metadata["analytical_loads"] //
              result.trace_metadata["isa_accesses_per_cache_transaction"])
            + -(-result.trace_metadata["analytical_stores"] //
                result.trace_metadata["isa_accesses_per_cache_transaction"])
        )
        self.assertEqual(
            result.trace_metadata["full_layer_cache_transactions"],
            transactions,
        )
        self.assertEqual(
            result.trace_metadata["memory_access_rate_per_s"],
            transactions * result.trace_metadata["policy_frequency_hz"],
        )
        self.assertGreater(
            transactions, result.trace_metadata["trace_cycle_accesses"]
        )

    def test_v5_sa_separates_destiny_converter_and_checks_compatibility(self) -> None:
        raw = replace(
            metrics(2.0, 0.2),
            sense_amp_latency_ns=1.5,
            sense_amp_read_energy_nj=0.12,
            sense_amp_leakage_mw=0.02,
            legacy_iv_converter_latency_ns=1.0,
            legacy_iv_converter_read_energy_nj=0.08,
            legacy_iv_converter_leakage_mw=0.01,
        )
        result = scope.evaluate_sense_amp(
            raw,
            self.library_v5["SOT-MRAM"],
            self.library_v5_raw["sense_amplifier_models"],
            "current",
        )
        self.assertAlmostEqual(result.base_voltage_latency_ns, 0.5)
        self.assertAlmostEqual(result.selected_latency_ns, 0.3)
        self.assertAlmostEqual(result.selected_leakage_mw, 0.0002)
        self.assertLess(result.selected_latency_ns, result.destiny_reported_latency_ns)
        with self.assertRaises(ValueError):
            scope.evaluate_sense_amp(
                raw,
                self.library_v5["SRAM"],
                self.library_v5_raw["sense_amplifier_models"],
                "current",
            )

    def test_v5_m3d_is_generic_and_scales_with_tiers(self) -> None:
        defaults = self.library_v5_raw["m3d_defaults"]
        one = scope.evaluate_m3d(
            {"enabled": True, "tiers": 1}, defaults,
            banks=4, line_bits=512, data_array_area_mm2=1.0,
        )
        four = scope.evaluate_m3d(
            {"enabled": True, "tiers": 4}, defaults,
            banks=4, line_bits=512, data_array_area_mm2=1.0,
        )
        self.assertEqual(one.latency_penalty_ns, 0.0)
        self.assertGreater(four.latency_penalty_ns, 0.0)
        self.assertGreater(four.energy_penalty_nj, 0.0)
        self.assertGreater(four.footprint_mm2, 0.0)

    def test_v5_thor_area_budget_and_target_capacities(self) -> None:
        raw = scope.select_workload(
            scope.load_json(ROOT / "config/scope_v5.json"), "attention"
        )
        variants = scope.design_variants(raw, True, ROOT, self.library_v5)
        target = next(
            variant for variant in variants
            if [item["device"] for item in variant["layers"]]
            == ["SRAM", "TFET-eDRAM", "OSFET-eDRAM"]
        )
        self.assertEqual(
            [item["capacity_bytes"] for item in target["layers"]],
            [256 * 1024, 32 * 1024 * 1024, 384 * 1024 * 1024],
        )
        self.assertTrue(target["layers"][2]["m3d"]["enabled"])

    def test_v6_edram_nonideal_penalty_scales_with_longer_rbl(self) -> None:
        entry = self.library_v6["OSFET-eDRAM"]
        short = replace(
            metrics(10.0, 0.1), subarray_rows=256, subarray_columns=512,
            rbl_capacitance_ff=80.0, rwl_resistance_ohm=665.0,
            rbl_resistance_ohm=1942.0,
        )
        long = replace(
            short, subarray_rows=1024, rbl_capacitance_ff=320.0,
            rbl_resistance_ohm=7767.0,
        )
        sa = {"input_referred_noise_mv": 8.0}
        short_result = scope.evaluate_nonideal(
            short, 10.0, 512, entry, sa, 1e-9
        )
        long_result = scope.evaluate_nonideal(
            long, 10.0, 512, entry, sa, 1e-9
        )
        self.assertGreater(long_result.rbl_ir_drop_mv, short_result.rbl_ir_drop_mv)
        self.assertGreater(long_result.read_latency_penalty_ns,
                           short_result.read_latency_penalty_ns)
        self.assertLess(long_result.real_selected_current_ua,
                        long_result.ideal_selected_current_ua)

    def test_v6_stt_read_disturb_limits_read_not_write_current(self) -> None:
        raw = replace(
            metrics(10.0, 0.1), subarray_rows=256, subarray_columns=256,
            rwl_resistance_ohm=200.0, rbl_resistance_ohm=300.0,
        )
        result = scope.evaluate_nonideal(
            raw, 10.0, 512, self.library_v6["STT-MRAM"],
            {"input_referred_noise_mv": 8.0}, 1e-12,
        )
        self.assertTrue(result.read_disturb_current_limited)
        self.assertLessEqual(result.read_disturb_probability_per_read,
                             result.read_disturb_target_per_read * (1.0 + 1e-9))
        self.assertGreater(result.read_latency_penalty_ns, 0.0)
        self.assertIn("write current is unchanged", result.reliability_model)

    def test_v6_m3d_uses_driven_delay_not_only_tiny_intrinsic_rc(self) -> None:
        result = scope.evaluate_m3d(
            {"enabled": True, "tiers": 4},
            self.library_v6_raw["m3d_defaults"],
            banks=16, line_bits=512, data_array_area_mm2=10.0,
        )
        self.assertAlmostEqual(result.latency_penalty_ns, 0.06)
        self.assertGreater(result.latency_penalty_ns,
                           result.intrinsic_rc_latency_ns * 1000)

    def test_v6_noc_separates_request_response_router_and_wire(self) -> None:
        raw = json.loads((ROOT / "config/scope_v6.json").read_text())
        self.assertEqual({layer["line_bytes"] for layer in raw["layers"]},
                         {128})
        self.assertEqual(raw["off_chip"]["transaction_bytes"], 128)
        link = scope.build_crossbar(raw["crossbars"][0])
        self.assertAlmostEqual(link.request_latency_ns, 1.0)
        self.assertAlmostEqual(link.response_latency_ns, 1.0)
        self.assertAlmostEqual(link.latency_ns, 2.0)
        self.assertEqual(link.response_bits, 1024)
        self.assertAlmostEqual(link.energy_nj, 0.218688)

    def test_v6_library_overlay_does_not_change_v5_schema(self) -> None:
        self.assertEqual(self.library_v5_raw["schema_version"], 5)
        self.assertEqual(self.library_v6_raw["schema_version"], 6)
        self.assertNotIn("nonideal", self.library_v5["OSFET-eDRAM"])
        self.assertTrue(self.library_v6["OSFET-eDRAM"]["nonideal"]["enabled"])
        self.assertEqual(
            self.library_v6["OSFET-eDRAM"]["read_circuit"],
            self.library_v5["OSFET-eDRAM"]["read_circuit"],
        )

    def test_v7_switches_disable_only_the_requested_model_overlays(self) -> None:
        raw_metrics = replace(
            metrics(3.0, 0.2, leakage=0.4),
            subarray_rows=256,
            subarray_columns=128,
            rbl_capacitance_ff=40.0,
            rbl_wire_capacitance_ff=30.0,
            rbl_cell_capacitance_ff=10.0,
            rwl_resistance_ohm=332.0,
            rbl_resistance_ohm=1942.0,
            peripheral_read_latency_ns=1.0,
            peripheral_read_energy_nj=0.05,
            sense_amp_latency_ns=0.02,
            sense_amp_read_energy_nj=0.01,
            sense_amp_leakage_mw=0.4,
            native_sense_amp_type="voltage",
        )
        raw_layer = {
            "name": "L3",
            "device": "OSFET-eDRAM",
            "destiny_config": "config/devices/osfet_edram.cfg",
            "capacity_bytes": 4096,
            "destiny_model_capacity_bytes": 4096,
            "associativity": 1,
            "line_bytes": 64,
            "replacement_policy": "LRU",
            "banks": 1,
            "ber": 1e-9,
            "allow_high_variation": True,
            "wear_leveling_efficiency": 1.0,
            "stacked_tiers": 4,
            "m3d": {"enabled": True, "tiers": 4},
            "peripheral_latency_ns": 0.5,
            "peripheral_energy_nj": 0.1,
        }
        enabled = scope.build_layer(
            raw_layer, raw_metrics, ROOT, 1e-8, self.library_v7,
            self.library_v7_raw, scope.FEATURE_SWITCH_DEFAULTS,
        )
        disabled = scope.build_layer(
            raw_layer, raw_metrics, ROOT, 1e-8, self.library_v7,
            self.library_v7_raw,
            {
                "array_nonideal": False,
                "configurable_peripherals": False,
                "m3d": False,
            },
        )
        self.assertTrue(enabled.nonideal["enabled"])
        self.assertTrue(enabled.m3d["enabled"])
        self.assertTrue(enabled.sense_amp["configuration_enabled"])
        self.assertFalse(disabled.nonideal["enabled"])
        self.assertFalse(disabled.m3d["enabled"])
        self.assertFalse(disabled.sense_amp["configuration_enabled"])
        self.assertEqual(disabled.peripheral_latency_ns, 0.0)
        self.assertEqual(disabled.peripheral_energy_nj, 0.0)
        self.assertLess(disabled.metrics.hit_latency_ns,
                        enabled.metrics.hit_latency_ns)
        self.assertGreater(disabled.metrics.leakage_power_mw,
                           enabled.metrics.leakage_power_mw)

    def test_v7_evaluation_cases_cover_requested_architectures_and_ablations(self) -> None:
        raw = scope.select_workload(
            scope.load_json(ROOT / "config/scope_v7.json"), "attention"
        )
        cases = scope.build_evaluation_cases(raw)
        self.assertEqual(
            [case["id"] for case in cases],
            [
                "all_sram", "all_osfet", "optimized",
                "optimized_array_nonideal_off",
                "optimized_configurable_peripherals_off",
                "optimized_m3d_off",
            ],
        )
        all_osfet = next(case for case in cases if case["id"] == "all_osfet")
        variants = scope.design_variants(
            all_osfet["config"], False, ROOT, self.library_v7
        )
        self.assertEqual(
            [layer["capacity_bytes"] for layer in variants[0]["layers"]],
            [4 * 1024 * 1024, 128 * 1024 * 1024, 384 * 1024 * 1024],
        )
        peripheral_off = next(
            case for case in cases
            if case["id"] == "optimized_configurable_peripherals_off"
        )
        self.assertEqual(
            len(scope.design_variants(
                peripheral_off["config"], True, ROOT, self.library_v7
            )),
            1,
        )
        self.assertEqual(scope.resolve_feature_switches({}),
                         scope.FEATURE_SWITCH_DEFAULTS)

    def test_v8_osfet_bti_is_available_but_refresh_is_disabled(self) -> None:
        bti = scope.evaluate_bti_retention(
            self.library_v8["OSFET-eDRAM"]["bti"],
            read_vgs_v=1.2,
            initial_vth_v=0.2,
            current_alpha=1.3,
        )
        self.assertAlmostEqual(bti.vth_max_mv, 70.0)
        self.assertAlmostEqual(
            bti.reference_equivalent_retention_s, 16913.9235815
        )
        self.assertGreater(bti.total_acceleration, 7.0e5)
        self.assertAlmostEqual(bti.equivalent_retention_s, 0.0222718334)
        self.assertLess(bti.refresh_interval_s, bti.equivalent_retention_s)
        self.assertLess(bti.shift_at_refresh_mv, bti.vth_max_mv)
        self.assertGreater(bti.read_latency_guardband, 1.0)

        raw = replace(
            metrics(2.0, 0.2),
            subarray_rows=256,
            subarray_columns=128,
            rbl_capacitance_ff=40.0,
            rbl_wire_capacitance_ff=30.0,
            rbl_cell_capacitance_ff=10.0,
            peripheral_read_latency_ns=0.5,
            peripheral_read_energy_nj=0.01,
        )
        effective, _, refresh_power, read, refresh, fitted = \
            scope._apply_device_library(
                raw, self.library_v8["OSFET-eDRAM"],
                4096, 64, 1, 0.0, 0.0,
            )
        self.assertFalse(refresh["enabled"])
        self.assertEqual(refresh_power, 0.0)
        self.assertIsNone(fitted)
        self.assertAlmostEqual(effective.hit_latency_ns,
                               read["total_read_latency_ns"])

    def test_v8_config_keeps_128b_lines_and_bti_library(self) -> None:
        raw = scope.select_workload(
            scope.load_json(ROOT / "config/scope_v8.json"), "attention"
        )
        self.assertEqual(raw["schema_version"], 8)
        self.assertEqual({layer["line_bytes"] for layer in raw["layers"]}, {128})
        self.assertEqual(
            raw["workload"]["hit_rate_model"]["cache_transaction_bytes"], 128
        )
        self.assertEqual(raw["workload"]["hit_rate_model"]["isa_access_bytes"], 16)
        self.assertEqual(raw["workload"]["hit_rate_model"]["trace_cycle_accesses"], 0)
        self.assertEqual(raw["workload"]["hit_rate_model"]["sample_accesses"], 0)
        self.assertEqual(
            raw["workload"]["hit_rate_model"]["cross_frame_l3_reuse_fraction"],
            0.88,
        )
        self.assertEqual(raw["layers"][2]["banks"], 64)
        self.assertEqual(raw["device_library"], "config/device_library_v8.json")
        self.assertEqual(
            raw["layers"][2]["device_overrides"]["OSFET-eDRAM"]
               ["circuit_calibration"]["read_latency_scale"],
            0.15,
        )
        self.assertEqual(
            self.library_v8["OSFET-eDRAM"]["refresh"]["model"],
            "none",
        )
        self.assertEqual(
            self.library_v8["SOT-MRAM"]["nonideal"]["ber_model"],
            "nominal",
        )
        mram_nonideal = scope.evaluate_nonideal(
            replace(
                metrics(2.0, 0.2),
                subarray_rows=256,
                subarray_columns=128,
                rwl_resistance_ohm=332.0,
                rbl_resistance_ohm=1942.0,
            ),
            2.0,
            1024,
            self.library_v8["SOT-MRAM"],
            {"input_referred_noise_mv": 8.0},
            1e-9,
        )
        self.assertEqual(mram_nonideal.effective_ber, 1e-9)
        sot_floor = self.library_v8["SOT-MRAM"]["cache_static_power_floor"]
        self.assertEqual(sot_floor["model"], "sram_reference_ratio")
        self.assertAlmostEqual(sot_floor["ratio_to_same_capacity_sram"], 0.2171)
        self.assertAlmostEqual(
            scope._cache_static_power_floor_mw(
                self.library_v8["SOT-MRAM"], 32 * 1024 * 1024
            ),
            32 * 1024 * 1024 * 27.5 * 0.2171 * 1e-9,
        )

    def test_v8_acceptance_guards_l1_hit_rate(self) -> None:
        config = scope.load_json(ROOT / "config/scope_v8.json")
        summaries = [
            {
                "case": "all_sram", "average_latency_ns": 20.0,
                "fom_per_ns_mw": 1.0,
            },
            {
                "case": "sram_osfet_osfet", "average_latency_ns": 8.0,
                "fom_per_ns_mw": 2.0,
            },
            {
                "case": "sram_mram_mram", "average_latency_ns": 10.0,
                "fom_per_ns_mw": 1.5,
            },
            {
                "case": "optimized", "average_latency_ns": 5.0,
                "fom_per_ns_mw": 3.0,
                "conditional_hit_rates": [0.5, 0.8, 0.9],
            },
        ]
        checks = scope.evaluation_acceptance(config, summaries)
        self.assertTrue(checks)
        self.assertTrue(all(item["pass"] for item in checks))
        summaries[3]["fom_per_ns_mw"] = 2.5
        with self.assertRaises(scope.ScopeError):
            scope.evaluation_acceptance(config, summaries)

    def test_v4_uses_orin_lpddr5_and_screened_search_space(self) -> None:
        config = json.loads((ROOT / "config/scope_v4.json").read_text())
        offchip = config["off_chip"]
        self.assertEqual(offchip["standard"], "LPDDR5-6400")
        self.assertEqual(offchip["bandwidth_gbps"], 204.8)
        self.assertEqual(offchip["bus_width_bits"], 256)
        self.assertAlmostEqual(
            offchip["energy_pj_per_bit"] * offchip["transaction_bytes"] * 8 / 1000,
            1.28,
        )
        self.assertEqual(
            config["layers"][0]["devices"],
            ["SRAM", "SOT-MRAM", "TFET-eDRAM"],
        )
        self.assertEqual(
            config["layers"][0]["device_overrides"]["TFET-eDRAM"]
                  ["peripheral_latency_ns"],
            0.25,
        )
        self.assertIn("TFET-eDRAM", config["layers"][1]["devices"])
        self.assertIn("OSFET-eDRAM", config["layers"][2]["devices"])
        resolved = scope.select_workload(
            scope.load_json(ROOT / "config/scope_v4.json"), "attention"
        )
        variants = scope.design_variants(
            resolved, True, ROOT, self.library_v3
        )
        self.assertEqual(len(variants), 48)
        tfet_l1 = next(
            variant["layers"][0] for variant in variants
            if variant["layers"][0]["device"] == "TFET-eDRAM"
        )
        self.assertEqual(tfet_l1["peripheral_latency_ns"], 0.25)

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
