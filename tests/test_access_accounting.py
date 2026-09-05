import json
import math
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import scope
from scope.power import access_power_mw, access_power_uw, legacy_v9_activity_counts
from test_scope import layer

ROOT = Path(__file__).resolve().parents[1]


def run_trace(operator="ffn", **overrides):
    options = {"operator": operator, "trace-kind": "tiled", "group-m": 2,
               "capacities": "128,512,2048", "associativities": "2,2,4",
               "policies": "LRU,LRU,LRU", "line-bytes": 16, "transaction-bytes": 16,
               "access-bytes": 4, "working-set-stride-bytes": 16,
               "sequence-tokens": 5, "hidden-size": 16, "attention-heads": 2,
               "head-dimension": 8, "intermediate-size": 24,
               "tile-m": 2, "tile-n": 8, "tile-k": 8,
               "warmup-accesses": 0, "dump-first": 100000}
    options.update(overrides)
    command = [str(ROOT / "scope_model")]
    for key, value in options.items():
        command.extend(["--" + key, str(value)])
    return json.loads(subprocess.check_output(command, text=True))


class TiledTraceTests(unittest.TestCase):
    def test_ffn_addresses_match_independent_nested_loops(self):
        report = run_trace()
        base, tensors = 0x100000000, {}
        shapes = [("input", 5, 16), ("Wgate", 24, 16), ("Wup", 24, 16),
                  ("Wdown", 16, 24), ("gate", 5, 24), ("up", 5, 24),
                  ("SwiGLU", 5, 24), ("output", 5, 16)]
        for name, rows, cols in shapes:
            tensors[name] = base
            base += math.ceil(rows * cols * 2 / 4096) * 4096
        self.assertEqual([t["base"] for t in report["tensors"]], list(tensors.values()))
        expected = []

        def block(op, start, stride, rows, width, phase):
            if stride == width:
                rows, width = 1, rows * width
            for row in range(rows):
                first = (start + row * stride) // 16 * 16
                last = start + row * stride + width
                for address in range(first, last, 16):
                    expected.append(dict(op=op, address=address, size_bytes=16, phase=phase))

        def gemm(a, w, c, m, n, k, phase):
            for group in range(0, math.ceil(m/2), 2):
                for col in range(0, n, 8):
                    for mt in range(group, min(math.ceil(m/2), group + 2)):
                        row, rows, cols = mt*2, min(2, m-mt*2), min(8, n-col)
                        for kk in range(0, k, 8):
                            inner = min(8, k-kk)
                            block("load", tensors[a]+(row*k+kk)*2, k*2, rows, inner*2, phase)
                            block("load", tensors[w]+(col*k+kk)*2, k*2, cols, inner*2, phase)
                        block("store", tensors[c]+(row*n+col)*2, n*2, rows, cols*2, phase)

        gemm("input", "Wgate", "gate", 5, 24, 16, 0)
        gemm("input", "Wup", "up", 5, 24, 16, 1)
        for row in range(0, 5, 2):
            for name, op in (("gate", "load"), ("up", "load"), ("SwiGLU", "store")):
                block(op, tensors[name]+row*24*2, 48, min(2, 5-row), 48, 2)
        gemm("SwiGLU", "Wdown", "output", 5, 16, 24, 3)
        self.assertEqual(report["debug_accesses"], expected)
        self.assertEqual(report["trace_cycle_accesses"], len(expected))

    def test_both_operators_conserve_paths_and_fills(self):
        for operator in ("attention", "ffn"):
            report = run_trace(operator)
            count = report["sample_accesses"]
            self.assertEqual(sum(report["load_hits"])+report["offchip_loads"]+
                             sum(report["store_hits"])+report["offchip_stores"], count)
            for level in range(3):
                self.assertEqual(report["hits"][level], report["load_hits"][level]+report["store_hits"][level])
                self.assertEqual(report["fills"][level], report["accesses"][level]-report["hits"][level])
                if level < 2:
                    self.assertEqual(report["fills"][level], report["accesses"][level+1])
            self.assertEqual(sum(p["loads"]+p["stores"] for p in report["phases"]), count)
            self.assertEqual(sum(t["bytes"] for t in report["tensors"]), report["analytical_working_set_bytes"])
            self.assertEqual(sum(p["measured_loads"] for p in report["phases"]),
                             sum(e["op"] == "load" for e in report["debug_accesses"]))
            for event in report["debug_accesses"]:
                self.assertTrue(any(t["base"] <= event["address"] < t["base"]+t["bytes"] for t in report["tensors"]))
            if operator == "attention":
                self.assertIn("causal FlashAttention", report["phases"][4]["name"])
                self.assertFalse(any("score" in t["name"] for t in report["tensors"]))

    def test_cache_replay_matches_independent_lru_reference(self):
        report = run_trace()
        caches = [scope.SetAssociativeCache(c, 16, a, "LRU", __import__("random").Random(7))
                  for c, a in ((128, 2), (512, 2), (2048, 4))]
        hits, accesses, wbs, paths = [0]*3, [0]*3, [0]*4, [[0]*4 for _ in range(2)]

        def insert(level, line, dirty, wb=False):
            if wb:
                wbs[level] += 1
            if level == 3:
                return
            evicted = caches[level].insert(line, dirty)
            if evicted and evicted.dirty:
                insert(level+1, evicted.line, True, True)

        for event in report["debug_accesses"]:
            address, store = event["address"]//16, event["op"] == "store"
            missed, destination = [], 3
            for i, cache in enumerate(caches):
                accesses[i] += 1
                if cache.probe(address, mark_dirty=store and i == 0):
                    hits[i] += 1
                    destination = i
                    break
                missed.append(i)
            paths[int(store)][destination] += 1
            for i in reversed(missed):
                insert(i, address, store and i == 0)
        self.assertEqual(hits, report["hits"])
        self.assertEqual(accesses, report["accesses"])
        self.assertEqual(paths[0], report["load_hits"]+[report["offchip_loads"]])
        self.assertEqual(paths[1], report["store_hits"]+[report["offchip_stores"]])
        for i in range(3):
            self.assertAlmostEqual(wbs[i]/report["sample_accesses"], report["writebacks_per_request"][i])
        self.assertAlmostEqual(wbs[3]/report["sample_accesses"], report["offchip_writebacks_per_request"])

    def test_grouping_changes_order_not_traffic_or_tensor_domain(self):
        grouped, row_major = run_trace(), run_trace(**{"group-m": 1})
        self.assertEqual(grouped["tensors"], row_major["tensors"])
        self.assertEqual(grouped["trace_cycle_accesses"], row_major["trace_cycle_accesses"])
        self.assertNotEqual(grouped["debug_accesses"], row_major["debug_accesses"])

    def test_hash_changes_mapping_not_workload(self):
        linear, hashed = run_trace(), run_trace(indexing="xor_fold")
        self.assertEqual(linear["debug_accesses"], hashed["debug_accesses"])
        self.assertNotEqual(linear["hit_rates"], hashed["hit_rates"])


class LegacyActivityTests(unittest.TestCase):
    def test_original_v9_counts_match_cpp_and_known_totals(self):
        shape = dict(sequence_tokens=295, hidden_size=4096,
                     intermediate_size=11008, tile_m=16)
        for operator, expected in (("attention", 1917056), ("ffn", 2373136)):
            counts = legacy_v9_activity_counts(operator, shape)
            self.assertEqual(counts["total"], expected)
            report = run_trace(operator, **{"trace-kind": "legacy_synthetic",
                "sequence-tokens": 295, "hidden-size": 4096, "intermediate-size": 11008,
                "tile-m": 16, "access-bytes": 16, "transaction-bytes": 128,
                "line-bytes": 128, "capacities": "1024,4096,8192",
                "sampled-working-set-bytes": 4096, "sample-accesses": 1, "dump-first": 1})
            self.assertEqual(counts["loads"], (report["analytical_loads"] + 7) // 8)
            self.assertEqual(counts["stores"], (report["analytical_stores"] + 7) // 8)

    def test_power_activity_and_frequency_do_not_change_trace(self):
        from scope.core import _cpp_openvla_hit_rates
        layers = [SimpleNamespace(capacity_bytes=c, line_bytes=16, associativity=a,
                                  replacement_policy="LRU")
                  for c, a in ((128, 2), (512, 2), (2048, 4))]
        model = dict(operator="ffn", trace_kind="tiled", sampled_working_set_bytes=0,
                     isa_access_bytes=4, cache_transaction_bytes=16, group_m=2,
                     policy_frequency_hz=5, power_activity_model="trace_transactions",
                     operator_shape=dict(sequence_tokens=5, hidden_size=16,
                                         intermediate_size=24, tile_m=2, tile_n=8, tile_k=8))
        full = _cpp_openvla_hit_rates(layers, model, ROOT)
        old = _cpp_openvla_hit_rates(layers, dict(model, power_activity_model="legacy_v9_analytical"), ROOT)
        faster = _cpp_openvla_hit_rates(layers, dict(model, policy_frequency_hz=10), ROOT)
        for other in (old, faster):
            self.assertEqual(full.hit_rates, other.hit_rates)
            self.assertEqual(full.load_path_counts, other.load_path_counts)
            self.assertEqual(full.store_path_counts, other.store_path_counts)
            self.assertEqual(full.trace_metadata["sample_accesses"], other.trace_metadata["sample_accesses"])
        self.assertEqual(faster.trace_metadata["memory_access_rate_per_s"],
                         2 * full.trace_metadata["memory_access_rate_per_s"])
        self.assertLess(old.trace_metadata["memory_access_rate_per_s"],
                        full.trace_metadata["memory_access_rate_per_s"])
        self.assertTrue(old.trace_metadata["measured_complete_operator"])

    def test_invalid_activity_inputs_are_rejected(self):
        shape = dict(sequence_tokens=5, hidden_size=16, intermediate_size=24, tile_m=2)
        for operator, changed, kwargs in (("other", shape, {}),
                ("ffn", dict(shape, tile_m=0), {}), ("ffn", dict(shape, hidden_size=1.5), {}),
                ("ffn", shape, {"transaction_bytes": 17})):
            with self.assertRaises(ValueError):
                legacy_v9_activity_counts(operator, changed, **kwargs)


class AccessPowerTests(unittest.TestCase):
    def model(self, frequency=1e6, power_metric="single_access_dynamic", **kwargs):
        return scope.ScopeModel(
            [layer("L1", 1, .1), layer("L2", 2, .2), layer("L3", 3, .3)],
            [scope.CrossbarSpec("12", 1, 1, 1, 100, 1), scope.CrossbarSpec("23", 1, 1, 1, 100, 1)],
            scope.OffChipSpec(10, 1),
            dict(memory_access_rate_per_s=frequency, power_metric=power_metric, lifetime_seconds=1),
            scope.HitRateResult((.5,.5,.5), (8,4,2), (4,2,1), (0,0,0), 0, .5,
                                load_path_counts=(4,0,0,0), store_path_counts=(0,2,1,1)), **kwargs)

    def test_femtojoule_example_and_bit_scaling(self):
        self.assertEqual(access_power_uw(20, 1), 20)
        self.assertEqual(access_power_uw(20*1024, 1), 20480)
        self.assertAlmostEqual(access_power_mw(20*1024/1e6, 1), 20.48)
        for energy, duration in ((-1,1),(1,0),(math.inf,1),(1,math.nan)):
            with self.assertRaises(ValueError): access_power_uw(energy, duration)

    def test_system_power_does_not_reward_slower_background_writes(self):
        model = self.model(power_metric="system_average")
        before = model.average()
        last = model.layers[-1]
        model.layers = model.layers[:-1] + (replace(last, metrics=replace(
            last.metrics, write_latency_ns=last.metrics.write_latency_ns + 50)),)
        after = model.average()
        self.assertGreater(after["serialized_service_time_ns_per_request"],
                           before["serialized_service_time_ns_per_request"])
        for metric in ("average_latency_ns", "average_power_mw", "fom_per_ns_mw",
                       "expected_dynamic_energy_nj_per_request"):
            self.assertAlmostEqual(after[metric], before[metric])
        self.assertAlmostEqual(before["average_power_mw"],
                               before["dynamic_power_mw"] + before["static_power_mw"]
                               + before["refresh_power_mw"])

    def test_joint_paths_use_actual_store_distribution(self):
        model = self.model()
        result = model.average()
        expected = .5*model.instruction("load","L1")["latency_ns"]
        for level, probability in (("L2",.25),("L3",.125),("OFF",.125)):
            expected += probability * model.instruction("store",level)["latency_ns"]
        self.assertAlmostEqual(result["average_latency_ns"], expected)
        self.assertAlmostEqual(sum(result["global_hit_probabilities"])+result["offchip_reach_probability"], 1)

    def test_access_power_independent_of_invocation_frequency(self):
        a, b = self.model(1e3).average(), self.model(1e6).average()
        self.assertEqual(a["average_power_mw"], b["average_power_mw"])
        self.assertEqual(a["fom_per_ns_mw"], b["fom_per_ns_mw"])
        self.assertAlmostEqual(b["dynamic_power_mw"] / a["dynamic_power_mw"], 1000)
        self.assertGreater(b["system_average_power_mw"], a["system_average_power_mw"])
        self.assertAlmostEqual(a["average_power_mw"], access_power_mw(a["expected_dynamic_energy_nj_per_request"], a["serialized_service_time_ns_per_request"]))

    def test_buffered_fill_energy_uses_physical_service_window(self):
        model = self.model()
        path = model.instruction("load", "OFF")
        self.assertGreater(path["serialized_service_time_ns"], path["latency_ns"])
        self.assertAlmostEqual(path["access_dynamic_power_mw"], access_power_mw(path["dynamic_energy_nj"], path["serialized_service_time_ns"]))
        blocked = self.model(refill_on_critical_path=True).instruction("load", "OFF")
        self.assertAlmostEqual(blocked["latency_ns"], path["serialized_service_time_ns"])
        self.assertEqual(blocked["dynamic_energy_nj"], path["dynamic_energy_nj"])

    def test_legacy_system_power_unit_is_not_relabelled(self):
        report = self.model(power_metric="system_average").average()
        self.assertEqual(report["average_power_mw"], report["system_average_power_mw"])
        self.assertEqual(report["average_power_uw"], 1000*report["average_power_mw"])

    def test_sot_backend_cell_uses_library_energy(self):
        from scope.core import design_variants, load_model_library, select_workload
        raw = select_workload(json.loads((ROOT/"config/scope_v9.json").read_text()), "attention")
        raw["layers"][1]["devices"] = ["SOT-MRAM"]
        library = load_model_library(ROOT/"config/device_library_v9.json", ROOT)
        variants = design_variants(raw, True, ROOT, library["devices"])
        path = ROOT/variants[0]["layers"][1]["destiny_config"]
        cell_name = next(line.split(":",1)[1].strip() for line in path.read_text().splitlines() if line.startswith("-MemoryCellInputFile:"))
        cell = (path.parent/cell_name).read_text()
        self.assertIn("-ResetEnergy (pJ): 0.102", cell)
        self.assertIn("-SetEnergy (pJ): 0.102", cell)
        self.assertIn("-ResetPulse (ns): 10.0", cell)


if __name__ == "__main__":
    unittest.main()
