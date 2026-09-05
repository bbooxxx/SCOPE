import math
import unittest
import json
from pathlib import Path

from scope.m3d import evaluate_m3d
from scope.core import load_model_library, select_workload, design_variants
from scripts.v9_reports import relative_fom_gain


class M3DTests(unittest.TestCase):
    def test_report_gain_uses_inverse_product_ratio(self):
        self.assertAlmostEqual(relative_fom_gain(1.1, 1.0), 0.1)
        self.assertAlmostEqual(relative_fom_gain(0.9, 1.0), -0.1)
        for candidate, baseline in ((0, 1), (1, 0), (-1, 1)):
            with self.assertRaises(ValueError):
                relative_fom_gain(candidate, baseline)

    def test_search_targets_reach_destiny_input(self):
        root = Path(__file__).resolve().parents[1]
        raw = select_workload(json.loads((root / 'config/scope_v9.json').read_text()), 'attention')
        library = load_model_library(root / 'config/device_library_v9.json', root)
        variants = design_variants(raw, True, root, library['devices'])
        targets = set()
        for variant in variants:
            layer = variant['layers'][1]
            target = layer['destiny_optimization_target']
            targets.add(target)
            self.assertIn('-OptimizationTarget: ' + target,
                          (root / layer['destiny_config']).read_text())
            self.assertEqual(layer['capacity_bytes'], 4 * 1048576)
        self.assertEqual(targets, {'ReadEDP', 'ReadDynamicEnergy', 'LeakagePower'})
        self.assertEqual({v['layers'][2]['destiny_optimization_target'] for v in variants},
                         targets)
        original_paths = {v['layers'][1]['destiny_config'] for v in variants}
        raw['layers'][1]['line_bytes'] = 64
        other = design_variants(raw, True, root, library['devices'])
        self.assertTrue(original_paths.isdisjoint(
            {v['layers'][1]['destiny_config'] for v in other}))
        for path in original_paths:
            self.assertIn('-WordWidth (bit): 1024', (root / path).read_text())

    def run_model(self, **overrides):
        config = dict(model="elmore_staircase", enabled=True, tiers=4,
                      via_resistance_ohm=5.5, via_capacitance_ff=0.1,
                      via_pitch_um=0.2, read_swing_v=0.1, write_swing_v=1.2,
                      read_active_lines=1, write_active_lines=1,
                      array_count=1, array_height_um=100, array_width_um=200)
        config.update(overrides)
        return evaluate_m3d(config, {}, banks=1, line_bits=128,
                            data_array_area_mm2=0.02)

    def test_ladder_and_staircase(self):
        result = self.run_model()
        self.assertAlmostEqual(result.per_tier[3]["latency_ns"], 0.69*5.5*0.1e-6*6)
        self.assertAlmostEqual(result.latency_penalty_ns, 0.69*5.5*0.1e-6*2.5)
        self.assertAlmostEqual(result.footprint_mm2, (4*16*0.2**2+2*4*0.2*300)/1e6)

    def test_nonuniform_ladder(self):
        result = self.run_model(segment_resistance_ohm=[1, 2, 4],
                                segment_capacitance_ff=[3, 5, 7])
        self.assertAlmostEqual(result.per_tier[3]["latency_ns"], 0.69*(3+3*5+7*7)*1e-6)

    def test_energy_units_and_read_write(self):
        result = self.run_model(read_current_ua=10, read_pulse_ns=2)
        expected = 0.5*3*0.1e-15*0.1**2 + (10e-6)**2*3*5.5*2e-9
        self.assertAlmostEqual(result.per_tier[3]["read_energy_nj"]*1e-9, expected, delta=1e-25)
        self.assertGreater(result.write_energy_penalty_nj, result.energy_penalty_nj)

    def test_disabled_and_invalid(self):
        result = self.run_model(enabled=False)
        self.assertEqual((result.latency_penalty_ns, result.energy_penalty_nj,
                          result.write_energy_penalty_nj, result.footprint_mm2), (0, 0, 0, 0))
        for values in ({"read_current_ua": 1}, {"via_resistance_ohm": -1},
                       {"tier_probabilities": [1, 1, 1, 1]},
                       {"segment_resistance_ohm": [1]}, {"via_pitch_um": math.nan}):
            with self.assertRaises(ValueError):
                self.run_model(**values)
