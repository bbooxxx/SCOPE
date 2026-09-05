import json
from pathlib import Path
import unittest

from scripts.v9_sensitivity import apply_parameters, scenarios


class SensitivityTests(unittest.TestCase):
    def test_global_device_update_preserves_baseline(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'config/scope_v9.json').read_text())
        library = json.loads((root / 'config/device_library_v9.json').read_text())
        changed, devices = apply_parameters(config, library,
            {'osfet_write_ns': 15, 'asy_write_ns': 2, 'link_length_mm': 0.5})
        self.assertEqual(library['devices']['OSFET-eDRAM']['write_latency']['value_ns'], 10)
        self.assertEqual(devices['devices']['OSFET-eDRAM']['write_latency']['value_ns'], 15)
        self.assertEqual(devices['devices']['AsyFET-eDRAM']['write_latency']['value_ns'], 2)
        self.assertEqual(changed['layers'], config['layers'])
        self.assertTrue(all(x['link_length_mm'] == 0.5 for x in changed['crossbars']))
        self.assertTrue(all(x['link_length_mm'] == 1 for x in config['crossbars']))
        scaled, _ = apply_parameters(config, library, {'l3_energy_scale': 0.5})
        for device in library['devices']:
            calibration = scaled['layers'][2]['device_overrides'][device]['circuit_calibration']
            self.assertEqual(calibration['read_energy_scale'], 0.5)
            self.assertEqual(calibration['write_energy_scale'], 0.5)
        self.assertEqual(scaled['layers'][2]['device_overrides']['OSFET-eDRAM']
                         ['circuit_calibration']['read_latency_scale'], 0.15)
        for parameters in ({'unknown': 1}, {'asy_ion': -1}, {'osfet_ion': float('nan')}):
            with self.assertRaises(ValueError):
                apply_parameters(config, library, parameters)

    def test_unique_scenarios_and_baseline(self):
        cases = scenarios()
        self.assertEqual(cases[0], ('baseline', {}))
        self.assertEqual(len(cases), len({name for name, _ in cases}))
