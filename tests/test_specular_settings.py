"""Validate live specular edits without loading Tk or the camera application."""
from dataclasses import fields
import unittest

from Algorithm.specular_detection import BlockSpatialSpecularConfig
from Algorithm.specular_settings import LABELS, format_value, parse_config_values


class SpecularSettingsTests(unittest.TestCase):
    def test_all_fields_round_trip_and_have_descriptions(self):
        config = BlockSpatialSpecularConfig()
        entries = {field.name: format_value(getattr(config, field.name))
                   for field in fields(config)}
        self.assertEqual(set(entries), set(LABELS))
        self.assertEqual(parse_config_values(config, entries), config)

    def test_valid_edits_are_atomic_and_preserve_original(self):
        original = BlockSpatialSpecularConfig()
        updated = parse_config_values(original, {
            'block_width_px': '64',
            'v_percentile': '12.5',
            'enable_prominence_gate': 'False',
            'dilate_px': '2',
        })
        self.assertEqual(updated.block_width_px, 64)
        self.assertEqual(updated.v_percentile, 12.5)
        self.assertFalse(updated.enable_prominence_gate)
        self.assertEqual(updated.dilate_px, 2)
        self.assertNotEqual(updated, original)
        self.assertTrue(original.enable_prominence_gate)

    def test_invalid_edits_are_rejected_before_apply(self):
        config = BlockSpatialSpecularConfig()
        for edits in (
            {'block_width_px': '0'}, {'block_height_px': '4.5'},
            {'v_percentile': '101'}, {'v_percentile': 'nan'},
            {'v_min': '250', 'v_max': '240'}, {'background_sigma': '0'},
            {'open_kernel_px': '2'}, {'dilate_px': '-1'},
            {'enable_prominence_gate': 'yes'}, {'unknown': '1'},
        ):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                parse_config_values(config, edits)


if __name__ == '__main__':
    unittest.main()
