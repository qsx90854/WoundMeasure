"""Validate live edits without starting Tk or loading the camera application."""
import ast
from pathlib import Path
import unittest
import cv2

from Algorithm.Region_SIFT_Matching import DEFAULT_CONFIG, RegionSIFTConfig
from Algorithm.region_sift_settings import format_value, parse_config_values, LABELS


class RegionSIFTSettingsTests(unittest.TestCase):
    def test_all_fields_round_trip_and_have_descriptions(self):
        entries = {name: format_value(value) for name, value in vars(DEFAULT_CONFIG).items()}
        self.assertEqual(set(entries), set(LABELS))
        self.assertEqual(parse_config_values(DEFAULT_CONFIG, entries), DEFAULT_CONFIG)

    def test_actual_application_config_round_trips(self):
        source = Path(__file__).resolve().parents[1] / 'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'REGION_SIFT_CONFIG'
                                  for t in node.targets))
        ns = {'RegionSIFTConfig': RegionSIFTConfig, 'cv2': cv2}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source), 'exec'), ns)
        config = ns['REGION_SIFT_CONFIG']
        self.assertEqual(parse_config_values(config, {
            name: format_value(value) for name, value in vars(config).items()}), config)

    def test_tuning_types_optional_thresholds_and_original_remain_unchanged(self):
        config = parse_config_values(DEFAULT_CONFIG, {
            'search_width_px': '7', 'search_length_px': '101',
            'points_per_cell': '5', 'scale_keypoint_sizes_px': '3.2, 4, 6.4',
            'normalize_descriptors': 'True', 'max_group_score': 'None',
            'max_objective_score_ratio': '', 'keep_best_count': '30'})
        self.assertEqual((config.search_width_px, config.search_length_px), (7., 101.))
        self.assertEqual(config.scale_keypoint_sizes_px, (3.2, 4., 6.4))
        self.assertEqual(config.keep_best_count, 30)
        self.assertTrue(config.normalize_descriptors)
        self.assertIsNone(config.max_group_score)
        self.assertIsNone(config.max_objective_score_ratio)
        self.assertEqual(DEFAULT_CONFIG.search_width_px, 5.)

    def test_invalid_edits_are_rejected_before_apply(self):
        for edits in (
            {'search_length_px': 'nan'}, {'search_across_step_px': '0'},
            {'grid_rows': '2'}, {'points_per_cell': '2.5'},
            {'points_per_cell': '1000'}, {'keep_best_count': '29'},
            {'scale_keypoint_sizes_px': '3.2, inf'},
            {'scale_keypoint_sizes_px': '10000'}, {'max_group_score': '-1'},
            {'max_objective_score_ratio': '1.1'}, {'normalize_descriptors': 'yes'},
            {'warp_interpolation': '100'}, {'unknown': '1'},
        ):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                parse_config_values(DEFAULT_CONFIG, edits)


if __name__ == '__main__':
    unittest.main()
