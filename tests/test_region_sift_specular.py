"""Reflection exclusion preserves anchor count and rejects unusable candidates."""
import ast
from pathlib import Path
import unittest

import cv2
import numpy as np

from Algorithm.Region_SIFT_Matching import (
    DEFAULT_CONFIG, RegionSIFTError, select_region_points,
    run_region_sift_matching, with_config,
)
from tests.test_region_sift_matching import _identity_candidate


class RegionSIFTSpecularTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(29).integers(0, 256, (256, 256), dtype=np.uint8)
        self.mask = np.zeros_like(self.image)
        self.config = with_config(DEFAULT_CONFIG, search_length_px=9., search_width_px=1.,
                                  auto_scale_orientation=False, keypoint_size_px=3.2)

    def match(self, **kwargs):
        args = dict(config=self.config, reject_specular=True,
                    left_spec_mask=self.mask, right_spec_mask=self.mask)
        args.update(kwargs)
        return run_region_sift_matching(self.image, self.image, (128., 128.),
                                        _identity_candidate(), np.eye(3), **args)

    def test_resamples_each_cell_and_retains_28_points(self):
        original, _, _, _ = select_region_points(self.image, (128., 128.), self.config)
        x, y = original[0].astype(int)
        self.mask[y, x] = 255
        points, metadata, _, _ = select_region_points(
            self.image, (128., 128.), self.config, exclusion_mask=self.mask)
        self.assertEqual(points.shape, (28, 2))
        self.assertTrue(np.all(self.mask[points[:, 1].astype(int), points[:, 0].astype(int)] == 0))
        self.assertTrue(metadata[-1]['is_click'])
        np.testing.assert_array_equal(points[-1], [128., 128.])
        for row in range(3):
            for col in range(3):
                self.assertEqual(sum(m['cell_row'] == row and m['cell_col'] == col
                                     and not m['is_click'] for m in metadata), 3)

    def test_reflective_click_or_full_cell_fails_without_reducing_group(self):
        self.mask[128, 128] = 255
        with self.assertRaisesRegex(RegionSIFTError, 'P overlaps'):
            select_region_points(self.image, (128., 128.), self.config, self.mask)
        self.mask[:] = 0
        self.mask[113:123, 113:123] = 255
        with self.assertRaisesRegex(RegionSIFTError, 'needs 3'):
            select_region_points(self.image, (128., 128.), self.config, self.mask)

    def test_strict_support_rejects_reflection_near_clean_anchor_center(self):
        self.mask[128, 133] = 255
        strict = self.match()
        self.assertIsNone(strict['m_pt'])
        self.assertIn('reflection-free support', strict['reject_reason'])
        center = self.match(config=with_config(self.config, specular_check_support=False))
        self.assertIsNone(center['reject_reason'])
        np.testing.assert_allclose(center['m_pt'], [128., 128.])

    def test_right_reflective_centers_reject_all_candidates(self):
        right_mask = self.mask.copy()
        right_mask[124:133, 128] = 255
        result = self.match(right_spec_mask=right_mask,
                            config=with_config(self.config, specular_check_support=False))
        self.assertIsNone(result['m_pt'])
        self.assertIn('reflection-free anchors', result['reject_reason'])
        self.assertEqual(result['timing_counts']['specular_center_rejected_candidates'], 9)

    def test_right_reflection_outside_anchor_grid_is_rejected_by_support(self):
        right_mask = self.mask.copy()
        # Outside the 30x30 grid but within the rightmost anchors' descriptor support.
        right_mask[:, 146] = 255
        result = self.match(right_spec_mask=right_mask)
        self.assertIsNone(result['m_pt'])
        self.assertIn('per-scale SIFT support', result['reject_reason'])
        self.assertEqual(result['timing_counts']['specular_center_rejected_candidates'], 0)
        self.assertGreater(result['timing_counts']['incomplete_support_rejected_candidates'], 0)

    def test_disabled_mode_and_mask_content_cache_invalidation(self):
        cache = {}
        first = self.match(left_cache=cache)
        self.assertIsNone(first['reject_reason'])
        again = self.match(left_cache=cache)
        self.assertTrue(again['timing_counts']['left_cache_hit'])
        self.mask[5, 5] = 255
        changed = self.match(left_cache=cache)
        self.assertIsNone(changed['reject_reason'])
        self.assertFalse(changed['timing_counts']['left_cache_hit'])
        disabled = self.match(reject_specular=False, left_cache=cache)
        self.assertIsNone(disabled['reject_reason'])
        self.assertFalse(disabled['timing_counts']['left_cache_hit'])
        np.testing.assert_array_equal(first['region_debug']['left_descriptors'],
                                      disabled['region_debug']['left_descriptors'])

    def test_missing_mask_fails_only_when_enabled(self):
        self.assertIn('mask missing', self.match(right_spec_mask=None)['reject_reason'])
        self.assertIsNone(self.match(reject_specular=False, right_spec_mask=None)['reject_reason'])

    def test_right_mask_is_warped_from_original_right_coordinates(self):
        cand = _identity_candidate()
        cand['t_rel'] = np.array([[10.], [0.], [0.]])
        cand['F'] = np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        right = cv2.warpAffine(self.image, np.array([[1., 0., 10.], [0., 1., 0.]]), (256, 256))
        right_mask = self.mask.copy()
        right_mask[128, 138] = 255
        result = run_region_sift_matching(
            self.image, right, (128., 128.), cand, np.eye(3),
            config=with_config(self.config, search_length_px=1., specular_check_support=False),
            reject_specular=True, left_spec_mask=self.mask, right_spec_mask=right_mask)
        self.assertIsNone(result['m_pt'])
        self.assertEqual(result['timing_counts']['specular_center_rejected_candidates'], 1)

    def test_application_passes_masks_and_ui_flag_to_matcher(self):
        path = Path(__file__).resolve().parents[1] / 'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'run_region_sift_matching')
        kwargs = {item.arg: item.value for item in call.keywords}
        for enabled in (False, True):
            ns = {'snap_view_state': {'reject_specular_candidates': enabled},
                  'left_spec_mask': self.mask, 'right_spec_mask': self.mask}
            values = {key: eval(compile(ast.Expression(kwargs[key]), str(path), 'eval'), ns)
                      for key in ('reject_specular', 'left_spec_mask', 'right_spec_mask')}
            self.assertEqual(values['reject_specular'], enabled)
            self.assertIs(values['left_spec_mask'], self.mask)
            self.assertIs(values['right_spec_mask'], self.mask)


if __name__ == '__main__':
    unittest.main()
