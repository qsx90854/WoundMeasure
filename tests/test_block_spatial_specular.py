"""Spatial-mask regression tests with synthetic images; no AI/camera/UI."""
import ast
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from Algorithm.specular_detection import (
    BlockSpatialSpecularConfig, compute_specular_mask_bgr_block_adaptive,
    compute_specular_mask_bgr, compute_rt_aligned_temporal_specular_mask_bgr,
)


class BlockSpatialSpecularTests(unittest.TestCase):
    def setUp(self):
        self.config = BlockSpatialSpecularConfig(
            block_width_px=32, block_height_px=32,
            v_min=170., rgb_min=180., v_percentile=90., rgb_percentile=92.,
            interpolate_thresholds=False, open_kernel_px=1, close_kernel_px=1, dilate_px=0)

    def test_gate_rejects_uniform_yellow_white_but_retains_local_highlight(self):
        image = np.full((128, 128, 3), (160, 200, 210), np.uint8)
        image[59:68, 59:68] = (220, 235, 245)
        config = replace(self.config, block_width_px=64, block_height_px=64,
                         v_percentile=10., rgb_percentile=5., local_hot_percentile=10.)
        enabled, debug = compute_specular_mask_bgr_block_adaptive(image, config, return_debug=True)
        disabled = compute_specular_mask_bgr_block_adaptive(
            image, replace(config, enable_prominence_gate=False))
        self.assertEqual(disabled[15, 15], 255)
        self.assertEqual(debug['candidate_mask'][15, 15], 255)
        np.testing.assert_array_equal(disabled, debug['candidate_mask'])
        self.assertEqual(enabled[15, 15], 0)
        self.assertEqual(enabled[63, 63], 255)
        self.assertLess(np.count_nonzero(enabled), np.count_nonzero(disabled))

    def test_strong_exception_preserves_saturated_white_but_not_saturated_yellow(self):
        for bgr, expected in (((252, 252, 252), 255), ((190, 245, 255), 0)):
            image = np.full((64, 64, 3), bgr, np.uint8)
            mask = compute_specular_mask_bgr_block_adaptive(image, self.config)
            self.assertTrue(np.all(mask == expected))
            no_exception = compute_specular_mask_bgr_block_adaptive(
                image, replace(self.config, enable_strong_highlight_exception=False))
            self.assertFalse(np.any(no_exception))

    def test_mad_threshold_adapts_to_texture_and_multiplier(self):
        rng = np.random.default_rng(77)
        gray = np.full((64, 128), 150, np.uint8)
        gray[:, 64:] = rng.integers(110, 191, (64, 64), dtype=np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        config = replace(self.config, block_width_px=64, block_height_px=64)
        _, debug = compute_specular_mask_bgr_block_adaptive(image, config, return_debug=True)
        delta = gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), config.background_sigma)
        for col in range(2):
            values = delta[:, col * 64:(col + 1) * 64]
            expected = max(config.prominence_min, np.median(values) +
                           config.prominence_mad_multiplier * np.median(np.abs(values - np.median(values))))
            self.assertAlmostEqual(debug['prominence_threshold_grid'][0, col], expected)
        self.assertGreater(debug['prominence_threshold_grid'][0, 1], debug['prominence_threshold_grid'][0, 0])
        _, relaxed = compute_specular_mask_bgr_block_adaptive(
            image, replace(config, prominence_mad_multiplier=1.), return_debug=True)
        self.assertLess(relaxed['prominence_threshold_grid'][0, 1], debug['prominence_threshold_grid'][0, 1])

    def test_same_brightness_uses_different_local_thresholds(self):
        image = np.full((32, 64, 3), 100, np.uint8)
        image[:, 32:] = 220
        image[12:17, 12:17] = 190
        image[12:17, 44:49] = 190
        before = image.copy()
        mask, debug = compute_specular_mask_bgr_block_adaptive(image, self.config, return_debug=True)
        np.testing.assert_array_equal(debug['threshold_grids'][0], [[170., 220.]])
        self.assertEqual(mask[14, 14], 255)
        self.assertEqual(mask[14, 46], 0)
        np.testing.assert_array_equal(image, before)

    def test_partial_tiles_and_interpolation_cover_every_pixel(self):
        image = np.full((35, 67, 3), 100, np.uint8)
        image[:, 32:] = 220
        mask, debug = compute_specular_mask_bgr_block_adaptive(
            image, replace(self.config, interpolate_thresholds=True), return_debug=True)
        self.assertEqual(mask.shape, (35, 67))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(debug['x_edges'], [0, 32, 64, 67])
        self.assertEqual(debug['y_edges'], [0, 32, 35])
        self.assertEqual(debug['threshold_maps'].shape, (3, 35, 67))
        self.assertTrue(np.all(np.isfinite(debug['threshold_maps'])))
        self.assertEqual(debug['prominence_threshold_map'].shape, (35, 67))
        self.assertTrue(np.all(np.isfinite(debug['prominence_threshold_map'])))
        v_map = debug['threshold_maps'][0]
        self.assertLess(abs(v_map[10, 32] - v_map[10, 31]), 3.)
        self.assertEqual(v_map[-1, -1], 220.)

    def test_dark_tiles_not_forced_to_contain_highlights_and_tiny_image(self):
        for shape in ((65, 67, 3), (1, 1, 3)):
            image = np.full(shape, 80, np.uint8)
            mask = compute_specular_mask_bgr_block_adaptive(image)
            self.assertFalse(np.any(mask))

    def test_block_size_is_tunable(self):
        image = np.full((32, 64, 3), 100, np.uint8)
        image[:, 32:] = 220
        _, debug = compute_specular_mask_bgr_block_adaptive(
            image, replace(self.config, block_width_px=64), return_debug=True)
        self.assertEqual(debug['threshold_grids'].shape, (3, 1, 1))
        self.assertEqual(debug['threshold_grids'][0, 0, 0], 220.)

    def test_invalid_parameters(self):
        image = np.zeros((32, 32, 3), np.uint8)
        for edits in ({'block_width_px': 0}, {'v_percentile': 101},
                      {'v_min': 250, 'v_max': 240}, {'dilate_px': -1},
                      {'background_sigma': float('nan')}, {'open_kernel_px': 2},
                      {'prominence_min': -1}, {'prominence_mad_multiplier': -1},
                      {'strong_v_min': 256}):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                compute_specular_mask_bgr_block_adaptive(image, replace(self.config, **edits))


class SpatialApplicationRoutingTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        names = ('compute_wound_adaptive_spatial_mask', 'compute_locked_spec_masks', 'on_adaptive_spatial_specular')
        nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        # Only the callback's nonlocal flag is needed; use globals in the harness.
        for node in nodes:
            node.body = [ast.Global(stmt.names) if isinstance(stmt, ast.Nonlocal) else stmt for stmt in node.body]
        self.ns = dict(np=np, cv2=cv2, use_wound_adaptive_spatial_specular=True,
                       use_specular_model_v2=False,
                       SPATIAL_BLOCK_CONFIG=BlockSpatialSpecularConfig(),
                       compute_specular_mask_bgr_block_adaptive=compute_specular_mask_bgr_block_adaptive,
                       compute_specular_mask_bgr_wound_adaptive=Mock(return_value=np.full((32, 32), 255, np.uint8)),
                       prediction_to_wound_mask=Mock(return_value=None),
                       compute_rt_aligned_temporal_specular_mask_bgr=compute_rt_aligned_temporal_specular_mask_bgr,
                       video_data={}, KL=np.eye(3), ACTUAL_MARKER_SIZE_MM=10,
                       process_view=None, preprocess_gray=None,
                       ENABLE_WOUND_AI=False, wound_state={'left_pred': None, 'right_pred': None},
                       view_state={'adaptive_spatial_specular': True},
                       c19=Mock(), refresh_wound_predictions=Mock(),
                       recompute_locked_spec_masks_from_wound=Mock(), request_blit_refresh=Mock())
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), self.ns)

    def test_without_prediction_routes_to_tiles_and_temporal_combination(self):
        image = np.full((32, 32, 3), 180, np.uint8)
        combined, spatial, temporal = self.ns['compute_locked_spec_masks'](image, 0)
        np.testing.assert_array_equal(spatial, compute_specular_mask_bgr_block_adaptive(image))
        np.testing.assert_array_equal(combined, cv2.bitwise_or(spatial, temporal))
        self.ns['prediction_to_wound_mask'].assert_not_called()
        self.ns['compute_specular_mask_bgr_wound_adaptive'].assert_not_called()

    def test_off_preserves_fixed_detector_and_valid_wound_preserves_wound_route(self):
        image = np.full((32, 32, 3), 180, np.uint8)
        self.ns['use_wound_adaptive_spatial_specular'] = False
        _, spatial, _ = self.ns['compute_locked_spec_masks'](image, 0)
        np.testing.assert_array_equal(spatial, compute_specular_mask_bgr(image))
        self.ns['use_wound_adaptive_spatial_specular'] = True
        self.ns['prediction_to_wound_mask'].return_value = np.ones((32, 32), np.uint8)
        self.ns['compute_locked_spec_masks'](image, 0, wound_prediction=['prediction'])
        self.ns['compute_specular_mask_bgr_wound_adaptive'].assert_called_once()

    def test_missing_or_tiny_wound_mask_uses_tiles(self):
        image = np.full((32, 32, 3), 180, np.uint8)
        for wound_mask in (None, np.zeros((32, 32), np.uint8), np.eye(3, dtype=np.uint8)):
            self.ns['prediction_to_wound_mask'].return_value = wound_mask
            _, spatial, _ = self.ns['compute_locked_spec_masks'](image, 0, ['prediction'])
            np.testing.assert_array_equal(spatial, compute_specular_mask_bgr_block_adaptive(image))
        self.ns['compute_specular_mask_bgr_wound_adaptive'].assert_not_called()

    def test_toggling_without_ai_always_recomputes_and_never_requests_inference(self):
        for expected in (False, True):
            self.ns['on_adaptive_spatial_specular'](None)
            self.assertEqual(self.ns['use_wound_adaptive_spatial_specular'], expected)
        self.assertEqual(self.ns['recompute_locked_spec_masks_from_wound'].call_count, 2)
        self.ns['refresh_wound_predictions'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
