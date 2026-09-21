"""Model behaviour and actual application callbacks, without camera/Tk startup."""
import ast
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from Algorithm.specular_detection import BlockSpatialSpecularConfig
from Algorithm.specular_model_v2 import (
    SpecularModelV2Config, detect_specular_model_v2, fit_dichromatic_models,
)
from Algorithm.specular_model_v2_settings import V2_LABELS, format_v2_value, parse_v2_values


def encode(rgb):
    return np.rint(np.clip(rgb, 0, 1)**(1/2.2)*255).astype(np.uint8)[..., ::-1].copy()


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = SpecularModelV2Config()
        self.tissue = np.empty((96, 128, 3), np.float32)
        self.tissue[:] = (.30, .06, .03)

    def test_shading_and_yellow_stripe_are_not_specular(self):
        shading = self.tissue.copy()
        shading[:, 50:65] *= 1.8
        yellow = encode(self.tissue)
        yellow[35:45, 10:115] = (160, 195, 210)
        for image in (encode(self.tissue), encode(shading), yellow):
            with self.subTest():
                self.assertFalse(np.any(detect_specular_model_v2(image)))

    def test_added_illuminant_is_detected_without_expanding_into_tissue(self):
        rgb = self.tissue.copy()
        rgb[43:50, 60:67] += .30
        before = encode(rgb)
        original = before.copy()
        mask, debug = detect_specular_model_v2(before, return_debug=True)
        self.assertGreater(np.count_nonzero(mask[43:50, 60:67]), 40)
        self.assertLessEqual(np.count_nonzero(mask), 49)
        self.assertFalse(np.any((mask > 0) & (debug['uncertain_mask'] > 0)))
        np.testing.assert_array_equal(before, original)

    def test_white_clipped_and_neutral_are_uncertain_not_excluded(self):
        for level in (180, 255):
            image = np.full((32, 32, 3), level, np.uint8)
            mask, debug = detect_specular_model_v2(image, return_debug=True)
            self.assertFalse(np.any(mask))
            self.assertTrue(np.all(debug['uncertain_mask'] == 255))

    def test_small_images_and_black_have_finite_diagnostics(self):
        for shape in ((1, 1, 3), (3, 20, 3), (31, 17, 3)):
            mask, debug = detect_specular_model_v2(np.zeros(shape, np.uint8), return_debug=True)
            self.assertEqual(mask.shape, shape[:2])
            self.assertFalse(np.any(mask))
            self.assertFalse(np.any(debug['uncertain_mask']))
            for value in debug.values():
                self.assertTrue(np.all(np.isfinite(value)))

    def test_nonnegative_fit_recovers_known_components_and_handles_boundaries(self):
        diffuse = np.array([.8, .4, .2], np.float32)
        diffuse /= np.linalg.norm(diffuse)
        light = np.ones(3, np.float32)/np.sqrt(3)
        rgb = .4*diffuse + .2*light
        ea, eb, a, s, separation = fit_dichromatic_models(rgb, diffuse, light)
        self.assertGreater(ea, eb)
        self.assertLess(eb, 1e-10)
        self.assertAlmostEqual(float(a), .4, places=5)
        self.assertAlmostEqual(float(s), .2, places=5)
        for point in (diffuse*.8, light*.7, np.zeros(3), np.array([1, 0, 0])):
            ea, eb, a, s, _ = fit_dichromatic_models(point, diffuse, light)
            self.assertGreaterEqual(a, 0)
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(eb, ea+1e-8)

    def test_settings_are_separate_validated_and_atomic(self):
        entries = {name: format_v2_value(value) for name, value in vars(self.config).items()}
        self.assertEqual(set(entries), set(V2_LABELS))
        self.assertTrue(all(name.startswith('v2_') for name in entries))
        self.assertEqual(parse_v2_values(self.config, entries), self.config)
        for edit in ({'v_percentile': '80'}, {'v2_seed_score': 'nan'},
                     {'v2_grow_score': '.9'}, {'v2_light_rgb': '0,0,0'},
                     {'v2_reference_radii_px': '8,4'}, {'v2_clip_channels': '4'},
                     {'v2_grow_distance_px': '1.5'}, {'v2_noise_floor': '0'}):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                parse_v2_values(self.config, edit)


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        names = ('compute_wound_adaptive_spatial_mask', 'compute_locked_spec_masks',
                 'recompute_locked_spec_masks_from_wound', 'commit_specular_backend',
                 'apply_spatial_specular_settings', 'apply_specular_v2_settings',
                 'on_specular_v2', 'on_adaptive_spatial_specular')
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
        for node in nodes:
            node.body = [ast.Global(n.names) if isinstance(n, ast.Nonlocal) else n for n in node.body]
        mask = np.full((16, 16), 255, np.uint8)
        uncertain = np.zeros_like(mask)
        details = dict(uncertain_mask=uncertain, mask_fraction=1., uncertain_fraction=0., elapsed_ms=1.)
        self.ns = dict(
            np=np, cv2=cv2, use_specular_model_v2=False, use_wound_adaptive_spatial_specular=False,
            SPECULAR_MODEL_V2_CONFIG=SpecularModelV2Config(), SPATIAL_BLOCK_CONFIG=BlockSpatialSpecularConfig(),
            locked_L_clean=np.zeros((16, 16, 3), np.uint8), locked_R_clean=np.ones((16, 16, 3), np.uint8),
            locked_L_idx=1, locked_R_idx=2, current_cand={},
            extra_candidates_list=[dict(spec_mask=mask, spec_spatial_mask=mask, spec_temporal_mask=mask)],
            specular_v2_display={'left': None, 'right': None},
            detect_specular_model_v2=Mock(return_value=(mask, details)),
            compute_specular_mask_bgr_block_adaptive=Mock(return_value=uncertain),
            compute_specular_mask_bgr_wound_adaptive=Mock(return_value=uncertain),
            prediction_to_wound_mask=Mock(return_value=mask),
            compute_rt_aligned_temporal_specular_mask_bgr=Mock(
                side_effect=lambda *args, **kw: (kw['base_mask'], kw['base_mask'], uncertain)),
            video_data={}, KL=np.eye(3), ACTUAL_MARKER_SIZE_MM=10, process_view=None, preprocess_gray=None,
            ENABLE_WOUND_AI=False, wound_state={'left_pred': None, 'right_pred': None},
            block_accuracy_state={'mode': 'idle'}, height_profile=None,
            view_state={}, c19=Mock(), c15=Mock(), btn_specular_v2=Mock(), depth_text=Mock(),
            mark_display_dirty=Mock(), request_blit_refresh=Mock())
        for side in ('L', 'R'):
            for kind in ('spec_mask', 'spec_spatial_mask', 'spec_temporal_mask'):
                self.ns[f'locked_{side}_{kind}'] = uncertain
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), self.ns)
        self.silent = patch('builtins.print')
        self.silent.start()
        self.addCleanup(self.silent.stop)

    def test_toggle_rebuilds_both_masks_and_invalidates_extra_frames(self):
        self.ns['on_specular_v2'](None)
        self.assertTrue(self.ns['use_specular_model_v2'])
        self.assertEqual(self.ns['detect_specular_model_v2'].call_count, 2)
        self.ns['compute_specular_mask_bgr_block_adaptive'].assert_not_called()
        self.assertIs(self.ns['current_cand']['spec_spatial_mask'], self.ns['locked_R_spec_spatial_mask'])
        self.assertTrue(all(value is None for value in self.ns['extra_candidates_list'][0].values()))
        self.assertTrue(self.ns['view_state']['show_spatial_specular_mask'])
        self.ns['mark_display_dirty'].assert_called_once()
        self.ns['on_specular_v2'](None)
        self.assertFalse(self.ns['use_specular_model_v2'])

    def test_failed_second_image_rolls_back_config_masks_and_overlay(self):
        previous_config = self.ns['SPECULAR_MODEL_V2_CONFIG']
        previous_left = self.ns['locked_L_spec_mask']
        first_result = self.ns['detect_specular_model_v2'].return_value
        self.ns['detect_specular_model_v2'].side_effect = [first_result, RuntimeError('right failed')]
        with self.assertRaises(RuntimeError):
            self.ns['apply_specular_v2_settings'](replace(previous_config, v2_seed_score=.8))
        self.assertIs(self.ns['SPECULAR_MODEL_V2_CONFIG'], previous_config)
        self.assertFalse(self.ns['use_specular_model_v2'])
        self.assertIs(self.ns['locked_L_spec_mask'], previous_left)
        self.assertEqual(self.ns['specular_v2_display'], {'left': None, 'right': None})
        self.assertIsNotNone(self.ns['extra_candidates_list'][0]['spec_mask'])

    def test_batch_and_profile_block_changes(self):
        self.ns['block_accuracy_state']['mode'] = 'running'
        with self.assertRaises(ValueError):
            self.ns['commit_specular_backend'](True)
        self.ns['detect_specular_model_v2'].assert_not_called()

    def test_legacy_settings_restore_legacy_backend(self):
        self.ns['commit_specular_backend'](True)
        updated = replace(self.ns['SPATIAL_BLOCK_CONFIG'], v_percentile=75.)
        self.ns['apply_spatial_specular_settings'](updated)
        self.assertFalse(self.ns['use_specular_model_v2'])
        self.assertTrue(self.ns['use_wound_adaptive_spatial_specular'])
        self.assertIs(self.ns['SPATIAL_BLOCK_CONFIG'], updated)
        self.assertEqual(self.ns['compute_specular_mask_bgr_block_adaptive'].call_count, 2)

    def test_v2_does_not_silently_fall_back_to_wound_or_fixed_detector(self):
        self.ns['use_specular_model_v2'] = True
        self.ns['compute_wound_adaptive_spatial_mask'](self.ns['locked_L_clean'], ['wound'])
        self.ns['prediction_to_wound_mask'].assert_not_called()
        self.ns['compute_specular_mask_bgr_wound_adaptive'].assert_not_called()
        self.ns['on_adaptive_spatial_specular'](None)
        self.assertFalse(self.ns['use_wound_adaptive_spatial_specular'])


if __name__ == '__main__':
    unittest.main()
