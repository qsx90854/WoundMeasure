from dataclasses import replace
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from Algorithm.Region_SIFT_Matching import DEFAULT_CONFIG, run_region_sift_matching
from Algorithm.region_sift_refinement import fit_cost_table, refine_match
from Algorithm.region_sift_settings import parse_config_values
from tests.test_region_sift_matching import _identity_candidate


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(DEFAULT_CONFIG, local_plane_refine=True,
            search_length_px=15, search_width_px=1, auto_scale_orientation=False,
            keypoint_size_px=3.2, max_objective_score_ratio=None)
        self.xy = np.array([(x,y) for y in (-10,0,10) for x in (-10,0,10)], float)
        self.metadata = [dict(cell_row=i//3, cell_col=i%3) for i in range(9)]

    def test_synthetic_cost_plane_recovers_center_and_slopes(self):
        cfg = replace(self.config, refine_sample_step_px=.25, refine_slope_penalty=0.)
        offsets = np.linspace(-2, 2, 17)
        target = .5 + .05*self.xy[:,0] - .025*self.xy[:,1]
        costs = 10 + 40*(offsets[:,None]-target[None])**2
        model, reason = fit_cost_table(offsets, costs, self.xy, self.metadata, cfg)
        self.assertIsNotNone(model, reason)
        self.assertAlmostEqual(model['center_shift_px'], .5, places=6)
        np.testing.assert_allclose(model['shifts'], target, atol=.13)
        self.assertLessEqual(model['model_count'], 125*cfg.refine_model_levels)

    def test_flat_one_sided_and_boundary_models_not_accepted(self):
        offsets = np.arange(-2,3.)
        model, _ = fit_cost_table(offsets, np.ones((5,9)), self.xy, self.metadata, self.config)
        self.assertIsNone(model)
        costs = (offsets[:,None] - np.zeros((1,9)))**2
        model, _ = fit_cost_table(offsets, costs, self.xy+30, self.metadata, self.config)
        self.assertIsNone(model)
        costs = (offsets[:,None] - np.full((1,9), 3))**2
        model, reason = fit_cost_table(offsets, costs, self.xy, self.metadata, self.config)
        self.assertIsNone(model)
        self.assertIn('boundary', reason)

    def test_parameters_validated(self):
        for field, value in [('refine_radius_px','0'), ('refine_sample_step_px','.3'),
                             ('refine_model_levels','6'), ('refine_slope_penalty','-1')]:
            with self.assertRaises(ValueError):
                parse_config_values(self.config, {field:value})

    def run_image(self, config, right=None, **kwargs):
        image = np.random.default_rng(77).integers(0,256,(180,220),dtype=np.uint8)
        return run_region_sift_matching(image, image if right is None else right,
            (110.,90.), _identity_candidate(), np.eye(3), config=config, **kwargs)

    def test_identity_keeps_original_and_counts_work(self):
        off = self.run_image(replace(self.config, local_plane_refine=False))
        on = self.run_image(self.config)
        self.assertIsNotNone(on['m_pt'], on['reject_reason'])
        np.testing.assert_array_equal(on['m_pt'], off['m_pt'])
        self.assertFalse(on['refinement']['applied'])
        self.assertGreater(on['refinement']['descriptor_rows'], 0)
        np.testing.assert_array_equal(on['region_debug']['candidate_group_scores'],
                                      off['region_debug']['candidate_group_scores'])
        self.assertAlmostEqual(on['elapsed_ms'], sum(on['timing_ms'].values()))

    def test_two_stage_refines_once_after_search_only(self):
        from Algorithm import region_sift_refinement as module
        with patch.object(module, 'refine_match', wraps=module.refine_match) as call:
            result = self.run_image(replace(self.config, two_stage_search=True,
                fine_half_length_px=4, adaptive_max_expansions=0))
        self.assertIsNotNone(result['m_pt'], result['reject_reason'])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result['timing_counts']['right_context_builds'], 1)
        self.assertAlmostEqual(sum(result['timing_ms'].values()), result['elapsed_ms'])

    def test_rejected_not_rescued(self):
        result = dict(m_pt=None, reject_reason='ambiguous', region_debug={})
        with patch('Algorithm.region_sift_search.evaluate_positions') as evaluate:
            refine_match(result, (0,0), self.config)
        evaluate.assert_not_called()
        self.assertIsNone(result['m_pt'])
        self.assertEqual(result['reject_reason'], 'ambiguous')

    def slanted_example(self, enabled=True):
        image = np.random.default_rng(77).integers(0,256,(180,220),dtype=np.uint8)
        image = cv2.GaussianBlur(image,(3,3),.6)
        right = cv2.warpAffine(image, np.array([[1,0,0],[.06,1,.6-.06*110]],np.float32), (220,180))
        cfg = replace(self.config, local_plane_refine=enabled, refine_sample_step_px=.5)
        return run_region_sift_matching(image, right, (110.,90.), _identity_candidate(),
                                       np.eye(3), config=cfg), cfg

    def test_actual_slanted_image_improves_center_and_preserves_original_debug(self):
        result, _ = self.slanted_example()
        report = result['refinement']
        self.assertTrue(report['applied'], report['reason'])
        truth = np.array([110.,90.6])
        self.assertLess(np.linalg.norm(result['m_pt']-truth),
                        np.linalg.norm(report['original_center_right']-truth))
        np.testing.assert_array_equal(result['region_debug']['best_center_right'],
                                      report['original_center_right'])
        self.assertGreater(report['verified_gain'], 0)
        self.assertIn('+LocalPlane', result['method'])
        self.assertLessEqual(report['descriptor_rows'], 28*10)

    def test_exact_recheck_failure_keeps_original_match(self):
        result, cfg = self.slanted_example(False)
        old = result['m_pt'].copy()
        from Algorithm.region_sift_search import evaluate_positions
        calls = []
        def evaluate(*args, **kwargs):
            output = evaluate_positions(*args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                output[1][:] = 10000
            return output
        with patch('Algorithm.region_sift_search.evaluate_positions', side_effect=evaluate):
            refine_match(result, (110.,90.), replace(cfg, local_plane_refine=True))
        self.assertEqual(len(calls), 2)
        self.assertFalse(result['refinement']['applied'])
        np.testing.assert_array_equal(old, result['m_pt'])

    def test_coordinate_unwarp_used_for_refined_center(self):
        result, cfg = self.slanted_example(False)
        H = np.array([[1.,.01,5.],[.02,1.,-3.],[.0001,0.,1.]])
        result['region_debug']['H_lr'] = H
        refine_match(result, (110.,90.), replace(cfg, local_plane_refine=True))
        report = result['refinement']
        self.assertTrue(report['applied'], report['reason'])
        expected = H @ np.r_[report['refined_center_warp'],1.]
        np.testing.assert_allclose(result['m_pt'], expected[:2]/expected[2], atol=1e-5)

    def test_masked_identity_retains_mask_backend(self):
        mask = np.zeros((180,220), np.uint8)
        result = self.run_image(replace(self.config, use_masked_sift=True),
                                left_spec_mask=mask, right_spec_mask=mask)
        self.assertIsNotNone(result['m_pt'], result['reject_reason'])
        self.assertEqual(result['region_debug']['descriptor_backend'], 'custom-masked')
        self.assertFalse(result['refinement']['applied'])

    def test_actual_ui_toggle_uses_settings_and_lock_checks(self):
        source = Path(__file__).parents[1]/'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        names = ('apply_region_settings', 'on_region_refine')
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = dict(REGION_SIFT_CONFIG=replace(self.config, local_plane_refine=False),
                  height_profile=None, block_accuracy_state={'mode':'idle'},
                  btn_sift_backend=SimpleNamespace(label=SimpleNamespace(set_text=Mock())),
                  btn_region_refine=SimpleNamespace(label=SimpleNamespace(set_text=Mock())),
                  btn_center_refine=SimpleNamespace(label=SimpleNamespace(set_text=Mock())),
                  center_refinement_viewer=SimpleNamespace(close=Mock()),
                  depth_text=SimpleNamespace(set_text=Mock()), request_blit_refresh=Mock())
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(source), 'exec'), ns)
        ns['on_region_refine'](None)
        self.assertTrue(ns['REGION_SIFT_CONFIG'].local_plane_refine)
        ns['block_accuracy_state']['mode'] = 'running'
        ns['on_region_refine'](None)
        self.assertTrue(ns['REGION_SIFT_CONFIG'].local_plane_refine)
        ns['block_accuracy_state']['mode'] = 'idle'
        ns['on_region_refine'](None)
        self.assertFalse(ns['REGION_SIFT_CONFIG'].local_plane_refine)


if __name__ == '__main__':
    unittest.main()
