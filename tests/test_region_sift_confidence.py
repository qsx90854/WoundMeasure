"""Confidence gates and observational shadow mode, without Tk/camera I/O."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from Algorithm.Region_SIFT_Matching import DEFAULT_CONFIG, run_region_sift_matching, select_region_points
from Algorithm.region_sift_confidence import (
    analyze_profile, analyze_evidence, finalize_confidence, attach_height_span, search_valley)
from Algorithm.region_sift_search import analyze_valley
from Algorithm.region_sift_settings import parse_config_values
from Algorithm import masked_sift_descriptor as masked


def debug_curve(x, y, n=28):
    x, y = np.asarray(x, float), np.asarray(y, float)
    metadata = [dict(cell_row=(i//3)//3, cell_col=(i//3)%3) for i in range(27)]
    metadata.append(dict(cell_row=1, cell_col=1))
    ids = np.flatnonzero(np.isfinite(y))
    rows = np.repeat(y[ids, None], n, axis=1)
    return dict(candidate_along_offsets=x, candidate_across_offsets=np.zeros(len(x)),
        candidate_objective_scores=y, candidate_group_scores=y,
        best_candidate_index=int(np.argmin(y)), candidate_frame_penalties=np.zeros(len(x)),
        point_metadata=metadata, evidence_candidate_indices=ids, evidence_distances=rows,
        evidence_texture_distances=rows*.8, evidence_missing_penalties=rows*.2,
        evidence_pair_valid=np.ones_like(rows, bool), evidence_common_fractions=np.ones_like(rows)*.8,
        best_center_right=np.array([0., 0.]), candidate_centers_warp=np.column_stack([x, np.zeros(len(x))]),
        H_lr=np.eye(3), config=DEFAULT_CONFIG)


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(-12., 13.)
        self.cfg = replace(DEFAULT_CONFIG, confidence_mode='shadow')

    def test_one_wide_plateau_is_not_two_basins(self):
        d = debug_curve(self.x, 100+np.maximum(np.abs(self.x)-4, 0)**2)
        p = analyze_profile(d, self.cfg)
        self.assertEqual(len(p['basins']), 1)
        self.assertIsNone(p['competing_basin'])
        self.assertGreater(p['localization_width_px'], 4)
        r = dict(region_debug=d, m_pt=np.zeros(2), reject_reason=None)
        finalize_confidence(r, self.cfg)
        self.assertIn('localization_too_wide', r['confidence']['reason_codes'])
        self.assertIsNotNone(r['m_pt'])  # shadow

    def test_distinct_equal_valleys_compete(self):
        d = debug_curve(self.x, 100+np.minimum((self.x-6)**2, (self.x+6)**2))
        p = analyze_profile(d, self.cfg)
        self.assertEqual(len(p['basins']), 2)
        self.assertIn('competing_basins', p['reason_codes'])

    def test_invalid_gap_and_physical_gap_are_not_interpolated(self):
        for x in (self.x, np.delete(self.x, np.where(self.x == 1))):
            y = 100+x*x
            if len(x) == len(self.x):
                y[x == 1] = np.inf
            p = analyze_profile(debug_curve(x, y), self.cfg)
            self.assertIn('localization_unbounded', p['reason_codes'])

    def test_small_ripples_cannot_hide_a_large_ridge_between_competitors(self):
        x = np.arange(-10., 11.)
        y = np.array([140,130,115,100,102,108,107,114,113,120,119,
                      120,113,114,107,108,102,101,115,130,140], float)
        for curve in (y, y[::-1]):
            cfg = replace(self.cfg, confidence_mode='active')
            d = debug_curve(x, curve)
            p = analyze_profile(d, cfg)
            self.assertEqual(len(p['basins']), 2)
            self.assertIn('competing_basins', p['reason_codes'])
            self.assertFalse(search_valley(d, cfg, analyze_valley(d, cfg))['clear'])
            r = dict(region_debug=d, m_pt=np.zeros(2), reject_reason=None)
            finalize_confidence(r, cfg)
            self.assertIsNone(r['m_pt'])
            self.assertIn('competing_basins', r['reject_reason'])

    def test_score_decomposition_preserves_original_keep_and_cell_weights(self):
        d = debug_curve(self.x, 100+self.x**2)
        rng = np.random.default_rng(2)
        values = rng.uniform(50, 400, d['evidence_distances'].shape)
        valid = d['evidence_pair_valid']
        valid[:, 0] = False
        values[:, 0] = 512*np.sqrt(2)
        raw = values*.7
        raw[:, 0] = np.nan
        d.update(evidence_distances=values, evidence_texture_distances=raw,
                 evidence_missing_penalties=values-np.where(valid, raw, 0))
        analyze_evidence(d, self.cfg, analyze_profile(d, self.cfg))
        for row in range(len(values)):
            kept = np.sort(values[row, 1:])[:21].mean()
            cell_means = []
            for cell in range(9):
                indices = [i for i, m in enumerate(d['point_metadata']) if m['cell_row']*3+m['cell_col'] == cell]
                cell_means.append(values[row, indices].mean())
            expected = .65*kept+.35*np.mean(cell_means)
            self.assertAlmostEqual(d['candidate_texture_contributions'][row]+d['candidate_missing_contributions'][row], expected)

    def test_active_routing_uses_objective_basin_not_group_shoulder_gate(self):
        cfg = replace(self.cfg, confidence_mode='active', valley_min_relative_depth=.9)
        d = debug_curve(self.x, 100+self.x**2)
        legacy = analyze_valley(d, cfg)
        self.assertFalse(legacy['clear'])
        self.assertTrue(search_valley(d, cfg, legacy)['clear'])

    def test_active_veto_and_shadow_do_not_rescue_hard_rejections(self):
        d = debug_curve(self.x, 100+self.x**2)
        for mode in ('shadow', 'active'):
            cfg = replace(self.cfg, confidence_mode=mode)
            r = dict(region_debug=d, m_pt=None, reject_reason='BestG exceeds max_group_score')
            finalize_confidence(r, cfg)
            self.assertIsNone(r['m_pt'])
            self.assertIn('upstream_rejection', r['confidence']['reason_codes'])

    def test_stability_uses_existing_rows_and_detects_region_dominance(self):
        d = debug_curve(self.x, 100+np.minimum((self.x-6)**2, (self.x+6)**2))
        d['evidence_distances'][:] = 300
        a, b = np.flatnonzero(self.x == -6)[0], np.flatnonzero(self.x == 6)[0]
        d['evidence_distances'][a] = 101
        d['evidence_distances'][a, :3] = 0
        d['evidence_distances'][b] = 100
        p = analyze_profile(d, self.cfg)
        result = analyze_evidence(d, self.cfg, p)
        self.assertFalse(result['cell_trials'][0]['stable'])
        self.assertLess(result['cell_stability'], 1.)

    def test_height_callback_and_optional_gate_shadow_vs_active(self):
        for mode in ('shadow', 'active'):
            cfg = replace(self.cfg, confidence_mode=mode, confidence_height_span_max_mm=.1)
            d = debug_curve(self.x, 100+self.x**2); d['config'] = cfg
            r = dict(region_debug=d, m_pt=np.zeros(2), reject_reason=None)
            finalize_confidence(r, cfg)
            calls = []
            def tri(q):
                calls.append(q)
                return np.array([0., 0., 10+q[0]])
            attach_height_span(r, tri, np.array([0, 0, 1]), np.zeros(3), offset=2.)
            self.assertEqual(len(calls), 5)
            self.assertEqual(r['confidence']['height_range_mm'], (6., 10.))
            self.assertEqual(r['m_pt'] is None, mode == 'active')

    def test_config_validation(self):
        self.assertEqual(parse_config_values(self.cfg, {'confidence_mode': 'ACTIVE'}).confidence_mode, 'active')
        for values in ({'confidence_mode':'foo'}, {'confidence_max_width_px':'0'},
                       {'confidence_min_cell_stability':'1.1'}, {'confidence_height_span_max_mm':'-1'}):
            with self.assertRaises(ValueError):
                parse_config_values(self.cfg, values)

    def test_active_pipeline_is_scoreable_and_rejects_flat_evidence(self):
        image = np.random.default_rng(12).integers(0, 256, (220, 260), dtype=np.uint8)
        cand = dict(K_R=np.eye(3), R_rel=np.eye(3), t_rel=np.zeros((3, 1)),
                    plane_n=np.array([0., 0., 1.]), plane_c=np.array([0., 0., 1.]),
                    F=np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]))
        for staged in (False, True):
            cfg = replace(DEFAULT_CONFIG, confidence_mode='active', auto_scale_orientation=False,
                          keypoint_size_px=3.2, search_length_px=31., two_stage_search=staged,
                          adaptive_max_expansions=0, fine_half_length_px=4.)
            for sample, accepted in ((image, True), (np.full_like(image, 100), False)):
                r = run_region_sift_matching(sample, sample.copy(), (130, 110), cand, np.eye(3), config=cfg)
                self.assertEqual(r['m_pt'] is not None, accepted, r['reject_reason'])
                self.assertEqual(r['candidate_status'] == 'accepted', accepted)

    def test_missing_candidate_geometry_is_not_a_zero_height_span(self):
        cfg = replace(self.cfg, confidence_mode='active', confidence_height_span_max_mm=1.)
        d = debug_curve(self.x, 100+self.x**2); d['config'] = cfg
        r = dict(region_debug=d, m_pt=np.zeros(2), reject_reason=None)
        finalize_confidence(r, cfg)
        attach_height_span(r, lambda q: None, np.array([0.,0.,1.]), np.zeros(3))
        self.assertIsNone(r['confidence']['height_span_mm'])
        self.assertIsNone(r['m_pt'])
        self.assertIn('invalid_candidate_geometry', r['confidence']['reason_codes'])

    def test_local_plane_height_interval_is_explicitly_unavailable(self):
        d = debug_curve(self.x, 100+self.x**2)
        r = dict(region_debug=d, m_pt=np.zeros(2), reject_reason=None, refinement={'applied':True})
        finalize_confidence(r, self.cfg)
        with patch('cv2.triangulatePoints', side_effect=AssertionError('no rigid interval for refined result')):
            attach_height_span(r, lambda q: None, np.array([0.,0.,1.]), np.zeros(3))
        self.assertEqual(r['confidence']['height_reason'], 'local_plane_result_not_covered')
        self.assertIsNone(r['confidence']['height_range_mm'])

    def test_masked_sampling_is_invariant_to_excluded_intensity(self):
        cfg = replace(DEFAULT_CONFIG, use_masked_sift=True, masked_sampling_gradient=True)
        image = np.random.default_rng(41).integers(50, 151, (220, 260), dtype=np.uint8)
        mask = np.zeros_like(image); mask[102:108, 120:126] = 1
        changed = image.copy(); image[mask > 0] = 0; changed[mask > 0] = 255
        a = select_region_points(image, (130, 110), cfg, exclusion_mask=mask)
        b = select_region_points(changed, (130, 110), cfg, exclusion_mask=mask)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])
        self.assertFalse(mask[a[0][:, 1].astype(int), a[0][:, 0].astype(int)].any())

    def test_clean_sampling_exactly_preserves_legacy_anchors_and_gradients(self):
        image = np.random.default_rng(41).integers(50, 151, (220, 260), dtype=np.uint8)
        mask = np.zeros_like(image)
        for size in (1, 3, 5, 7):
            cfg = replace(DEFAULT_CONFIG, use_masked_sift=True, sobel_ksize=size)
            a = select_region_points(image, (130,110), replace(cfg, masked_sampling_gradient=False), exclusion_mask=mask)
            b = select_region_points(image, (130,110), replace(cfg, masked_sampling_gradient=True), exclusion_mask=mask)
            np.testing.assert_array_equal(a[0], b[0])
            x0,y0,x1,y1 = a[3]
            np.testing.assert_array_equal(a[2][y0:y1,x0:x1], b[2][y0:y1,x0:x1])

    def test_remote_mask_does_not_change_clean_roi_anchors(self):
        image = np.random.default_rng(42).integers(0, 256, (220,260), dtype=np.uint8)
        mask = np.zeros_like(image); mask[:40,:40] = 1
        cfg = replace(DEFAULT_CONFIG, use_masked_sift=True)
        a = select_region_points(image, (130,110), replace(cfg, masked_sampling_gradient=False), exclusion_mask=mask)
        b = select_region_points(image, (130,110), replace(cfg, masked_sampling_gradient=True), exclusion_mask=mask)
        np.testing.assert_array_equal(a[0], b[0])

    def test_shadow_same_search_result_and_descriptor_work_as_legacy(self):
        image = np.random.default_rng(42).integers(0, 256, (220, 260), dtype=np.uint8)
        mask = np.zeros_like(image); mask[99:102, 118:121] = 1
        cand = dict(K_R=np.eye(3), R_rel=np.eye(3), t_rel=np.zeros((3, 1)),
                    plane_n=np.array([0., 0., 1.]), plane_c=np.array([0., 0., 1.]),
                    F=np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]))
        for staged in (False, True):
            cfg = replace(DEFAULT_CONFIG, use_masked_sift=True, auto_scale_orientation=False,
                          keypoint_size_px=3.2, search_length_px=31., two_stage_search=staged,
                          adaptive_max_expansions=0, fine_half_length_px=4.)
            def run(c):
                return run_region_sift_matching(image, image.copy(), (130, 110), cand, np.eye(3),
                                               config=c, left_spec_mask=mask, right_spec_mask=mask)
            a = run(cfg)
            with patch('matplotlib.pyplot.figure', side_effect=AssertionError('no plots')):
                b = run(replace(cfg, confidence_mode='shadow'))
            self.assertEqual(a['reject_reason'], b['reject_reason'])
            np.testing.assert_array_equal(a['m_pt'], b['m_pt'])
            self.assertEqual(a['timing_counts']['right_descriptor_rows'], b['timing_counts']['right_descriptor_rows'])
            self.assertEqual(a['timing_counts']['right_descriptor_batches'], b['timing_counts']['right_descriptor_batches'])
            self.assertIn('confidence', b)
            self.assertGreater(len(b['confidence'].get('cell_trials', [])), 0)


if __name__ == '__main__':
    unittest.main()
