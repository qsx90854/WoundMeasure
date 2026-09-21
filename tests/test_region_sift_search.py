"""Two-stage decisions, real-image cache equivalence, and read-only diagnostics."""
from dataclasses import replace
import unittest
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Algorithm.Region_SIFT_Matching import DEFAULT_CONFIG, run_region_sift_matching
from Algorithm.region_sift_search import analyze_valley, run_two_stage, format_stage_funnel
from Algorithm.region_sift_settings import parse_config_values


def score_debug(x, scores):
    return dict(candidate_along_offsets=np.asarray(x, float),
                candidate_across_offsets=np.zeros(len(x)),
                candidate_group_scores=np.asarray(scores, float),
                candidate_objective_scores=np.asarray(scores, float))


class ValleyTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(-46., 47., 2.)
        self.config = replace(DEFAULT_CONFIG, two_stage_search=True, search_length_px=95.)

    def test_clear_single_valley(self):
        result = analyze_valley(score_debug(self.x, 100+np.minimum(self.x**2, 200)), self.config)
        self.assertTrue(result['clear'], result)
        self.assertEqual(result['along'], 0.)

    def test_flat_shallow_boundary_and_missing_rejected(self):
        cases = [np.full(len(self.x), 100.), 100+.0001*self.x**2,
                 100+(self.x-self.x[0])**2]
        missing = 100+self.x**2
        missing[self.x == 2] = np.inf
        cases.append(missing)
        for scores in cases:
            self.assertFalse(analyze_valley(score_debug(self.x, scores), self.config)['clear'])

    def test_equal_separated_valleys_rejected(self):
        scores = 100+np.minimum((self.x-20)**2, (self.x+20)**2)
        result = analyze_valley(score_debug(self.x, scores), self.config)
        self.assertFalse(result['clear'])
        self.assertIn('competing', result['reason'])

    def test_plateau_is_one_basin_but_overwide_bottom_rejected(self):
        scores = 100+np.maximum(np.abs(self.x)-4, 0)**2
        self.assertTrue(analyze_valley(score_debug(self.x, scores), self.config)['clear'])
        broad = replace(self.config, valley_shoulder_distance_px=40)
        scores = 100+np.maximum(np.abs(self.x)-14, 0)**2
        result = analyze_valley(score_debug(self.x, scores), broad)
        self.assertFalse(result['clear'])

    def test_invalid_coverage_not_confused_with_deep_valley(self):
        scores = 100+self.x**2
        scores[np.abs(self.x) > 12] = np.inf
        result = analyze_valley(score_debug(self.x, scores), self.config)
        self.assertFalse(result['clear'])
        self.assertEqual(result['valid_count'], 13)
        self.assertEqual(result['sample_count'], len(self.x))
        self.assertEqual(result['required_valid_count'], 33)

    def test_settings_validate_new_fields(self):
        for changes in ({'adaptive_max_expansions': '3'}, {'fine_step_px': '0'},
                        {'coarse_along_step_px': 'nan'}, {'valley_min_relative_depth': '-1'},
                        {'adaptive_points_increment': '0'}, {'valley_min_side_samples': '0'}):
            with self.assertRaises(ValueError):
                parse_config_values(self.config, changes)


class PipelineTests(unittest.TestCase):
    def run_curve(self, curve, **changes):
        calls = []
        cfg = replace(DEFAULT_CONFIG, two_stage_search=True, search_length_px=95., **changes)
        def engine(left, right, point, cand, K, **kwargs):
            config = kwargs['config']
            offsets = kwargs['_offsets']
            values = np.array([curve(a, b, config) for a, b in offsets])
            index = int(np.argmin(values))
            debug = dict(candidate_along_offsets=offsets[:, 0],
                candidate_across_offsets=offsets[:, 1], candidate_group_scores=values,
                candidate_objective_scores=values, valid_candidate_count=len(values),
                best_along_offset_px=float(offsets[index, 0]), best_across_offset_px=float(offsets[index, 1]),
                config=config)
            calls.append((config, offsets.copy()))
            return dict(m_pt=offsets[index].copy(), method='Region-SIFT', region_debug=debug,
                        reject_reason=None, timing_counts={}, timing_ms={}, elapsed_ms=0.)
        result = run_two_stage(engine, None, None, (0, 0), {}, np.eye(3), sift=None,
                              config=cfg, left_cache=None)
        return result, calls, cfg

    def test_expands_twice_from_ui_base_then_keeps_final_anchors(self):
        def curve(a, b, cfg):
            return 100. if cfg.points_per_cell < 7 else 100+min(a*a, 200)+b*b
        result, calls, original = self.run_curve(curve, coarse_side_rescue=False)
        self.assertIsNotNone(result['m_pt'])
        self.assertEqual([(c.cell_width_px, c.points_per_cell) for c, _ in calls],
                         [(10, 3), (25, 5), (40, 7), (40, 7)])
        self.assertEqual(original.cell_width_px, 10)
        self.assertEqual(result['effective_config'].points_per_cell, 7)

    def test_three_rounds_limit_and_never_force_ambiguous_match(self):
        result, calls, _ = self.run_curve(lambda a, b, c: 100., coarse_side_rescue=False)
        self.assertIsNone(result['m_pt'])
        self.assertEqual(len(calls), 3)
        self.assertIn('localization unclear', result['reject_reason'])

    def test_side_rescue_before_expansion(self):
        def curve(a, b, c):
            return 400. if b == 0 else 100+min(a*a, 200)
        result, calls, _ = self.run_curve(curve)
        self.assertEqual([s['stage'] for s in result['search_history']],
                         ['coarse', 'coarse+sides', 'fine+global-coarse'])
        self.assertTrue(all(c.points_per_cell == 3 for c, _ in calls))
        self.assertIsNotNone(result['m_pt'])

    def test_stage_formatter_exposes_funnel_and_parameter_pointer(self):
        stage = dict(stage='coarse', round=0, point_count=28, candidate_count=49,
            new_candidates=49, cache_hits=0, reason='insufficient valid coarse coverage',
            candidate_funnel=dict(requested_candidates=49, in_bounds_candidates=49,
                warp_support_candidates=49, center_mask_candidates=49, new_candidates=49,
                cache_hits=0, new_frame_accepted=49, new_descriptor_group_accepted=10,
                new_descriptor_group_rejected=39, final_scoreable_candidates=10,
                required_valid_points=21, valid_points_min=8, valid_points_median=17.,
                valid_points_max=23, deficient_cells_min=0, deficient_cells_median=2.,
                deficient_cells_max=5, pair_samples=1372,
                left_coverage_invalid_pairs=0, right_coverage_invalid_pairs=100,
                frame_invalid_pairs=0, common_coverage_invalid_pairs=400),
            thresholds=dict(use_masked_sift=True, required_valid_points=21,
                masked_min_valid_fraction=.6, masked_min_cell_fraction=.8,
                masked_min_common_fraction=.5, masked_min_group_fraction=.75,
                keep_best_ratio=.75, keep_best_count=None, masked_min_points_per_cell=1,
                masked_max_deficient_cells=1, valley_min_valid_fraction=.7,
                valley_min_relative_depth=.08, valley_max_basin_ratio=.95,
                fine_half_length_px=10., search_length_px=95.),
            valley=dict(valid_count=10, sample_count=49, valid_fraction=10/49,
                        required_valid_count=35))
        text = '\n'.join(format_stage_funnel(stage))
        self.assertIn('scoreable(total)=10', text)
        self.assertIn('10/49', text)
        self.assertIn('masked_min_common_fraction', text)
        self.assertIn('valley_min_valid_fraction', text)

    def test_remote_coarse_valley_not_discarded_during_fine(self):
        # Two unequal valleys: retain the remote one even though only the best is refined.
        result, calls, _ = self.run_curve(lambda a, b, c: min(100+a*a, 180+(a-30)**2)+b*b)
        self.assertIsNotNone(result['m_pt'])
        final = calls[-1][1]
        self.assertTrue(np.any(final[:, 0] == 31))
        self.assertEqual(result['search_history'][-1]['fine_candidate_count'], 105)

    def test_boundary_valley_never_reaches_fine(self):
        result, calls, _ = self.run_curve(lambda a, b, c: 100+(a+47)**2, coarse_side_rescue=False)
        self.assertIsNone(result['m_pt'])
        self.assertEqual(len(calls), 3)
        self.assertIn('boundary', result['reject_reason'])


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(42).integers(0, 256, (250, 300), dtype=np.uint8)
        self.cand = dict(K_R=np.eye(3), R_rel=np.eye(3), t_rel=np.zeros((3, 1)),
            plane_n=np.array([0., 0., 1.]), plane_c=np.array([0., 0., 1.]),
            F=np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]))
        self.config = replace(DEFAULT_CONFIG, auto_scale_orientation=False,
                              keypoint_size_px=3.2, search_length_px=95.)
        self.addCleanup(plt.close, 'all')

    def run_match(self, cfg, right=None, **kwargs):
        return run_region_sift_matching(self.image, self.image.copy() if right is None else right,
                                       (150, 125), self.cand, np.eye(3), config=cfg, **kwargs)

    def test_full_mode_same_result_and_common_candidate_scores_exactly_equal(self):
        full = self.run_match(self.config)
        fast = self.run_match(replace(self.config, two_stage_search=True))
        np.testing.assert_array_equal(fast['m_pt'], full['m_pt'])
        self.assertIsNone(fast['reject_reason'])
        fd, dd = full['region_debug'], fast['region_debug']
        lookup = {tuple(pair): i for i, pair in enumerate(fd['candidate_centers_warp'])}
        for j, pair in enumerate(dd['candidate_centers_warp']):
            i = lookup[tuple(pair)]
            self.assertEqual(dd['candidate_group_scores'][j], fd['candidate_group_scores'][i])
            self.assertEqual(dd['candidate_objective_scores'][j], fd['candidate_objective_scores'][i])
        counts = fast['timing_counts']
        self.assertEqual(counts['warp_builds'], 1)
        self.assertEqual(counts['left_context_builds'], 1)
        self.assertEqual(counts['right_context_builds'], 1)
        self.assertGreater(counts['candidate_cache_hits'], 0)
        self.assertLess(counts['right_descriptor_rows'], full['timing_counts']['right_descriptor_rows'])

    def test_actual_shift_and_float_coordinates_preserved(self):
        right = cv2.warpAffine(self.image, np.float32([[1, 0, 8], [0, 1, 0]]), (300, 250))
        fast = self.run_match(replace(self.config, two_stage_search=True), right)
        self.assertIsNotNone(fast['m_pt'], fast['reject_reason'])
        np.testing.assert_allclose(fast['m_pt'], [158, 125], atol=.01)
        self.assertEqual(fast['m_pt'].dtype, np.float32)

    def test_automatic_scale_orientation_same_scores_and_single_pyramids(self):
        cfg = replace(self.config, auto_scale_orientation=True, search_length_px=31.,
                      adaptive_max_expansions=0)
        full = self.run_match(cfg)
        fast = self.run_match(replace(cfg, two_stage_search=True))
        self.assertIsNotNone(fast['region_debug'], fast['reject_reason'])
        fd, dd = full['region_debug'], fast['region_debug']
        lookup = {tuple(pair): i for i, pair in enumerate(fd['candidate_centers_warp'])}
        for j, pair in enumerate(dd['candidate_centers_warp']):
            self.assertAlmostEqual(dd['candidate_group_scores'][j],
                                   fd['candidate_group_scores'][lookup[tuple(pair)]], places=5)
        self.assertEqual(fast['timing_counts']['right_context_builds'], 1)

    def test_image_mutation_does_not_reuse_previous_left_features(self):
        cfg = replace(self.config, two_stage_search=True)
        cache = {}
        first = self.run_match(cfg, left_cache=cache)
        self.image[:] = 100
        second = self.run_match(cfg, left_cache=cache)
        self.assertIsNotNone(first['m_pt'])
        self.assertIsNone(second['m_pt'])
        self.assertEqual(second['timing_counts']['warp_builds'], 1)

    def test_invalid_mask_stops_without_expansion(self):
        mask = np.ones_like(self.image)
        fast = self.run_match(replace(self.config, two_stage_search=True),
                              reject_specular=True, left_spec_mask=mask, right_spec_mask=mask)
        self.assertIsNone(fast['m_pt'])
        self.assertEqual(len(fast['search_history']), 1)

    def test_all_flat_expansion_limit_and_search_debug_render(self):
        self.image[:] = 100
        fast = self.run_match(replace(self.config, two_stage_search=True, coarse_side_rescue=False))
        self.assertIsNone(fast['m_pt'])
        self.assertEqual(len(fast['search_history']), 3)
        self.assertEqual([s['point_count'] for s in fast['search_history']], [28, 46, 64])
        from Algorithm.region_sift_search_debug import draw_search_history
        figure = plt.figure(figsize=(12, 8))
        draw_search_history(figure, fast['search_history'], fast['reject_reason'])
        figure.canvas.draw()
        self.assertEqual(len(figure.axes), 3)

    def test_real_fast_result_diagnostic_and_profile_cache(self):
        from Algorithm.region_sift_diagnostics import RegionSIFTDiagnostics
        from Algorithm.drag_height_profile import ProfileCache
        fast = self.run_match(replace(self.config, two_stage_search=True))
        result = dict(u=150, v=125, pt=fast['m_pt'], region_debug=fast['region_debug'],
                      region_search_history=fast['search_history'], debug_left_gray=self.image,
                      region_diagnostic_config=self.config)
        cache = ProfileCache()
        try:
            cache.save(0, result)
            restored = cache.load(0)
            self.assertEqual(len(restored['region_search_history']), 2)
            viewer = RegionSIFTDiagnostics()
            viewer.show(restored)
            viewer.figure.canvas.draw()
            viewer.search_figure.canvas.draw()
            viewer.show(dict(fail_reason='No support', region_search_history=fast['search_history']))
            viewer.figure.canvas.draw()
        finally:
            cache.close()


if __name__ == '__main__':
    unittest.main()
