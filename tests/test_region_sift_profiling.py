"""Instrumentation must not alter numeric results, caches or context lifetime."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from Algorithm import region_sift_frames as frames
from Algorithm import region_sift_profiling as profiling
from Algorithm.Region_SIFT_Matching import DEFAULT_CONFIG, run_region_sift_matching


class ProfilingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(41).integers(0, 256, (220, 260), dtype=np.uint8)
        self.cfg = replace(DEFAULT_CONFIG, search_length_px=31., adaptive_max_expansions=0)
        self.cand = dict(K_R=np.eye(3), R_rel=np.eye(3), t_rel=np.zeros((3, 1)),
                         plane_n=np.array([0., 0., 1.]), plane_c=np.array([0., 0., 1.]),
                         F=np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]))

    def match(self, **kwargs):
        return run_region_sift_matching(self.image, self.image.copy(), (130, 110),
                                       self.cand, np.eye(3), **kwargs)

    def test_frames_and_descriptors_bitwise_equal_with_profiling(self):
        points = np.array([[120., 110.], [120., 110.], [124.5, 112.25]])
        context = frames.create_frame_context(self.image, self.cfg)
        expected = frames.estimate_dense_sift_frames(self.image, points, self.cfg, context=context)
        def descriptor(result):
            return frames.compute_descriptors_at_points(self.image, points, config=self.cfg,
                sizes_px=result['size_px'], angles_deg=result['angle_deg'], octaves=result['octave'])
        expected_desc = descriptor(expected)
        profile = {}
        token = profiling.begin(profile)
        try:
            actual = frames.estimate_dense_sift_frames(self.image, points, self.cfg, context=context)
            actual_desc = descriptor(actual)
        finally:
            profiling.end(token)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(actual_desc, expected_desc)
        self.assertEqual(profile['left.frames']['counts']['input_points'], 3)
        self.assertEqual(profile['left.frames']['counts']['unique_points'], 2)
        self.assertEqual(profile['left.frames']['counts']['duplicate_points'], 1)
        self.assertEqual(profile['left.angle']['calls'], 2)
        self.assertEqual(profile['left.descriptor']['counts']['rows'], 3)
        self.assertIn('OpenCV SIFT.compute（含內部前置）', profile['left.descriptor']['ms'])
        for bucket in profile.values():
            self.assertGreaterEqual(bucket['total_ms']+1e-6, sum(bucket['ms'].values()))

    def test_full_search_and_two_stage_unchanged_without_timers(self):
        for fast in (False, True):
            cfg = replace(self.cfg, two_stage_search=fast)
            actual = self.match(config=cfg)
            with patch('Algorithm.Region_SIFT_Matching.begin_detail', return_value=None), \
                 patch('Algorithm.Region_SIFT_Matching.end_detail'):
                expected = self.match(config=cfg)
            self.assertEqual(actual['reject_reason'], expected['reject_reason'])
            np.testing.assert_array_equal(actual['m_pt'], expected['m_pt'])
            for key in ('candidate_group_scores', 'candidate_objective_scores', 'left_descriptors',
                        'right_descriptors', 'best_center_right'):
                np.testing.assert_array_equal(actual['region_debug'][key], expected['region_debug'][key])
            self.assertIsNone(profiling._active.get())

    def test_multistage_aggregation_and_cache_do_not_double_count(self):
        result = self.match(config=replace(self.cfg, two_stage_search=True))
        profile = result['detail_profile']
        rebuilt = {}
        for stage in result['search_history']:
            profiling.merge_profile(rebuilt, stage['detail_profile'])
        self.assertEqual(profile, rebuilt)
        self.assertEqual(profile['left.pyramid']['calls'], 1)
        self.assertEqual(profile['right.pyramid']['calls'], 1)
        self.assertEqual(profile['left.frames']['calls'], 1)
        self.assertEqual(profile['right.descriptor']['counts']['rows'],
                         result['timing_counts']['right_descriptor_rows'])
        self.assertEqual(profile['right.frames']['counts']['input_points'],
                         28*result['timing_counts']['new_candidate_evaluations'])
        self.assertIs(profile, result['region_debug']['detail_profile'])
        text = '\n'.join(profiling.format_profile(profile))
        self.assertIn('右圖 angle', text)
        self.assertIn('Python 端無法', text)

    def test_failure_restores_context_and_partial_profile_is_available(self):
        result = self.match(config=self.cfg, reject_specular=True,
                            left_spec_mask=np.ones_like(self.image), right_spec_mask=np.ones_like(self.image))
        self.assertIsNone(result['m_pt'])
        self.assertIsNone(profiling._active.get())
        self.assertIn('left.pyramid', result['detail_profile'])
        frames.create_frame_context(self.image, self.cfg)
        self.assertEqual(result['detail_profile']['left.pyramid']['calls'], 1)

    def test_empty_profile_has_readable_output(self):
        self.assertTrue(profiling.format_profile({}))


if __name__ == '__main__':
    unittest.main()
