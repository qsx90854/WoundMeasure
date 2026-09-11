import cv2
import numpy as np
import unittest
from unittest.mock import patch

from Algorithm.Region_SIFT_Matching import (
    DEFAULT_CONFIG,
    RegionSIFTError,
    build_epipolar_band,
    estimate_dense_sift_frames,
    run_region_sift_matching,
    score_descriptor_group,
    select_region_points,
    with_config,
)


def _identity_candidate():
    # This rank-2 F produces x=x_left vertical epipolar lines.  RT/plane H is
    # identity so the unit test can isolate the shared-displacement search.
    return {
        "K_R": np.eye(3, dtype=np.float64),
        "R_rel": np.eye(3, dtype=np.float64),
        "t_rel": np.zeros((3, 1), dtype=np.float64),
        "plane_n": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "plane_c": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "F": np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=np.float64),
    }


class RegionSIFTMatchingTests(unittest.TestCase):
    def test_timing_records_disjoint_stages_and_accumulates_batches(self):
        image = np.random.default_rng(573).integers(0, 256, (180, 220), dtype=np.uint8)
        config = with_config(DEFAULT_CONFIG, search_length_px=7, search_width_px=3,
                             descriptor_batch_size=56)
        result = run_region_sift_matching(
            image, image, (110., 90.), _identity_candidate(), np.eye(3), config=config)
        self.assertIsNone(result['reject_reason'])
        np.testing.assert_allclose(result['m_pt'], (110., 90.))
        debug = result['region_debug']
        times = result['timing_ms']
        self.assertTrue(all(np.isfinite(value) and value >= 0 for value in times.values()))
        self.assertAlmostEqual(sum(times.values()), result['elapsed_ms'])
        self.assertEqual(times, debug['timing_ms'])
        self.assertEqual(result['elapsed_ms'], debug['elapsed_ms'])
        self.assertIn('右圖尺度金字塔與響應圖', times)
        self.assertIn('右圖尺度選擇與角度估計', times)
        self.assertIn('右圖SIFT descriptor（各batch累計）', times)
        self.assertIn('L2距離、群組評分與候選保存（累計）', times)
        counts = result['timing_counts']
        self.assertEqual(counts['right_descriptor_rows'], debug['valid_candidate_count'] * 28)
        self.assertEqual(counts['right_descriptor_batches'],
                         (debug['valid_candidate_count'] + 1) // 2)
        self.assertGreater(counts['right_descriptor_batches'], 1)

    def test_timing_is_available_when_match_fails_before_debug(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        candidate = _identity_candidate()
        candidate['F'] = None
        result = run_region_sift_matching(image, image, (50, 50), candidate, np.eye(3))
        self.assertIsNotNone(result['reject_reason'])
        self.assertIsNone(result['region_debug'])
        self.assertIn('初始化與快取檢查', result['timing_ms'])
        self.assertAlmostEqual(sum(result['timing_ms'].values()), result['elapsed_ms'])
        self.assertEqual(result['timing_counts']['right_descriptor_batches'], 0)

    def test_configuration_rejects_inconsistent_frame_boundaries(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        for sigma in (0.5, float('nan'), float('inf')):
            with self.subTest(sigma=sigma), self.assertRaises(RegionSIFTError):
                select_region_points(image, (50, 50),
                                     with_config(DEFAULT_CONFIG, sift_sigma=sigma))
        with self.assertRaisesRegex(RegionSIFTError, 'at least 3'):
            select_region_points(image, (50, 50),
                                 with_config(DEFAULT_CONFIG, frame_min_reliable_pairs=2))
        points, *_ = select_region_points(
            image, (50, 50), with_config(DEFAULT_CONFIG, frame_min_reliable_pairs=3))
        self.assertEqual(len(points), 28)

    def test_left_cache_reuses_content_but_invalidates_updated_frame(self):
        rng = np.random.default_rng(571)
        image = rng.integers(0, 256, (180, 220), dtype=np.uint8)
        config = with_config(DEFAULT_CONFIG, search_length_px=7, search_width_px=3)
        cache = {}

        def match(frame, saved_cache):
            result = run_region_sift_matching(
                frame, frame.copy(), (110., 90.), _identity_candidate(),
                np.eye(3), config=config, left_cache=saved_cache)
            self.assertIsNone(result['reject_reason'])
            return result['region_debug']

        with patch('Algorithm.Region_SIFT_Matching.select_region_points',
                   wraps=select_region_points) as sampler:
            original = match(image, cache)
            duplicate = match(image.copy(), cache)
            self.assertEqual(sampler.call_count, 1)
            self.assertFalse(original['timing_counts']['left_cache_hit'])
            self.assertTrue(duplicate['timing_counts']['left_cache_hit'])
            self.assertNotIn('左圖SIFT descriptor', duplicate['timing_ms'])
            self.assertIn('左圖快取讀取', duplicate['timing_ms'])
            np.testing.assert_array_equal(original['left_descriptors'],
                                          duplicate['left_descriptors'])
            # Simulate a camera reusing the same ndarray for another frame.
            image[:] = rng.integers(0, 256, image.shape, dtype=np.uint8)
            updated = match(image, cache)
            self.assertEqual(sampler.call_count, 2)
            fresh = match(image, None)
            self.assertEqual(sampler.call_count, 3)
        self.assertFalse(np.array_equal(original['left_descriptors'],
                                        updated['left_descriptors']))
        np.testing.assert_array_equal(updated['left_descriptors'], fresh['left_descriptors'])
        np.testing.assert_array_equal(updated['candidate_group_scores'],
                                      fresh['candidate_group_scores'])

    def test_zero_balance_weight_marks_cell_score_diagnostic_only(self):
        image = np.random.default_rng(572).integers(0, 256, (180, 220), dtype=np.uint8)
        config = with_config(DEFAULT_CONFIG, group_balance_weight=0,
                             search_length_px=7, search_width_px=3)
        result = run_region_sift_matching(
            image, image, (110., 90.), _identity_candidate(), np.eye(3), config=config)
        self.assertIsNone(result['reject_reason'])
        debug = result['region_debug']
        self.assertTrue(all(not row['balanced_score_active']
                            for row in debug['point_metadata']))
        np.testing.assert_allclose(debug['candidate_group_scores'],
                                   debug['candidate_trimmed_scores'])

    def test_default_sampling_is_27_plus_click_and_28x128(self):
        rng = np.random.default_rng(10)
        image = rng.integers(0, 256, (140, 180), dtype=np.uint8)
        point_p = (90.25, 70.25)

        points, metadata, _magnitude, roi = select_region_points(
            image, point_p, DEFAULT_CONFIG)

        self.assertEqual(points.shape, (28, 2))
        self.assertEqual(roi, (75, 55, 30, 30))
        self.assertTrue(metadata[-1]["is_click"])
        np.testing.assert_allclose(points[-1], point_p)
        self.assertEqual(len({tuple(point) for point in points}), 28)

        for row in range(3):
            for col in range(3):
                cell = [
                    record for record in metadata[:-1]
                    if record["cell_row"] == row and record["cell_col"] == col
                ]
                self.assertEqual(
                    [record["gradient_label"] for record in cell],
                    ["high", "mid", "low"])

        result = run_region_sift_matching(
            image, image.copy(), point_p, _identity_candidate(), np.eye(3))
        self.assertIsNone(result["reject_reason"])
        self.assertEqual(
            result["region_debug"]["left_descriptors"].shape, (28, 128))
        self.assertEqual(
            result["region_debug"]["right_descriptors"].shape, (28, 128))
        self.assertEqual(result["region_debug"]["keep_count"], 21)
        self.assertEqual(
            result["region_debug"]["left_frame_sizes_px"].shape, (28,))
        self.assertEqual(
            result["region_debug"]["right_frame_angles_deg"].shape, (28,))
        self.assertIn("left_scale_px", result["region_debug"]["point_metadata"][0])
        self.assertIn("right_angle_deg", result["region_debug"]["point_metadata"][0])

    def test_fixed_anchors_get_independent_scale_and_orientation(self):
        # Large-scale support now includes the Gaussian dependencies. Keep
        # both blobs well away from image boundaries to test scale selection.
        yy, xx = np.mgrid[0:400, 0:600]
        blobs = (
            255.0 * np.exp(-((xx - 150) ** 2 + (yy - 200) ** 2) / (2.0 * 2.0 ** 2))
            + 255.0 * np.exp(-((xx - 450) ** 2 + (yy - 200) ** 2) / (2.0 * 9.0 ** 2))
        )
        blobs = np.clip(blobs, 0, 255).astype(np.uint8)
        points = np.array([[150.0, 200.0], [450.0, 200.0]], dtype=np.float32)
        scale_config = with_config(
            DEFAULT_CONFIG,
            scale_keypoint_sizes_px=(4.0, 6.0, 8.0, 10.0, 12.0,
                                     16.0, 20.0, 24.0),
            descriptor_max_support_radius_px=None,
        )
        scale_frames = estimate_dense_sift_frames(
            blobs, points, scale_config)

        # The anchors are inputs, so frame estimation returns one frame for
        # each without moving or suppressing either point.  The broad blob
        # should select a larger descriptor frame than the narrow blob.
        self.assertEqual(scale_frames["size_px"].shape, (2,))
        self.assertLess(
            float(scale_frames["size_px"][0]),
            float(scale_frames["size_px"][1]))

        ramps = np.zeros((240, 360), dtype=np.uint8)
        ramps[:, :180] = np.tile(np.arange(180, dtype=np.uint8), (240, 1))
        ramps[:, 180:] = np.tile(
            np.arange(240, dtype=np.uint8)[:, None], (1, 180))
        angle_points = np.array(
            [[90.0, 120.0], [270.0, 120.0]], dtype=np.float32)
        angle_config = with_config(
            DEFAULT_CONFIG, scale_keypoint_sizes_px=(8.0,),
            descriptor_max_support_radius_px=None)
        angle_frames = estimate_dense_sift_frames(
            ramps, angle_points, angle_config)
        angle_0_error = min(
            abs(float(angle_frames["angle_deg"][0])),
            abs(float(angle_frames["angle_deg"][0]) - 360.0))
        self.assertLess(angle_0_error, 3.0)
        self.assertAlmostEqual(
            float(angle_frames["angle_deg"][1]), 90.0, delta=3.0)

    def test_auto_frame_estimation_can_be_disabled_for_control_runs(self):
        image = np.zeros((80, 100), dtype=np.uint8)
        points = np.array([[30.0, 30.0], [70.0, 50.0]], dtype=np.float32)
        config = with_config(
            DEFAULT_CONFIG,
            auto_scale_orientation=False,
            keypoint_size_px=13.0,
            keypoint_angle_deg=27.0,
        )
        frames = estimate_dense_sift_frames(image, points, config)
        np.testing.assert_allclose(frames["size_px"], [13.0, 13.0])
        np.testing.assert_allclose(frames["angle_deg"], [27.0, 27.0])
        np.testing.assert_array_equal(frames["scale_index"], [-1, -1])

    def test_points_per_cell_and_keep_ratio_are_tunable(self):
        image = np.arange(160 * 180, dtype=np.uint32).reshape(160, 180)
        image = np.asarray(image % 256, dtype=np.uint8)
        config = with_config(DEFAULT_CONFIG, points_per_cell=4)
        points, metadata, _magnitude, _roi = select_region_points(
            image, (90.0, 80.0), config)

        self.assertEqual(points.shape, (37, 2))
        self.assertEqual(len(metadata), 37)
        self.assertEqual(metadata[0]["gradient_label"], "q100")
        self.assertEqual(metadata[3]["gradient_label"], "q00")

        left = np.zeros((37, 128), dtype=np.float32)
        right = np.zeros_like(left)
        right[:, 0] = np.arange(37, dtype=np.float32)
        score, distances, keep_mask, keep_count = score_descriptor_group(
            left, right, config)
        self.assertEqual(keep_count, 28)
        self.assertEqual(int(np.count_nonzero(keep_mask)), 28)
        self.assertAlmostEqual(
            score, float(np.mean(np.arange(28, dtype=np.float32))))
        self.assertEqual(distances.shape, (37,))

    def test_vertical_epipolar_line_builds_rotated_5x75_band(self):
        cand = _identity_candidate()
        band = build_epipolar_band(
            (90.0, 70.0), cand["F"], np.eye(3), DEFAULT_CONFIG)

        centers = band["centers_warp"]
        self.assertEqual(centers.shape, (375, 2))
        self.assertEqual(len(np.unique(band["along_offsets"])), 75)
        self.assertEqual(len(np.unique(band["across_offsets"])), 5)
        self.assertAlmostEqual(float(np.ptp(centers[:, 0])), 4.0)
        self.assertAlmostEqual(float(np.ptp(centers[:, 1])), 74.0)

        tuned = with_config(
            DEFAULT_CONFIG, search_length_px=11.0, search_width_px=7.0)
        tuned_band = build_epipolar_band(
            (90.0, 70.0), cand["F"], np.eye(3), tuned)
        self.assertEqual(tuned_band["centers_warp"].shape, (77, 2))

    def test_best_21_score_trims_seven_descriptor_outliers(self):
        left = np.zeros((28, 128), dtype=np.float32)
        right = np.zeros_like(left)
        right[:21, 0] = 2.0
        right[21:, 0] = 1000.0

        score, distances, keep_mask, keep_count = score_descriptor_group(
            left, right, DEFAULT_CONFIG)

        self.assertEqual(keep_count, 21)
        self.assertAlmostEqual(score, 2.0)
        self.assertTrue(np.all(keep_mask[:21]))
        self.assertFalse(np.any(keep_mask[21:]))
        self.assertEqual(float(np.max(distances[keep_mask])), 2.0)

    def test_shared_displacement_recovers_known_2d_shift(self):
        rng = np.random.default_rng(4)
        left = rng.integers(0, 256, (180, 220), dtype=np.uint8)
        expected_shift = np.array([2.0, 7.0], dtype=np.float32)
        right = cv2.warpAffine(
            left,
            np.array([[1.0, 0.0, expected_shift[0]],
                      [0.0, 1.0, expected_shift[1]]], dtype=np.float32),
            (left.shape[1], left.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        config = with_config(
            DEFAULT_CONFIG, search_length_px=21.0, search_width_px=5.0)
        point_p = np.array([110.0, 90.0], dtype=np.float32)

        result = run_region_sift_matching(
            left, right, point_p, _identity_candidate(), np.eye(3),
            config=config)

        self.assertIsNone(result["reject_reason"])
        np.testing.assert_allclose(
            result["m_pt"], point_p + expected_shift, atol=0.1)
        debug = result["region_debug"]
        np.testing.assert_allclose(
            debug["best_displacement_warp"], expected_shift, atol=0.1)
        self.assertAlmostEqual(debug["group_score"], 0.0, places=6)
        self.assertTrue(np.isfinite(debug["second_group_score"]))
        self.assertGreater(debug["second_group_score"], debug["group_score"])
        self.assertGreater(debug["group_score_margin"], 0.0)
        self.assertAlmostEqual(debug["objective_score_ratio"], 0.0, places=6)
        self.assertAlmostEqual(debug["kept_l2_stats"]["mean"], 0.0, places=6)
        self.assertGreaterEqual(debug["kept_cell_coverage"], 1)
        second_distance = np.hypot(
            debug["second_along_offset_px"] - debug["best_along_offset_px"],
            debug["second_across_offset_px"] - debug["best_across_offset_px"],
        )
        self.assertGreater(
            second_distance, debug["second_best_exclusion_radius_px"])
        # Every row uses the exact same displacement in warped coordinates.
        displacements = debug["best_points_warp"] - debug["left_points"]
        np.testing.assert_allclose(
            displacements,
            np.repeat(expected_shift.reshape(1, 2), len(displacements), axis=0),
            atol=1e-6)

    def test_plane_warp_match_is_converted_back_to_original_right_coordinates(self):
        rng = np.random.default_rng(19)
        left = rng.integers(0, 256, (180, 240), dtype=np.uint8)
        K = np.array([
            [100.0, 0.0, 120.0],
            [0.0, 100.0, 90.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        t = np.array([[10.0], [0.0], [0.0]], dtype=np.float64)
        tx = np.array([
            [0.0, -t[2, 0], t[1, 0]],
            [t[2, 0], 0.0, -t[0, 0]],
            [-t[1, 0], t[0, 0], 0.0],
        ])
        F = np.linalg.inv(K).T @ tx @ np.linalg.inv(K)
        cand = {
            "K_R": K,
            "R_rel": np.eye(3, dtype=np.float64),
            "t_rel": t,
            "plane_n": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "plane_c": np.array([0.0, 0.0, 100.0], dtype=np.float64),
            "F": F,
        }
        # Plane H contributes +10 px in original-right x.  Region displacement
        # in canonical warped coordinates contributes another (+7,+2).
        H_lr = np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        delta = np.array([7.0, 2.0], dtype=np.float32)
        translate = np.array([
            [1.0, 0.0, delta[0]],
            [0.0, 1.0, delta[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        right = cv2.warpPerspective(
            left, H_lr @ translate, (left.shape[1], left.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
        point_p = np.array([110.0, 90.0], dtype=np.float32)
        config = with_config(
            DEFAULT_CONFIG, search_length_px=21.0, search_width_px=5.0)

        result = run_region_sift_matching(
            left, right, point_p, cand, K, config=config)

        self.assertIsNone(result["reject_reason"])
        np.testing.assert_allclose(
            result["region_debug"]["best_displacement_warp"], delta,
            atol=0.1)
        np.testing.assert_allclose(
            result["m_pt"], point_p + np.array([17.0, 2.0]), atol=0.1)

    def test_click_near_border_fails_instead_of_shrinking_the_group(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaisesRegex(RegionSIFTError, "outside the left image"):
            select_region_points(image, (10.0, 10.0), DEFAULT_CONFIG)

    def test_all_flat_group_retains_rows_but_rejects_location(self):
        image = np.full((180, 220), 128, dtype=np.uint8)
        result = run_region_sift_matching(
            image, image, (110.0, 90.0), _identity_candidate(), np.eye(3))
        self.assertIsNone(result['m_pt'])
        self.assertIn('zero', result['reject_reason'])
        debug = result['region_debug']
        self.assertEqual(debug['left_descriptors'].shape, (28, 128))
        self.assertTrue(np.all(debug['left_frames']['valid']))
        self.assertFalse(np.any(debug['left_frames']['orientation_reliable']))
        self.assertEqual(debug['frame_penalty'], 0.0)
        self.assertEqual(debug['balanced_score'], 0.0)

    def test_flat_and_textured_layout_uses_all_rows_and_shared_shift(self):
        rng = np.random.default_rng(451)
        image = np.full((260, 300), 128, dtype=np.uint8)
        image[60:118, 70:138] = rng.integers(0, 256, (58, 68), dtype=np.uint8)
        shifted = cv2.warpAffine(
            image, np.array([[1, 0, 1], [0, 1, 4]], np.float32),
            (300, 260), borderMode=cv2.BORDER_CONSTANT, borderValue=128)
        result = run_region_sift_matching(
            image, shifted, (145., 125.), _identity_candidate(), np.eye(3),
            config=with_config(DEFAULT_CONFIG, search_length_px=15.))
        self.assertIsNone(result['reject_reason'])
        debug = result['region_debug']
        np.testing.assert_allclose(result['m_pt'], [146., 129.])
        self.assertEqual(debug['left_descriptors'].shape, (28, 128))
        norms = np.linalg.norm(debug['left_descriptors'], axis=1)
        self.assertTrue(np.any(norms == 0))
        self.assertTrue(np.any(norms > 0))
        self.assertTrue(all(row['balanced_score_active'] for row in debug['point_metadata']))
        self.assertAlmostEqual(debug['group_score'],
                               0.65 * debug['trimmed_score'] + 0.35 * debug['balanced_score'])

    def test_oblique_band_filters_border_candidates_without_aborting(self):
        image = np.random.default_rng(4).integers(0, 256, (180, 220), dtype=np.uint8)
        cand = _identity_candidate()
        cand['F'] = np.array([[0, 0, .8], [0, 0, -.6], [-.8, .6, 0]])
        result = run_region_sift_matching(image, image, (42., 42.), cand, np.eye(3))
        self.assertIsNone(result['reject_reason'])
        np.testing.assert_allclose(result['m_pt'], [42., 42.], atol=1e-5)
        self.assertLess(result['region_debug']['valid_candidate_count'],
                        result['region_debug']['candidate_count'])


if __name__ == "__main__":
    unittest.main()
