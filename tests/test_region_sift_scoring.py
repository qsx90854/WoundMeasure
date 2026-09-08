"""Regression cases for preserving every anchor's group-level evidence."""

from types import SimpleNamespace
import unittest

import numpy as np

from Algorithm.region_sift_scoring import score_candidate_groups


def _config(**changes):
    settings = dict(
        group_balance_weight=0.35, frame_consistency_weight=0.05,
        frame_scale_tolerance_log2=0.5, frame_angle_tolerance_deg=30.0,
        frame_min_reliable_pairs=3, keep_best_ratio=0.75,
        keep_best_count=None, normalize_descriptors=False)
    settings.update(changes)
    return SimpleNamespace(**settings)


def _metadata():
    records = [dict(cell_row=row, cell_col=col, gradient_label=label)
               for row in range(3) for col in range(3)
               for label in ("high", "mid", "low")]
    return records + [dict(cell_row=1, cell_col=1, is_click=True)]


def _frames(count=28, reliable=True):
    return dict(
        size_px=np.linspace(4.0, 16.0, count),
        angle_deg=np.linspace(0.0, 340.0, count),
        scale_reliable=np.full(count, reliable),
        orientation_reliable=np.full(count, reliable))


class RegionSIFTScoringTests(unittest.TestCase):
    def test_seven_contradicting_rows_survive_robust_trimming(self):
        distances = np.zeros((2, 28))
        # Seven left flat descriptors matching rich right patches.  Whether
        # they are labelled low or high must not reduce this evidence.
        distances[1, np.arange(2, 23, 3)] = 512.0
        left = _frames(reliable=False)
        right = {key: np.tile(value, (2, 1)) for key, value in left.items()}
        result = score_candidate_groups(distances, _metadata(), left, right, _config())
        np.testing.assert_equal(result["trimmed_mean"], [0.0, 0.0])
        self.assertEqual(result["keep_count"], 21)
        np.testing.assert_equal(result["keep_mask"].sum(axis=1), [21, 21])
        self.assertEqual(result["group_score"][0], 0.0)
        self.assertGreater(result["group_score"][1], 0.0)

    def test_click_is_counted_with_center_cell_without_overweighting_it(self):
        metadata = _metadata()
        distances = np.zeros((2, 28))
        center_indices = [i for i, item in enumerate(metadata)
                          if (item["cell_row"], item["cell_col"]) == (1, 1)]
        distances[0, center_indices] = 90.0
        distances[1, :3] = 90.0
        left = _frames()
        right = {key: np.tile(value, (2, 1)) for key, value in left.items()}
        result = score_candidate_groups(distances, metadata, left, right, _config())
        np.testing.assert_allclose(result["balanced_mean"], [10.0, 10.0])
        self.assertEqual(result["point_cell_indices"][-1], 4)
        distances[:] = 0.0
        distances[0, -1] = 360.0
        result = score_candidate_groups(distances, metadata, left, right, _config())
        self.assertEqual(result["balanced_mean"][0], 10.0)

    def test_intrinsic_frames_can_be_different_across_anchors(self):
        left = _frames()
        result = score_candidate_groups(
            np.zeros((1, 28)), _metadata(), left, left, _config())
        self.assertEqual(result["frame_penalty"][0], 0.0)
        np.testing.assert_equal(result["frame_scale_residual_log2"], 0.0)
        np.testing.assert_equal(result["frame_angle_residual_deg"], 0.0)

    def test_common_scale_and_rotation_are_allowed_across_angle_wrap(self):
        left = _frames()
        right = {key: value.copy() for key, value in left.items()}
        right["size_px"] *= 2.0
        right["angle_deg"] = (right["angle_deg"] + 179.0) % 360.0
        result = score_candidate_groups(
            np.zeros((1, 28)), _metadata(), left, right, _config())
        self.assertAlmostEqual(result["frame_penalty"][0], 0.0)
        self.assertAlmostEqual(result["frame_scale_center_log2"][0], 1.0)
        self.assertAlmostEqual(result["frame_angle_center_deg"][0], 179.0)

    def test_flat_frames_do_not_penalize_arbitrary_scale_or_direction(self):
        left = _frames(reliable=False)
        right = _frames(reliable=False)
        right["size_px"] = right["size_px"][::-1]
        right["angle_deg"] = right["angle_deg"][::-1]
        result = score_candidate_groups(
            np.zeros((1, 28)), _metadata(), left, right, _config())
        self.assertEqual(result["frame_penalty"][0], 0.0)
        self.assertEqual(result["frame_scale_pair_count"][0], 0)
        self.assertTrue(np.all(np.isnan(result["frame_angle_residual_deg"])))

    def test_reliable_frame_outlier_adds_bounded_penalty(self):
        left = _frames()
        right = _frames()
        right["size_px"][-1] *= 4.0
        right["angle_deg"][-1] += 120.0
        result = score_candidate_groups(
            np.zeros((1, 28)), _metadata(), left, right, _config())
        self.assertGreater(result["frame_penalty"][0], 0.0)
        self.assertAlmostEqual(result["frame_penalty"][0], 0.05 * 512.0 / 28.0)
        self.assertEqual(result["frame_scale_center_log2"][0], 0.0)
        self.assertEqual(result["frame_angle_center_deg"][0], 0.0)
        normalized = score_candidate_groups(
            np.zeros((1, 28)), _metadata(), left, right,
            _config(normalize_descriptors=True))
        self.assertAlmostEqual(
            normalized["frame_penalty"][0] * 512, result["frame_penalty"][0])

    def test_too_few_or_missing_reliability_flags_do_not_create_evidence(self):
        left = _frames(reliable=False)
        right = _frames()
        left["scale_reliable"][:2] = True
        left["orientation_reliable"][:2] = True
        right["angle_deg"][0] += 90
        result = score_candidate_groups(
            np.zeros((1, 28)), _metadata(), left, right, _config())
        self.assertEqual(result["frame_scale_pair_count"][0], 2)
        self.assertEqual(result["frame_penalty"][0], 0.0)
        del right["orientation_reliable"]
        del right["scale_reliable"]
        result = score_candidate_groups(
            np.zeros((1, 28)), _metadata(), left, right, _config())
        self.assertEqual(result["frame_scale_pair_count"][0], 0)

    def test_config_can_restore_legacy_descriptor_score(self):
        rng = np.random.default_rng(3)
        distances = rng.uniform(0, 512, (3, 28))
        left = _frames()
        right = {key: np.tile(value, (3, 1)) for key, value in left.items()}
        result = score_candidate_groups(
            distances, _metadata(), left, right,
            _config(group_balance_weight=0.0, frame_consistency_weight=0.0))
        expected = np.sort(distances, axis=1)[:, :21].mean(axis=1)
        np.testing.assert_allclose(result["objective_without_epi"], expected)

    def test_shape_mismatches_are_rejected(self):
        frames = _frames()
        with self.assertRaises(ValueError):
            score_candidate_groups(np.zeros(28), _metadata(), frames, frames, _config())
        with self.assertRaises(ValueError):
            score_candidate_groups(np.zeros((1, 28)), _metadata()[:-1], frames, frames, _config())
        wrong = {**frames, "size_px": np.ones(27)}
        with self.assertRaises(ValueError):
            score_candidate_groups(np.zeros((1, 28)), _metadata(), frames, wrong, _config())
        with self.assertRaises(ValueError):
            score_candidate_groups(np.zeros((2, 28)), _metadata(), frames, frames, _config())


if __name__ == "__main__":
    unittest.main()
