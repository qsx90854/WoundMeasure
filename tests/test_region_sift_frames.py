"""Behavioral checks for fixed anchors, scale layers, and complete support."""

from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from Algorithm.region_sift_frames import (
    compute_descriptors_at_points,
    create_frame_context,
    estimate_dense_sift_frames,
)


def _config(**changes):
    fields = dict(scale_keypoint_sizes_px=(3.2, 4.0, 5.0, 6.4, 8.0, 10.0),
                  descriptor_max_support_radius_px=40.0,
                  flat_keypoint_size_px=3.2,
                  frame_coordinate_quantization_px=0.0)
    fields.update(changes)
    return SimpleNamespace(**fields)


class DenseSIFTFrameTests(unittest.TestCase):
    @staticmethod
    def _asymmetric_blob():
        yy, xx = np.mgrid[:256, :256]
        image = (25 + 160 * np.exp(-((xx - 110) ** 2 + (yy - 140) ** 2) / (2 * 3.3 ** 2))
                 + 90 * np.exp(-((xx - 117) ** 2 + (yy - 144) ** 2) / (2 * 2.0 ** 2)))
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _descriptor(image, point, frame, config):
        return compute_descriptors_at_points(image, point, config=config,
                                              sizes_px=frame["size_px"],
                                              angles_deg=frame["angle_deg"],
                                              octaves=frame["octave"])

    def test_rotation_preserves_selected_scale_and_rotates_independent_angle(self):
        image = self._asymmetric_blob()
        config = _config(descriptor_max_support_radius_px=None)
        point = np.array([[110, 140]], np.float32)
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_point = np.array([[115, 110]], np.float32)
        frame = estimate_dense_sift_frames(image, point, config)
        rotated_frame = estimate_dense_sift_frames(rotated, rotated_point, config)
        np.testing.assert_array_equal(frame["size_px"], rotated_frame["size_px"])
        difference = (float(rotated_frame["angle_deg"][0] - frame["angle_deg"][0]) - 90 + 180) % 360 - 180
        self.assertLess(abs(difference), 0.1)
        descriptor = self._descriptor(image, point, frame, config)
        rotated_descriptor = self._descriptor(rotated, rotated_point, rotated_frame, config)
        self.assertLess(float(np.linalg.norm(descriptor - rotated_descriptor)), 1.0)

    def test_doubled_image_selects_doubled_scale_and_beats_wrong_scale_descriptor(self):
        image = self._asymmetric_blob()
        config = _config(scale_keypoint_sizes_px=(3.2, 4, 5, 6.4, 8, 10, 12.8, 16, 20, 25.6),
                         descriptor_max_support_radius_px=None)
        point = np.array([[110, 140]], np.float32)
        enlarged = cv2.warpAffine(image, np.array([[2, 0, 0], [0, 2, 0]], np.float32),
                                 (512, 512), flags=cv2.INTER_LINEAR)
        frame = estimate_dense_sift_frames(image, point, config)
        enlarged_frame = estimate_dense_sift_frames(enlarged, point * 2, config)
        self.assertTrue(frame["valid"][0] and enlarged_frame["valid"][0])
        self.assertAlmostEqual(float(enlarged_frame["size_px"][0] / frame["size_px"][0]), 2.0, places=5)
        descriptor = self._descriptor(image, point, frame, config)
        correct = self._descriptor(enlarged, point * 2, enlarged_frame, config)
        wrong = self._descriptor(enlarged, point * 2, frame, config)
        error = float(np.linalg.norm(descriptor - correct))
        self.assertLess(error, 20)
        self.assertGreater(float(np.linalg.norm(descriptor - wrong)), 5 * max(error, 1))

    def test_frames_are_independent_of_other_anchors_and_batch_order(self):
        image = np.random.default_rng(11).integers(0, 256, (260, 300), dtype=np.uint8)
        config = _config()
        points = np.array([[90.25, 110.75], [170.5, 150.25], [90.25, 110.75]], np.float32)
        together = estimate_dense_sift_frames(image, points, config)
        alone = estimate_dense_sift_frames(image, points[:1], config)
        context = create_frame_context(image, config)
        reversed_frames = estimate_dense_sift_frames(image, points[::-1], config, context=context)
        for field in together:
            np.testing.assert_array_equal(together[field][:1], alone[field])
            np.testing.assert_array_equal(together[field], reversed_frames[field][::-1])
        self.assertTrue(np.all(together["valid"]))

    def test_packed_layers_match_selected_gaussian_scale_and_compute(self):
        image = np.random.default_rng(12).integers(0, 256, (240, 260), dtype=np.uint8)
        config = _config()
        point = np.array([[120.0, 110.0]], np.float32)
        for requested in (3.2, 4.0, 5.0, 6.4):
            selected_config = _config(scale_keypoint_sizes_px=(requested,))
            frame = estimate_dense_sift_frames(image, point, selected_config)
            packed = int(frame["octave"][0])
            octave, layer = packed & 255, (packed >> 8) & 255
            actual = 3.2 * 2 ** (octave + layer / 3)
            self.assertAlmostEqual(float(frame["size_px"][0]), actual, places=5)
            actual_descriptor = compute_descriptors_at_points(
                image, point, config=selected_config, sizes_px=frame["size_px"],
                angles_deg=frame["angle_deg"], octaves=frame["octave"])
            keypoint = cv2.KeyPoint(120.0, 110.0, actual, float(frame["angle_deg"][0]),
                                   0, packed)
            _, expected = cv2.SIFT_create().compute(image, [keypoint])
            np.testing.assert_array_equal(actual_descriptor, expected)
        self.assertGreater(packed & 255, 0)

    def test_flat_anchor_keeps_zero_descriptor_and_compact_uncertain_frame(self):
        image = np.full((220, 240), 100, np.uint8)
        point = np.array([[110.0, 110.0]], np.float32)
        config = _config()
        frame = estimate_dense_sift_frames(image, point, config)
        self.assertTrue(frame["valid"][0])
        self.assertFalse(frame["scale_reliable"][0])
        self.assertFalse(frame["orientation_reliable"][0])
        self.assertAlmostEqual(float(frame["size_px"][0]), 3.2, places=6)
        descriptor = compute_descriptors_at_points(image, point, config=config,
                                                   sizes_px=frame["size_px"],
                                                   angles_deg=frame["angle_deg"],
                                                   octaves=frame["octave"])
        self.assertEqual(descriptor.shape, (1, 128))
        np.testing.assert_array_equal(descriptor, np.zeros((1, 128)))
        textured = image.copy()
        textured[105:115, 107:113] = 250
        other = compute_descriptors_at_points(textured, point, config=config,
                                               sizes_px=frame["size_px"],
                                               angles_deg=frame["angle_deg"],
                                               octaves=frame["octave"])
        self.assertGreater(float(np.linalg.norm(other - descriptor)), 400)

    def test_real_support_rejects_warp_hole_beyond_old_eight_pixel_margin(self):
        image = np.full((220, 240), 100, np.uint8)
        config = _config()
        points = np.array([[70.0, 100.0], [160.0, 100.0]], np.float32)
        mask = np.ones_like(image, np.uint8)
        mask[100, 90] = 0
        frames = estimate_dense_sift_frames(image, points, config, valid_mask=mask)
        np.testing.assert_array_equal(frames["valid"], [False, True])
        self.assertGreater(float(frames["support_radius_px"][1]), 20)

    def test_warp_fraction_tuning_does_not_relax_image_bounds(self):
        image = np.full((180, 180), 100, np.uint8)
        mask = np.ones_like(image, np.uint8)
        mask[90, 90] = 0
        points = np.array([[90, 90], [5, 5]], np.float32)
        strict = estimate_dense_sift_frames(image, points, _config(), valid_mask=mask)
        permissive = estimate_dense_sift_frames(image, points, _config(min_valid_warp_ratio=0.99),
                                                 valid_mask=mask)
        np.testing.assert_array_equal(strict["valid"], [False, False])
        np.testing.assert_array_equal(permissive["valid"], [True, False])

    def test_selects_valid_small_scale_when_large_scale_reaches_border(self):
        image = np.random.default_rng(6).integers(0, 256, (180, 180), dtype=np.uint8)
        config = _config()
        frames = estimate_dense_sift_frames(image, np.array([[30.0, 90.0]]), config)
        self.assertTrue(frames["valid"][0])
        self.assertLessEqual(frames["support_radius_px"][0], 30)
        self.assertAlmostEqual(float(frames["size_px"][0]), 3.2, places=6)

    def test_nominal_support_cap_and_invalid_coordinates(self):
        image = np.full((180, 180), 100, np.uint8)
        config = _config()
        context = create_frame_context(image, config)
        self.assertEqual(len(context["active_sizes_px"]), 4)
        self.assertTrue(all(entry["nominal_radius"] <= 40 for entry in context["entries"]))
        points = np.array([[90, 90], [-5, 90], [np.nan, 90]], np.float32)
        frames = estimate_dense_sift_frames(image, points, config, context=context)
        np.testing.assert_array_equal(frames["valid"], [True, False, False])
        self.assertTrue(np.all(np.isfinite(frames["size_px"])))

    def test_custom_pyramid_parameters_override_mismatched_sift_object(self):
        image = np.random.default_rng(16).integers(0, 256, (220, 240), dtype=np.uint8)
        config = _config(sift_n_octave_layers=4, sift_sigma=1.8)
        point = np.array([[110, 110]], np.float32)
        frame = estimate_dense_sift_frames(image, point, config)
        params = dict(config=config, sizes_px=frame["size_px"], angles_deg=frame["angle_deg"],
                      octaves=frame["octave"])
        expected = compute_descriptors_at_points(image, point, **params)
        actual = compute_descriptors_at_points(image, point, sift=cv2.SIFT_create(), **params)
        np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
