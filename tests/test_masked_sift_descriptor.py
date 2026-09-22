import ast
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from Algorithm import masked_sift_descriptor as masked, region_sift_frames as frames
from Algorithm.Region_SIFT_Matching import (
    DEFAULT_CONFIG, run_region_sift_matching, _left_context_roi, _right_context_roi)
from Algorithm.region_sift_settings import parse_config_values, LABELS


class MaskedSIFTTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(41).integers(0, 256, (220, 260), dtype=np.uint8)
        self.mask = np.zeros_like(self.image)
        self.cfg = replace(DEFAULT_CONFIG, use_masked_sift=True,
            scale_keypoint_sizes_px=(3.2, 4., 6.4), search_length_px=31.,
            adaptive_max_expansions=0)
        self.points = np.array([[130., 110.], [135., 112.]], np.float32)
        self.cand = dict(K_R=np.eye(3), R_rel=np.eye(3), t_rel=np.zeros((3, 1)),
            plane_n=np.array([0., 0., 1.]), plane_c=np.array([0., 0., 1.]),
            F=np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]))

    def packet(self, image=None, mask=None, cfg=None):
        image = self.image if image is None else image
        cfg = self.cfg if cfg is None else cfg
        context = masked.create_context(image, self.mask if mask is None else mask, cfg)
        f = frames.estimate_dense_sift_frames(image, self.points, cfg, context=context)
        p = masked.compute_descriptors(context, self.points, f['size_px'], f['angle_deg'], cfg)
        return f, p

    def match(self, config=None, **kwargs):
        return run_region_sift_matching(self.image, self.image.copy(), (130, 110),
            self.cand, np.eye(3), config=self.cfg if config is None else config,
            left_spec_mask=self.mask, right_spec_mask=self.mask, **kwargs)

    def test_fixed_mask_pixel_values_cannot_influence_frames_or_descriptor(self):
        self.mask[98:105, 118:125] = 255
        changed = self.image.copy()
        changed[self.mask > 0] = 255-changed[self.mask > 0]
        a, da = self.packet()
        b, db = self.packet(changed)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for key in da:
            np.testing.assert_array_equal(da[key], db[key])

    def test_flat_valid_pixels_are_not_missing_or_fake_edges(self):
        self.mask[100:105, 119:124] = 255
        image = np.full_like(self.image, 100)
        image[self.mask > 0] = 255
        _, packet = self.packet(image)
        self.assertTrue(packet['valid_points'].all())
        np.testing.assert_array_equal(packet['descriptors'], 0)
        self.assertTrue(np.all(packet['valid_fraction'] > .6))

    def test_empty_mask_descriptor_shape_norm_and_self_distance(self):
        _, packet = self.packet()
        self.assertEqual(packet['descriptors'].shape, (2, 128))
        np.testing.assert_allclose(np.linalg.norm(packet['descriptors'], axis=1), 512, atol=1e-4)
        d, valid, common = masked.compare_descriptors(packet, packet, self.cfg)
        np.testing.assert_array_equal(d, 0)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(common, 1)

    def test_all_masked_is_invalid_not_a_good_zero_match(self):
        _, packet = self.packet(mask=np.ones_like(self.mask))
        self.assertFalse(packet['valid_points'].any())
        d, valid, _ = masked.compare_descriptors(packet, packet, self.cfg)
        self.assertTrue(np.isfinite(d).all())
        self.assertFalse(valid.any())

    def test_common_support_rejects_disjoint_evidence(self):
        _, a = self.packet()
        b = {k: v.copy() for k, v in a.items()}
        a['cell_valid_fraction'][:, :8] = 0
        b['cell_valid_fraction'][:, 8:] = 0
        _, valid, common = masked.compare_descriptors(a, b, self.cfg)
        self.assertFalse(valid.any())
        np.testing.assert_array_equal(common, 0)

    def test_no_mask_opencv_direction_and_bin_layout(self):
        cfg = replace(self.cfg, auto_scale_orientation=False, keypoint_size_px=3.2)
        context = masked.create_context(self.image, self.mask, cfg)
        a = masked.compute_descriptors(context, self.points, [3.2, 3.2], [30, 135], cfg)['descriptors']
        b = frames.compute_descriptors_at_points(self.image, self.points, config=cfg,
            sizes_px=[3.2, 3.2], angles_deg=[30, 135])
        cosine = np.sum(a*b, axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
        self.assertTrue(np.all(cosine > .98), cosine)

    def test_rotation_and_angle_compensation(self):
        cfg = replace(self.cfg, auto_scale_orientation=False, keypoint_size_px=3.2)
        image = self.image[:180, :180]
        rotated = np.rot90(image).copy()
        p = np.array([[90., 90.]])
        a = masked.compute_descriptors(masked.create_context(image, np.zeros_like(image), cfg), p, [3.2], [30], cfg)
        b = masked.compute_descriptors(masked.create_context(rotated, np.zeros_like(rotated), cfg), [[90., 89.]], [3.2], [300], cfg)
        np.testing.assert_allclose(a['descriptors'], b['descriptors'], atol=1e-4)

    def test_invalid_warp_pixels_cannot_influence_maps(self):
        validity = np.ones_like(self.mask)
        validity[:, :100] = 0
        changed = self.image.copy()
        changed[:, :100] = 255
        a = masked.create_context(self.image, self.mask, self.cfg, validity)
        b = masked.create_context(changed, self.mask, self.cfg, validity)
        for ea, eb in zip(a['entries'], b['entries']):
            np.testing.assert_array_equal(ea['gx'], eb['gx'])
            np.testing.assert_array_equal(ea['response'], eb['response'])

    def test_full_and_coarsefine_use_custom_not_opencv_compute(self):
        for fast in (False, True):
            with patch('Algorithm.Region_SIFT_Matching.compute_descriptors_at_points', side_effect=AssertionError('OpenCV compute used')):
                r = self.match(replace(self.cfg, two_stage_search=fast))
            self.assertIsNotNone(r['m_pt'], r['reject_reason'])
            np.testing.assert_allclose(r['m_pt'], [130, 110], atol=1e-4)
            self.assertEqual(r['descriptor_backend'], 'custom-masked')
            self.assertTrue(r['timing_counts']['reject_specular'])
            self.assertFalse(r['timing_counts']['specular_check_support'])
            self.assertIn('masked_common_fraction', r['region_debug']['right_frames'])

    def test_cache_backend_and_parameter_changes_do_not_reuse_wrong_descriptors(self):
        cache = {}
        self.match(left_cache=cache)
        self.assertTrue(self.match(left_cache=cache)['timing_counts']['left_cache_hit'])
        other = self.match(replace(self.cfg, use_masked_sift=False), left_cache=cache)
        self.assertFalse(other['timing_counts']['left_cache_hit'])
        self.assertEqual(other['descriptor_backend'], 'opencv')
        again = self.match(left_cache=cache)
        self.assertFalse(again['timing_counts']['left_cache_hit'])
        changed = self.match(replace(self.cfg, masked_min_common_fraction=.6), left_cache=cache)
        self.assertFalse(changed['timing_counts']['left_cache_hit'])

    def test_missing_mask_fails_explicitly(self):
        r = run_region_sift_matching(self.image, self.image, (130, 110), self.cand,
                                     np.eye(3), config=self.cfg)
        self.assertIsNone(r['m_pt'])
        self.assertIn('mask missing', r['reject_reason'])

    def test_gt_uses_custom_backend_and_reproduces_candidate_score(self):
        from Algorithm.region_sift_gt import evaluate_gt
        self.mask[102:106, 105:109] = 255
        debug = self.match()['region_debug']
        self.assertIsNotNone(debug)
        debug['diagnostic_geometry'] = dict(K_L=np.eye(3), K_R=np.eye(3),
                                            R=np.eye(3), t=np.zeros(3))
        debug['diagnostic_height_plane'] = dict(n=np.array([0., 0., 1.]),
                                               c=np.array([0., 0., 100.]), offset=0.)
        with patch('Algorithm.region_sift_gt.compute_descriptors_at_points', side_effect=AssertionError('OpenCV used')):
            exact = evaluate_gt(debug, 6.5)
        self.assertTrue(exact['scoreable'], exact['reason'])
        self.assertAlmostEqual(exact['objective_score'], exact['nearest_objective'], places=4)

    def test_extra_margin_is_effective_and_input_is_not_modified(self):
        self.mask[100, 100] = 255
        before = self.mask.copy()
        expanded = masked.exclusion_mask(self.mask, replace(self.cfg, masked_extra_margin_px=2))
        self.assertEqual(np.count_nonzero(expanded), 25)
        np.testing.assert_array_equal(self.mask, before)

    def test_roi_crop_matches_full_image_left_descriptor(self):
        """Tier-1 perf change: cropping the pyramid input to a safe ROI around
        the anchor must reproduce the exact frames/descriptor computed on the
        full image -- proves the crop margin (max_support_radius +
        masked_extra_margin_px) is sufficient and origin bookkeeping correct."""
        from Algorithm import region_sift_search as search
        self.mask[95:108, 115:128] = 255
        full_context = masked.create_context(self.image, self.mask, self.cfg)
        margin = masked.max_support_radius(self.cfg) + self.cfg.masked_extra_margin_px + 2
        x0 = max(0, int(self.points[:, 0].min()) - margin)
        y0 = max(0, int(self.points[:, 1].min()) - margin)
        x1 = min(self.image.shape[1], int(self.points[:, 0].max()) + margin + 1)
        y1 = min(self.image.shape[0], int(self.points[:, 1].max()) + margin + 1)
        self.assertLess(x1 - x0, self.image.shape[1])  # the crop is a proper subset
        self.assertLess(y1 - y0, self.image.shape[0])
        cropped_context = search.frame_context(
            self.image, self.cfg, None, 'left', self.mask, roi=(x0, y0, x1, y1))
        self.assertEqual(tuple(cropped_context['origin']), (x0, y0))
        origin = np.asarray(cropped_context['origin'], np.float32)
        local_points = self.points - origin
        f_full = frames.estimate_dense_sift_frames(self.image, self.points, self.cfg, context=full_context)
        f_crop = frames.estimate_dense_sift_frames(
            self.image[y0:y1, x0:x1], local_points, self.cfg, context=cropped_context)
        for key in f_full:
            np.testing.assert_array_equal(f_full[key], f_crop[key])
        d_full = masked.compute_descriptors(
            full_context, self.points, f_full['size_px'], f_full['angle_deg'], self.cfg)
        d_crop = masked.compute_descriptors(
            cropped_context, local_points, f_crop['size_px'], f_crop['angle_deg'], self.cfg)
        for key in d_full:
            np.testing.assert_array_equal(d_full[key], d_crop[key])

    def test_left_context_roi_skips_crop_when_quantization_is_active(self):
        """frame_coordinate_quantization_px rounds to a step generally
        unaligned with the crop's pixel origin, so cropping first could pick a
        different scale/angle than the full image would (found in review of
        the Tier-1 perf change). _left_context_roi must fall back to the full
        image (None) whenever that knob is on, and still crop when it is off."""
        left_roi = (100, 90, 30, 30)
        self.assertIsNotNone(_left_context_roi(self.cfg, left_roi, self.image.shape))
        quantized = replace(self.cfg, frame_coordinate_quantization_px=1.5)
        self.assertIsNone(_left_context_roi(quantized, left_roi, self.image.shape))
        opencv_cfg = replace(self.cfg, use_masked_sift=False)
        self.assertIsNone(_left_context_roi(opencv_cfg, left_roi, self.image.shape))

    def test_right_context_roi_skips_crop_when_quantization_is_active(self):
        """Mirrors test_left_context_roi_skips_crop_when_quantization_is_active
        for the right-side ROI (gap flagged in independent review of the
        Tier-1 perf change: only the end-to-end path was covered there)."""
        band = dict(seed_on_line=np.array([130., 110.], np.float32),
                    tangent=np.array([1., 0.], np.float32),
                    normal=np.array([0., 1.], np.float32))
        point_offsets = np.zeros((28, 2), np.float32)
        self.assertIsNotNone(_right_context_roi(self.cfg, band, point_offsets, self.image.shape))
        quantized = replace(self.cfg, frame_coordinate_quantization_px=1.5)
        self.assertIsNone(_right_context_roi(quantized, band, point_offsets, self.image.shape))
        opencv_cfg = replace(self.cfg, use_masked_sift=False)
        self.assertIsNone(_right_context_roi(opencv_cfg, band, point_offsets, self.image.shape))

    def test_right_roi_crop_matches_full_image_search(self):
        """Tier-1 perf change (right side): cropping the right-image pyramid
        input to a safe ROI must reproduce exactly the same match as
        searching on the full warped image -- proves the crop (built from
        the maximal search-band polygon, bounded by search_length_px/
        search_width_px regardless of stage or adaptive expansion) covers
        every candidate the whole search could ever sample. Forcing
        _right_context_roi to return None gets the pre-optimization
        reference without duplicating run_region_sift_matching."""
        from Algorithm import Region_SIFT_Matching as matcher
        cropped = self.match()
        self.assertIsNotNone(cropped['m_pt'], cropped['reject_reason'])
        with patch.object(matcher, '_right_context_roi', return_value=None):
            full = self.match()
        self.assertIsNotNone(full['m_pt'], full['reject_reason'])
        np.testing.assert_array_equal(cropped['m_pt'], full['m_pt'])
        np.testing.assert_array_equal(
            cropped['region_debug']['distances'], full['region_debug']['distances'])
        self.assertEqual(cropped['method'], full['method'])

    def test_right_reflected_anchor_center_is_not_whole_candidate_veto(self):
        right_mask = self.mask.copy()
        right_mask[110, 130] = 255
        cfg = replace(self.cfg, search_length_px=5, search_width_px=1,
                      auto_scale_orientation=False, keypoint_size_px=3.2,
                      max_objective_score_ratio=None)
        r = run_region_sift_matching(self.image, self.image, (130, 110), self.cand,
            np.eye(3), config=cfg, left_spec_mask=self.mask, right_spec_mask=right_mask)
        self.assertIsNotNone(r['region_debug'], r['reject_reason'])
        self.assertEqual(r['timing_counts']['specular_center_rejected_candidates'], 0)
        self.assertGreater(r['region_debug']['valid_candidate_count'], 0)

    def test_left_click_on_specular_is_still_rejected(self):
        self.mask[110, 130] = 255
        result = self.match()
        self.assertIsNone(result['m_pt'])
        self.assertIn('P overlaps specular', result['reject_reason'])

    def test_group_gate_allows_few_missing_but_requires_best_k(self):
        metadata = [dict(cell_row=i//3, cell_col=i%3) for i in range(9) for _ in range(3)]
        metadata.append(dict(cell_row=1, cell_col=1))
        valid = np.ones((1, 28), bool)
        valid[0, np.arange(7)*3] = False
        self.assertTrue(masked.group_validity(valid, metadata, self.cfg)[0])
        valid[0, 21] = False
        self.assertFalse(masked.group_validity(valid, metadata, self.cfg)[0])

    def test_group_gate_tolerates_up_to_configured_deficient_cells(self):
        """masked_max_deficient_cells=0 reproduces the original all-cells rule;
        the default (1) rescues a single specular-covered cell but still vetoes
        a second one, and a deficient cell still pays the ceiling-distance
        penalty in CellAll (see test_lost_coverage_cannot_improve_fixed_histogram_score)."""
        metadata = [dict(cell_row=i//3, cell_col=i%3) for i in range(9) for _ in range(3)]
        metadata.append(dict(cell_row=1, cell_col=1))
        valid = np.ones((1, 28), bool)
        valid[0, :3] = False  # cell (0,0) fully empty; 25/28 pairs remain
        strict = replace(self.cfg, masked_max_deficient_cells=0)
        self.assertFalse(masked.group_validity(valid, metadata, strict)[0])
        details = masked.group_validity(valid, metadata, self.cfg, return_details=True)
        self.assertTrue(details['valid'][0])
        self.assertEqual(details['deficient_cell_count'][0], 1)
        valid[0, 3:6] = False  # cell (0,1) also fully empty
        self.assertFalse(masked.group_validity(valid, metadata, self.cfg)[0])
        lenient = replace(self.cfg, masked_max_deficient_cells=2)
        self.assertTrue(masked.group_validity(valid, metadata, lenient)[0])

    def test_lost_coverage_cannot_improve_fixed_histogram_score(self):
        _, a = self.packet()
        b = {k: v.copy() for k, v in a.items()}
        original, _, _ = masked.compare_descriptors(a, b, self.cfg)
        b['cell_valid_fraction'][:] = .95
        partial, usable, _ = masked.compare_descriptors(a, b, self.cfg)
        self.assertTrue(usable.all())
        self.assertTrue(np.all(partial > original))
        b['valid_points'][0] = False
        missing, usable, _ = masked.compare_descriptors(a, b, self.cfg)
        self.assertFalse(usable[0])
        self.assertAlmostEqual(missing[0], 512*np.sqrt(2), places=3)
        self.assertGreaterEqual(missing[0], partial[0])

    def test_reflected_center_still_uses_only_clean_surrounding_pixels(self):
        self.mask[109:112, 129:132] = 255
        changed = self.image.copy()
        changed[self.mask > 0] = 255-changed[self.mask > 0]
        f1, d1 = self.packet()
        f2, d2 = self.packet(changed)
        self.assertTrue(f1['valid'][0])
        self.assertTrue(d1['valid_points'][0])
        for key in f1:
            np.testing.assert_array_equal(f1[key], f2[key])
        for key in d1:
            np.testing.assert_array_equal(d1[key], d2[key])

    def test_pipeline_keeps_27_valid_pairs_and_never_keeps_missing_pair(self):
        original = masked.compare_descriptors
        def one_missing(left, right, config):
            right['valid_points'][..., 0] = False
            return original(left, right, config)
        with patch.object(masked, 'compare_descriptors', side_effect=one_missing):
            result = self.match(replace(self.cfg, search_length_px=5, search_width_px=1,
                                        max_objective_score_ratio=None))
        self.assertIsNotNone(result['m_pt'], result['reject_reason'])
        d = result['region_debug']
        self.assertEqual(np.count_nonzero(d['right_frames']['masked_point_valid']), 27)
        self.assertFalse(d['keep_mask'][0])
        self.assertFalse(d['point_metadata'][0]['masked_pair_valid'])
        self.assertAlmostEqual(d['distances'][0], 512*np.sqrt(2), places=3)
        self.assertFalse(d['right_frames']['scale_reliable'][0])

    def test_custom_parameters_are_labelled_and_validated(self):
        for name in ('use_masked_sift', 'masked_extra_margin_px', 'masked_min_blur_weight',
                     'masked_min_valid_fraction', 'masked_min_cell_fraction',
                     'masked_min_common_fraction', 'masked_descriptor_clip',
                     'masked_max_deficient_cells'):
            self.assertIn('自製算子', LABELS[name])
        for changes in ({'masked_min_blur_weight': '0'}, {'masked_extra_margin_px': '-1'},
                        {'masked_min_cell_fraction': '1.1'}, {'masked_descriptor_clip': 'nan'},
                        {'masked_max_deficient_cells': '-1'}):
            with self.assertRaises(ValueError):
                parse_config_values(self.cfg, changes)
        self.assertFalse(parse_config_values(self.cfg, {'use_masked_sift': 'False'}).use_masked_sift)

    def test_ui_toggle_calls_same_atomic_settings_apply(self):
        path = Path(__file__).resolve().parents[1] / 'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'on_sift_backend')
        received = []
        namespace = dict(REGION_SIFT_CONFIG=self.cfg, apply_region_settings=received.append)
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(path), 'exec'), namespace)
        namespace['on_sift_backend'](None)
        self.assertEqual(received, [replace(self.cfg, use_masked_sift=False)])


if __name__ == '__main__':
    unittest.main()
