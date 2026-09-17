import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from Algorithm.Region_SIFT_Matching import DEFAULT_CONFIG, run_region_sift_matching, with_config
from Algorithm.region_sift_diagnostics import RegionSIFTDiagnostics, score_grid
from Algorithm.region_sift_gt import evaluate_gt, project_gt


class RegionDiagnosticsTests(unittest.TestCase):
    def test_first_failure_opens_window_and_restores_minimized_window(self):
        viewer = RegionSIFTDiagnostics()
        failure = dict(region_debug=None, fail_reason='anchors outside image',
                       region_diagnostic_config=with_config(DEFAULT_CONFIG, cell_width_px=20))
        try:
            viewer.show(failure)
            self.assertIsNotNone(viewer.figure)
            self.assertTrue(plt.fignum_exists(viewer.figure.number))
            self.assertIn('anchors outside image', viewer.figure.texts[0].get_text())
            self.assertIn('20x10px/cell', viewer.figure.texts[0].get_text())
            window = Mock()
            with patch.object(viewer.figure.canvas.manager, 'window', window, create=True), \
                 patch.object(viewer.figure.canvas, 'draw_idle') as draw:
                viewer.show(failure)
                window.deiconify.assert_called_once()
                window.lift.assert_called_once()
                draw.assert_called()
            old_figure = viewer.figure
            viewer._on_close(SimpleNamespace(canvas=old_figure.canvas))
            plt.close(old_figure)
            viewer.show(failure)
            self.assertIsNot(viewer.figure, old_figure)
            viewer.figure.canvas.draw()
        finally:
            plt.close(viewer.figure)

    def test_grid_preserves_invalid_and_singleton_coordinates(self):
        debug = dict(candidate_along_offsets=[2, 0, 4],
                     candidate_across_offsets=[0, 0, 0], scores=[8, 3, np.inf])
        x, y, grid = score_grid(debug, 'scores')
        np.testing.assert_array_equal(x, [0, 2, 4])
        np.testing.assert_array_equal(y, [0])
        np.testing.assert_allclose(grid, [[3, 8, np.nan]])

    def test_rejected_match_renders_and_controls_do_not_change_scores(self):
        rng = np.random.default_rng(39)
        left = rng.integers(0, 256, (180, 220), dtype=np.uint8)
        right = rng.integers(0, 256, (180, 220), dtype=np.uint8)
        cand = dict(K_R=np.eye(3), R_rel=np.eye(3), t_rel=np.zeros((3, 1)),
                    plane_n=np.array([0., 0., 1.]), plane_c=np.array([0., 0., 1.]),
                    F=np.array([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]))
        config = with_config(DEFAULT_CONFIG, auto_scale_orientation=False,
                             keypoint_size_px=3.2, search_length_px=9,
                             search_width_px=5, max_group_score=0.)
        result = run_region_sift_matching(left, right, (110, 90), cand, np.eye(3), config=config)
        self.assertIsNone(result['m_pt'])
        self.assertIn('BestG', result['reject_reason'])
        debug = result['region_debug']
        debug['diagnostic_geometry'] = dict(K_L=np.eye(3), K_R=np.eye(3),
                                            R=np.eye(3), t=np.zeros(3))
        debug['diagnostic_height_plane'] = dict(n=np.array([0., 0., 1.]),
                                               c=np.array([0., 0., 100.]), offset=2.)
        projected = project_gt(debug, 6.5)
        self.assertAlmostEqual(projected['xyz'][2], 108.5)
        exact = evaluate_gt(debug, 6.5)
        self.assertTrue(exact['scoreable'])
        self.assertTrue(exact['inside_band'])
        self.assertAlmostEqual(exact['nearest_distance_px'], 0., places=5)
        self.assertAlmostEqual(exact['objective_score'], exact['nearest_objective'], places=4)
        blocked = dict(debug, reject_specular=True,
                       warped_specular_mask=np.zeros_like(left))
        xy = np.rint(debug['left_points'][0]).astype(int)
        blocked['warped_specular_mask'][xy[1], xy[0]] = 255
        rejected_gt = evaluate_gt(blocked, 6.5)
        self.assertFalse(rejected_gt['scoreable'])
        self.assertEqual(rejected_gt['reason'], 'specular anchor center')
        outside = dict(debug, diagnostic_geometry=dict(debug['diagnostic_geometry'], t=np.array([2000., 0., 0.])))
        self.assertFalse(evaluate_gt(outside, 6.5)['inside_band'])
        behind = dict(debug, diagnostic_height_plane=dict(debug['diagnostic_height_plane'], offset=-200.))
        with self.assertRaises(ValueError):
            project_gt(behind, 0.)
        with self.assertRaises(ValueError):
            project_gt(debug, float('nan'))
        second = debug['second_candidate']
        self.assertIsNotNone(second)
        self.assertEqual(second['distances'].shape, (28,))
        self.assertEqual(int(second['keep_mask'].sum()), 21)
        shift = second['points_warp'] - debug['left_points']
        np.testing.assert_allclose(shift, np.tile(shift[0], (28, 1)), atol=1e-5)
        original = debug['candidate_objective_scores'].copy()
        viewer = RegionSIFTDiagnostics()
        viewer.figure = plt.figure(figsize=(15, 10))
        viewer.result = dict(region_debug=debug, debug_left_gray=left,
                             fail_reason=result['reject_reason'])
        viewer.debug = debug
        viewer.selected = 27
        try:
            viewer._build()
            viewer._submit_gt('6.5')
            self.assertTrue(viewer.gt['scoreable'])
            for index in range(4):
                viewer.checks.set_active(index)
            viewer.radio.set_active(1)
            viewer.figure.canvas.draw()
            point = debug['left_points'][0]
            screen = viewer.crops[0].transData.transform(point)
            viewer._pick(SimpleNamespace(inaxes=viewer.crops[0], xdata=point[0],
                                         x=screen[0], y=screen[1]))
            self.assertEqual(viewer.selected, 0)
            viewer.figure.canvas.draw()
            np.testing.assert_array_equal(original, debug['candidate_objective_scores'])
            viewer._submit_gt('invalid')
            self.assertIsNone(viewer.gt)
            self.assertTrue(viewer.gt_error)
            viewer._submit_gt('6.5')
            debug['second_candidate'] = None
            debug['second_candidate_index'] = None
            # Keep references so garbage collection cannot hide stale callbacks.
            old_widgets = [viewer.checks, viewer.radio, viewer.gt_box, viewer.gt_button]
            viewer.gt_box.begin_typing()
            viewer._build()
            viewer.figure.canvas.draw()
            self.assertFalse(old_widgets[2].capturekeystrokes)
            for widget in old_widgets:
                for cid in widget._cids:
                    self.assertFalse(any(cid in callbacks for callbacks in
                        viewer.figure.canvas.callbacks.callbacks.values()))
            saved_result = viewer.result
            old_checks = viewer.checks
            viewer.show(dict(region_debug=None, fail_reason='no valid support'))
            viewer.figure.canvas.draw()
            self.assertIsNone(viewer.checks)
            self.assertFalse(old_checks.active)
            self.assertEqual(viewer.crops, [])
            viewer.show(saved_result)
            viewer.figure.canvas.draw()
            viewer.checks.set_active(0)
            viewer.radio.set_active(0)
            viewer.figure.canvas.draw()
            # Simulate applying SIFT settings: rebuild for a different group size.
            larger_config = with_config(config, grid_rows=5, grid_cols=5)
            larger = run_region_sift_matching(left, right, (110, 90), cand, np.eye(3), config=larger_config)
            self.assertIsNotNone(larger['region_debug'])
            viewer.show(dict(region_debug=larger['region_debug'], debug_left_gray=left,
                             fail_reason=larger['reject_reason']))
            self.assertEqual(len(viewer.point_sets[0]), 76)
            self.assertEqual(viewer.selected, 75)
            viewer.figure.canvas.draw()
            viewer.gt_box.begin_typing()
            closed_figure = viewer.figure
            viewer._on_close(SimpleNamespace(canvas=closed_figure.canvas))
            plt.close(closed_figure)
            self.assertIsNone(viewer.gt_box)
        finally:
            plt.close(viewer.figure)


if __name__ == '__main__':
    unittest.main()
