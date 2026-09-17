"""Headless drag sampling, UI lifecycle and diagnostic replay tests."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from Algorithm.drag_height_profile import DragProfileConfig, DragHeightProfile, ProfileCache, sample_drag_path


class SamplingTests(unittest.TestCase):
    def test_ten_samples_include_endpoints(self):
        points, distances, spacing = sample_drag_path([(10, 10), (190, 10)], DragProfileConfig())
        np.testing.assert_array_equal(points[:, 0], np.arange(10, 191, 20))
        np.testing.assert_array_equal(distances, np.arange(0, 181, 20))
        self.assertEqual(spacing, 20)

    def test_bent_path_uses_arclength_not_endpoint_chord(self):
        points, _, _ = sample_drag_path([(0, 0), (0, 0), (40, 0), (40, 40)], DragProfileConfig())
        np.testing.assert_array_equal(points, [[0, 0], [20, 0], [40, 0], [40, 20], [40, 40]])

    def test_cap_preserves_endpoints_and_tiny_spacing_deduplicates_pixels(self):
        points, _, _ = sample_drag_path([(0, 0), (999, 0)], DragProfileConfig(max_points=7))
        self.assertEqual(len(points), 7)
        np.testing.assert_array_equal(points[[0, -1]], [[0, 0], [999, 0]])
        points, _, _ = sample_drag_path([(0, 0), (3, 0)], DragProfileConfig(spacing_px=.1))
        np.testing.assert_array_equal(points[:, 0], [0, 1, 2, 3])

    def test_short_or_nonfinite_paths_and_invalid_config_are_rejected(self):
        for path in ([(0, 0)], [(0, 0), (1, 0)], [(0, 0), (np.nan, 1)]):
            with self.assertRaises(ValueError):
                sample_drag_path(path, DragProfileConfig())
        with self.assertRaises(ValueError):
            sample_drag_path([(0, 0), (20, 0)], DragProfileConfig(spacing_px=0))

    def test_cache_is_snapshot_and_cleans_own_directory(self):
        cache = ProfileCache()
        directory = Path(cache.directory.name)
        try:
            original = dict(u=12, v=13, region_debug={'scores': np.array([1., 2.])},
                            debug_left_gray=np.zeros((4, 5), np.uint8))
            cache.save(0, original)
            original['region_debug']['scores'][0] = 99
            self.assertEqual(cache.load(0)['region_debug']['scores'][0], 1)
        finally:
            cache.close()
        self.assertFalse(directory.exists())

    def test_real_matcher_diagnostic_snapshot_round_trip(self):
        from Algorithm.Region_SIFT_Matching import DEFAULT_CONFIG, with_config, run_region_sift_matching
        rng = np.random.default_rng(19)
        left = rng.integers(0, 256, (180, 220), dtype=np.uint8)
        cand = dict(K_R=np.eye(3), R_rel=np.eye(3), t_rel=np.zeros((3, 1)),
                    plane_n=np.array([0., 0., 1.]), plane_c=np.array([0., 0., 1.]),
                    F=np.array([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]))
        config = with_config(DEFAULT_CONFIG, auto_scale_orientation=False,
                             keypoint_size_px=3.2, search_length_px=9, search_width_px=5)
        match = run_region_sift_matching(left, left.copy(), (110, 90), cand, np.eye(3), config=config)
        self.assertIsInstance(match['region_debug'], dict)
        result = dict(u=110, v=90, pt=match['m_pt'], region_debug=match['region_debug'],
                      debug_left_gray=left, debug_right_gray=left,
                      region_diagnostic_config=config, fail_reason=match.get('reject_reason'))
        cache = ProfileCache()
        try:
            cache.save(0, result)
            replayed = cache.load(0)
            np.testing.assert_array_equal(replayed['region_debug']['candidate_group_scores'],
                                          result['region_debug']['candidate_group_scores'])
            self.assertEqual(replayed['region_diagnostic_config'], config)
            from Algorithm.region_sift_diagnostics import RegionSIFTDiagnostics
            viewer = RegionSIFTDiagnostics()
            try:
                viewer.show(replayed)
                viewer.figure.canvas.draw()
            finally:
                if viewer.figure is not None:
                    plt.close(viewer.figure)
        finally:
            cache.close()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.fig, (self.left, self.right) = plt.subplots(1, 2)
        for ax in (self.left, self.right):
            ax.set(xlim=(0, 220), ylim=(100, 0))
        button = Button(self.fig.add_axes([.1, .01, .2, .05]), 'Profile', useblit=False)
        self.measure = Mock(side_effect=self.result)
        self.replay, self.restore = Mock(), Mock()
        self.controller = DragHeightProfile(
            self.fig, self.left, self.right, button, DragProfileConfig(),
            activate=Mock(return_value=True), deactivate=self.restore,
            measure=self.measure, replay=self.replay, notify=Mock(), refresh=Mock(),
            image_shape=lambda: (100, 220, 3))
        self.fig.canvas.draw()
        self.addCleanup(plt.close, 'all')
        self.addCleanup(self.controller.close, False)

    def result(self, u, v):
        if u == 50:
            return dict(u=u, v=v, height_display_mm=None, pt=None,
                        region_debug={'best_center_right': [u+5, v]}, fail_reason='Ambiguous')
        return dict(u=u, v=v, height_display_mm=u/20, pt=np.array([u+5, v]),
                    region_debug={'point': u}, fail_reason='')

    def event(self, ax, x, y):
        sx, sy = ax.transData.transform([x, y])
        return SimpleNamespace(inaxes=ax, xdata=x, ydata=y, x=sx, y=sy, button=1)

    def drag(self):
        self.controller.toggle()
        self.assertTrue(self.controller.on_press(self.event(self.left, 10, 10)))
        self.controller.on_motion(self.event(self.left, 100, 10))
        self.controller.on_release(self.event(self.left, 190, 10))

    def finish(self):
        for _ in range(10):
            self.controller._tick()

    def test_ten_point_run_nan_gap_and_right_click_cached_replay(self):
        self.drag()
        self.finish()
        c = self.controller
        self.assertEqual(c.mode, 'review')
        self.assertEqual(self.measure.call_count, 10)
        self.assertEqual(len(c.records), 10)
        self.assertTrue(np.isnan(c.chart_ax.lines[0].get_ydata()[2]))
        self.assertFalse(c.records[2]['trusted'])
        self.assertTrue(c.on_press(self.event(self.right, 95, 10)))
        self.assertEqual(c.selected, 4)
        self.assertEqual(self.replay.call_args.args[0]['u'], 90)
        self.assertEqual(self.measure.call_count, 10)
        c.draw_overlays()
        c.chart.canvas.draw()

    def test_rejected_right_candidate_and_chart_failure_pick(self):
        self.drag()
        self.finish()
        self.controller.on_press(self.event(self.right, 55, 10))
        self.assertEqual(self.replay.call_args.args[0]['fail_reason'], 'Ambiguous')
        self.controller._chart_pick(SimpleNamespace(inaxes=self.controller.chart_ax,
                                                    xdata=3, button=1))
        self.assertEqual(self.controller.selected, 2)

    def test_none_and_exception_results_still_replay_with_left_coordinates(self):
        self.measure.side_effect = [None, RuntimeError('synthetic failure')]
        self.drag()
        self.controller._tick()
        self.controller._tick()
        self.controller.toggle()  # cancel at point boundary
        self.assertEqual(self.controller.mode, 'review')
        self.controller.select(0)
        self.assertEqual(self.replay.call_args.args[0]['u'], 10)
        self.controller.select(1)
        self.assertIn('synthetic failure', self.replay.call_args.args[0]['fail_reason'])

    def test_cancel_partial_then_exit_cleans_cache_and_restores_controls(self):
        self.drag()
        self.controller._tick()
        directory = Path(self.controller.cache.directory.name)
        self.controller.toggle()
        self.assertEqual(len(self.controller.records), 1)
        self.assertEqual(self.controller.mode, 'review')
        self.controller.toggle()
        self.assertFalse(self.controller.enabled)
        self.restore.assert_called_once()
        self.assertFalse(directory.exists())

    def test_new_stroke_discards_previous_profile(self):
        self.drag()
        self.finish()
        directory = Path(self.controller.cache.directory.name)
        self.controller.on_press(self.event(self.left, 10, 80))
        self.assertEqual(self.controller.mode, 'drawing')
        self.assertEqual(len(self.controller.records), 0)
        self.assertFalse(directory.exists())

    def test_close_during_measure(self):
        self.drag()
        self.measure.side_effect = lambda u, v: self.controller.close(redraw=False)
        self.controller._tick()
        self.assertEqual(self.controller.mode, 'off')
        self.assertIsNone(self.controller.cache)

    def test_release_outside_image_uses_last_valid_drag_point(self):
        self.controller.toggle()
        self.controller.on_press(self.event(self.left, 10, 10))
        self.controller.on_motion(self.event(self.left, 190, 10))
        self.controller.on_release(SimpleNamespace(inaxes=None, xdata=None, ydata=None, button=1))
        self.assertEqual(self.controller.mode, 'running')
        self.assertEqual(len(self.controller.samples), 10)

    def test_failed_activation_leaves_mode_off(self):
        self.controller.activate.return_value = False
        self.controller.toggle()
        self.assertEqual(self.controller.mode, 'off')
        self.measure.assert_not_called()


if __name__ == '__main__':
    unittest.main()
