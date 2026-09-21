"""Navigation cache regression tests, without camera/models or a GUI."""
import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from Algorithm.paired_view_renderer import PairedViewRenderer


class PairedRendererTests(unittest.TestCase):
    def setUp(self):
        self.fig, axes = plt.subplots(2, 2, figsize=(6, 5), dpi=80)
        self.axes = axes
        self.fig.subplots_adjust(hspace=.4)
        rng = np.random.default_rng(1)
        for ax in axes.flat:
            ax.imshow(rng.integers(0, 255, (80, 100, 3), dtype=np.uint8), animated=True)
            ax.scatter([20, 60], [30, 40], color='red')
            ax.plot([10, 70], [60, 15], animated=True, color='cyan')
            ax.set_title('ROI')
            ax.axis('off')
        for left, right in axes:
            right.add_artist(ConnectionPatch((20, 30), (60, 40), 'data',
                             axesA=left, axesB=right, color='yellow'))
        self.hud = self.fig.text(.45, .5, 'HUD', animated=True)
        self.renderer = PairedViewRenderer(self.fig, axes, (self.hud,))

    def tearDown(self):
        plt.close(self.fig)

    def pixels(self):
        return np.asarray(self.fig.canvas.buffer_rgba()).copy()

    def assert_matches_full(self, cached):
        self.renderer.draw(None)
        np.testing.assert_array_equal(cached, self.pixels())

    def test_zoom_pan_and_pair_switch_match_full_render(self):
        self.renderer.draw(0)
        for index in (0, 0, 1, 1, 0):
            self.axes[index, 0].set_xlim(10, 75)
            self.axes[index, 1].set_ylim(65, 5)
            self.renderer.draw(index)
            cached = self.pixels()
            self.assert_matches_full(cached)
            self.renderer.draw(index)

    def test_warm_zoom_does_not_draw_other_pair_or_full_canvas(self):
        self.renderer.draw(0)
        self.axes[0, 0].set_xlim(15, 70)
        with patch.object(self.fig.canvas, 'draw', wraps=self.fig.canvas.draw) as full, \
             patch.object(self.axes[1, 0], 'draw', wraps=self.axes[1, 0].draw) as other:
            self.assertFalse(self.renderer.draw(0))
            full.assert_not_called()
            other.assert_not_called()
        self.assert_matches_full(self.pixels())

    def test_resize_visibility_content_and_hud_invalidation(self):
        self.renderer.draw(0)
        self.fig.set_size_inches(7, 6)
        self.assertTrue(self.renderer.draw(0))
        self.assert_matches_full(self.pixels())
        self.axes[0, 1].set_visible(False)
        self.axes[1, 0].images[0].set_data(np.zeros((80, 100, 3), np.uint8))
        self.hud.set_position((.2, .7))
        self.renderer.invalidate()
        self.renderer.draw(1)
        self.assert_matches_full(self.pixels())

    def test_animated_state_restored(self):
        artists = [self.hud] + [a for ax in self.axes.flat for a in ax.get_children()]
        before = [a.get_animated() for a in artists]
        self.renderer.draw(0)
        self.assertEqual(before, [a.get_animated() for a in artists])


if __name__ == '__main__':
    unittest.main()
