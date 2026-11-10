import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

from Algorithm.Region_SIFT_Matching import DEFAULT_CONFIG
from Algorithm.region_sift_disparity_debug import DisparityAuditViewer, summarize_table


class CenterAnchorOverlayTests(unittest.TestCase):
    def result(self):
        offsets = np.arange(-2., 3.)
        costs = np.array([[8, 8], [3, 6], [5, 3], [7, 5], [9, 8.]])
        candidate = dict(summary=summarize_table(offsets, costs), costs=costs,
                         objective=costs.mean(axis=1), valid_counts=np.full(5, 2),
                         label='Top 1', along=0., across=0.)
        audit = dict(status='ok', candidates=[candidate, dict(candidate, label='Top 2')],
                     offsets=offsets, anchors=np.array([[45., 45.], [50., 50.]]),
                     point=np.array([50., 50.]), config=DEFAULT_CONFIG,
                     left_roi=(35, 35, 30, 30), click_index=1,
                     metadata=[dict(cell_row=0, cell_col=0), dict(cell_row=1, cell_col=1)],
                     center_mask=np.array([False, True]), accepted=True,
                     descriptor_rows=10, elapsed_ms=1.)
        center_costs = costs.copy()
        center_costs[:, 1] = np.nan
        center = dict(points=np.array([[48., 49.], [50., 50.]]), offsets=offsets,
                      point_costs=center_costs, roi=(44, 44, 12, 12))
        return dict(debug_left_gray=np.zeros((100, 100)),
                    region_debug=dict(disparity_audit=audit, center_refinement=center))

    def test_top1_draws_center_points_top2_does_not(self):
        viewer = DisparityAuditViewer()
        result = self.result()
        original = result['region_debug']['center_refinement']['point_costs'].copy()
        try:
            with patch('matplotlib.figure.Figure.show'):
                viewer.show(result, enabled=True)
                ax = viewer.figure.axes[0]
                self.assertIn('C0', [t.get_text() for t in ax.texts])
                self.assertIn('C1', [t.get_text() for t in ax.texts])
                # Invalid center evidence is hollow gray, never a colored valid dot.
                np.testing.assert_allclose(ax.collections[-1].get_edgecolors()[0, :3], [.5]*3, atol=.01)
                viewer._switch(None)
                self.assertNotIn('C0', [t.get_text() for t in viewer.figure.axes[0].texts])
                self.assertIn('Top 1 only', viewer.figure.axes[0].get_title())
            np.testing.assert_array_equal(original, result['region_debug']['center_refinement']['point_costs'])
        finally:
            viewer.close()

    def test_disabled_or_old_result(self):
        viewer = DisparityAuditViewer()
        with patch('matplotlib.pyplot.figure') as figure:
            viewer.show({}, enabled=False)
            figure.assert_not_called()
        result = self.result()
        del result['region_debug']['center_refinement']
        try:
            with patch('matplotlib.figure.Figure.show'):
                viewer.show(result, enabled=True)
            self.assertNotIn('C0', [t.get_text() for t in viewer.figure.axes[0].texts])
        finally:
            viewer.close()


if __name__ == '__main__':
    unittest.main()
