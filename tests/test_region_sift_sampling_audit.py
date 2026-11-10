import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Algorithm import masked_sift_descriptor as masked
from Algorithm.region_sift_sampling_audit import sampling_weights, inspect_samples, build_point_audit, SamplingAuditWindow
from tests import test_masked_sift_descriptor as fixture_module

class SamplingAuditTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture_module.MaskedSIFTTests()
        self.f.setUp()

    def test_weights_reproduce_descriptor_coverage_and_mass(self):
        f = self.f
        f.mask[100:110, 125:135] = 255
        context = masked.create_context(f.image, f.mask, f.cfg)
        bad = masked.exclusion_mask(f.mask, f.cfg) > 0
        size = context['entries'][0]['size']
        for angle in (0., 37., 90., 179.):
            for point in ((130.4,110.6),(1.2,2.7)):
                audit = inspect_samples(context, point, size, angle, bad)
                packet = masked.compute_descriptors(context, [point], [size], [angle], f.cfg)
                np.testing.assert_allclose(audit['coverage'],packet['cell_valid_fraction'][0],atol=2e-7)
                mass = (audit['weights']*audit['good']*audit['strength']).sum(axis=(1,2))
                np.testing.assert_allclose(mass, packet['histograms'][0].reshape(16,8).sum(axis=1),rtol=2e-6,atol=.002)

    def test_context_crop_matches_saved_left_and_right_cells(self):
        f = self.f
        f.mask[102:106, 123:129] = 255
        debug = f.match()['region_debug']
        self.assertIsNotNone(debug)
        for index in (0, debug['point_count']-1):
            left, right = build_point_audit(debug,index)
            np.testing.assert_allclose(left['coverage'],debug['left_masked_coverage'][index],atol=2e-7)
            np.testing.assert_allclose(right['coverage'],debug['right_frames']['masked_cell_fraction'][index],atol=2e-7)

    def test_cell_support_overlaps_and_window_interaction(self):
        xx, yy, weights = sampling_weights(4.,0.)
        self.assertTrue(np.any((weights[5]>0)&(weights[6]>0)))
        # Cell (1,1) nominal right edge is x=0, but its support extends to x>0.
        self.assertTrue(np.any((weights[5]>0)&(xx>0)))
        debug = self.f.match()['region_debug']
        with patch('matplotlib.pyplot.show'):
            window = SamplingAuditWindow(debug,0)
        try:
            for cell in (0,5,15):
                window.on_click(SimpleNamespace(inaxes=window.cell_axes[2],xdata=cell%4,ydata=cell//4))
                self.assertEqual(window.cell,cell)
                window.fig.canvas.draw()
                self.assertEqual(len(window.fig.axes),7)
            window.toggle_mass(); self.assertTrue(window.mass)
            window.all_cells(); self.assertIsNone(window.cell)
            window.fig.canvas.draw()
            window.fig.savefig('CodeBackup/descriptor_sampling_audit_20261102_160343_669/audit_preview.png',dpi=110)
        finally:
            plt.close(window.fig)

if __name__ == '__main__':
    unittest.main()
