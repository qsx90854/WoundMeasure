import ast
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from Algorithm import masked_sift_descriptor as masked
from Algorithm.Region_SIFT_Matching import run_region_sift_matching
from tests import test_masked_sift_descriptor as fixture_module

class FailedDebugTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.MaskedSIFTTests()
        self.fixture.setUp()

    def failed(self, two_stage=True):
        f = self.fixture
        original = masked.compare_descriptors
        def reject(*args, **kwargs):
            result = original(*args, **kwargs)
            # Force the actual group rejection path while retaining real cell evidence.
            result[1][...] = False
            return result
        cfg = replace(f.cfg, two_stage_search=two_stage)
        with patch.object(masked, 'compare_descriptors', side_effect=reject):
            return f.match(config=cfg)

    def test_failed_candidate_never_becomes_measurement(self):
        for two_stage in (False, True):
            result = self.failed(two_stage)
            self.assertIsNone(result['m_pt'])
            debug = result['region_debug']
            self.assertIsNotNone(debug, result['reject_reason'])
            self.assertTrue(debug['diagnostic_only'])
            self.assertEqual(debug['valid_candidate_count'], 0)
            self.assertTrue(np.all(np.isinf(debug['candidate_objective_scores'])))
            self.assertEqual(debug['right_frames']['masked_cell_fraction'].shape,
                             (debug['point_count'], 16))

    def test_diagnostic_prefers_smallest_point_deficit(self):
        f = self.fixture
        original = masked.compare_descriptors
        def varied(*args, **kwargs):
            result = original(*args, **kwargs)
            valid = result[1]
            valid[...] = False
            # Every candidate fails, but the last evaluated one has two valid pairs.
            valid[-1, :2] = True
            return result
        with patch.object(masked, 'compare_descriptors', side_effect=varied):
            result = f.match()
        self.assertIsNone(result['m_pt'])
        self.assertEqual(result['region_debug']['diagnostic_valid_points'], 2)

    def test_selected_cells_render_and_clear(self):
        result = self.failed()
        debug = result['region_debug']
        debug['left_masked_coverage'][0, :4] = 0.
        source = Path('depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py').read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == 'select_region_sift_debug_point')
        fig, (left, right) = plt.subplots(1, 2)
        state = dict(records=debug['point_metadata'], mask_debug=debug,
                     support_artists=[], base_left_text='', base_right_text='')
        ns = dict(np=np, Rectangle=Rectangle, Polygon=Polygon,
                  DEBUG_LEFT_PANEL_ENABLED=True, DEBUG_RIGHT_PANEL_ENABLED=True,
                  debug_region_state=state, ax_debug_A=left, ax_debug_B=right,
                  debug_axis_enabled=lambda ax: True, debug_region_artists=[],
                  dbg_audit_selected_A=left.scatter([], []),
                  dbg_info_A=left.text(0, 0, ''), dbg_info_B=right.text(0, 0, ''))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<selection>', 'exec'), ns)
        try:
            for index in (0, 1, 0):
                self.assertTrue(ns[node.name](index=index))
                fig.canvas.draw()
                self.assertIn('Excluded cells', ns['dbg_info_B'].get_text())
                self.assertEqual(len(state['support_artists']), 66)
        finally:
            plt.close(fig)

if __name__ == '__main__':
    unittest.main()
