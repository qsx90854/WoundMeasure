"""Display-only gates; no camera or Tk required."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Algorithm.region_sift_diagnostics import RegionSIFTDiagnostics


class WindowSwitchTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_all_four_combinations(self):
        for search in (False, True):
            for ambiguity in (False, True):
                viewer = RegionSIFTDiagnostics(show_search=search, show_ambiguity=ambiguity)
                result = {'region_debug': {'left_points': [(1,2)]},
                          'debug_left_gray': object(), 'region_search_history': [{'round':0}]}
                with patch('Algorithm.region_sift_search_debug.draw_search_history') as draw, patch.object(viewer, '_build') as build, patch.object(viewer, '_present') as present:
                    viewer.show(result)
                self.assertEqual(draw.call_count, int(search))
                self.assertEqual(build.call_count, int(ambiguity))
                self.assertEqual(present.call_count, int(ambiguity))
                self.assertEqual(viewer.search_figure is not None, search)
                self.assertEqual(viewer.figure is not None, ambiguity)
                self.assertIs(viewer.result, result)
                self.assertEqual(result['region_search_history'], [{'round':0}])
                plt.close('all')

    def test_both_disabled_never_create_even_on_failure(self):
        viewer = RegionSIFTDiagnostics(show_search=False, show_ambiguity=False)
        with patch('Algorithm.region_sift_diagnostics.plt.figure') as create:
            viewer.show({'fail_reason':'no support', 'region_search_history':[{}]})
            viewer.show({'region_debug':None})
        create.assert_not_called()

    def test_disable_existing_windows_independently(self):
        viewer = RegionSIFTDiagnostics()
        with patch('Algorithm.region_sift_search_debug.draw_search_history'):
            viewer.show({'fail_reason':'no support', 'region_search_history':[{}]})
            search = viewer.search_figure
            viewer.show_ambiguity = False
            viewer.show({'region_search_history':[{}]})
            self.assertIsNone(viewer.figure)
            self.assertIs(viewer.search_figure, search)
            self.assertTrue(plt.fignum_exists(search.number))
            viewer.show_search = False
            viewer.show({'region_search_history':[{}]})
            self.assertIsNone(viewer.search_figure)
            self.assertFalse(plt.fignum_exists(search.number))

    def test_manual_close_does_not_close_other_window(self):
        viewer = RegionSIFTDiagnostics()
        viewer.figure = plt.figure()
        viewer.search_figure = plt.figure()
        viewer._on_close(SimpleNamespace(canvas=viewer.figure.canvas))
        self.assertIsNone(viewer.figure)
        self.assertTrue(plt.fignum_exists(viewer.search_figure.number))

    def test_main_wires_both_flags(self):
        path = Path(__file__).parents[1]/'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'RegionSIFTDiagnostics']
        self.assertEqual(len(calls), 1)
        self.assertEqual({k.arg:k.value.id for k in calls[0].keywords}, {
            'show_search':'REGION_SIFT_SHOW_COARSE_FINE_WINDOW',
            'show_ambiguity':'REGION_SIFT_SHOW_AMBIGUITY_WINDOW'})


if __name__ == '__main__':
    unittest.main()
