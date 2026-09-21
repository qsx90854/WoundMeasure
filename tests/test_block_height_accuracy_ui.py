"""Headless callback tests: no camera, video loading, Tk window or live UI.

Extract the actual nested callbacks into a small closure and supply controlled
measurement/widget doubles, so production event wiring logic is exercised.
"""
import ast
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button, RadioButtons, TextBox
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba
import numpy as np

from Algorithm.block_height_accuracy import (
    BlockAccuracyConfig, BlockAccuracyReport, build_block_plan, measurement_record,
    block_mae_color, summarize_block_records,
)


SOURCE = Path(__file__).resolve().parents[1] / 'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'


class FakeWidget:
    def __init__(self):
        self.active = True
        self.text = ''
        self.label = SimpleNamespace(set_text=Mock())

    def set_active(self, active):
        self.active = active


class BlockAccuracyCallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.figure = Figure()
        self.axis = self.figure.add_subplot()
        self.timer = SimpleNamespace(start=Mock(), stop=Mock(), add_callback=Mock())
        self.state = dict(mode='idle', corners=[], plan=None, report=None, timer=None,
                          artists=[], disabled_widgets=[], started=None,
                          summaries=None, show_mae_colors=False)
        self.ns = dict(np=np, time=time, Polygon=Polygon, BlockAccuracyConfig=BlockAccuracyConfig,
                       BlockAccuracyReport=BlockAccuracyReport, build_block_plan=build_block_plan,
                       measurement_record=measurement_record, block_mae_color=block_mae_color,
                       summarize_block_records=summarize_block_records,
                       block_accuracy_state=self.state,
                       height_profile=None,
                       ax_A=self.axis, fig=SimpleNamespace(canvas=SimpleNamespace(new_timer=Mock(return_value=self.timer))),
                       depth_text=SimpleNamespace(set_text=Mock()), request_blit_refresh=Mock(),
                       locked_L_clean=np.zeros((600, 800, 3), np.uint8),
                       BLOCK_ACCURACY_CONFIG=BlockAccuracyConfig(samples_per_axis=3),
                       BLOCK_ACCURACY_OUTPUT_DIR=Path(self.temp.name),
                       custom_plane_mode=False, custom_plane_fitted=False,
                       custom_plane_n=None, custom_plane_c=None,
                       view_state={'manual': False, 'show_rt_warp_view': False, 'region_sift': True},
                       current_cand={'idx': 102}, extra_candidates_list=[],
                       get_selected_height_plane=lambda: (np.array([0, 0, 1]), np.zeros(3), 'Shared Pattern Plane'),
                       VIDEO_PATH='test.mp4', MEASURE_MODE='dual_direct',
                       DEFAULT_WOUND_HEIGHT_OFFSET_MM=0, shared_height_plane_diag={},
                       REGION_SIFT_CONFIG=SimpleNamespace(group_balance_weight=0.35, use_masked_sift=False), KL=np.eye(3),
                       scatter_A=SimpleNamespace(set_offsets=Mock()),
                       do_measure=Mock(return_value={'height_display_mm': 12.5, 'fail_reason': ''}))
        widget_names = [f'c{i}' for i in range(1, 20)] + [
            'radio_mode', 'text_box', 'btn_lock_L', 'btn_lock_R', 'btn_hide_R', 'btn_norm_toggle',
            'btn_calc', 'btn_auto_calc', 'btn_grad_toggle', 'btn_custom_plane', 'btn_high_grad_pts',
            'btn_mid_grad_pts', 'btn_rt_diff', 'btn_return_menu', 'btn_wound_toggle',
            'btn_wound_pts_toggle', 'btn_aruco_overlay', 'btn_rt_sift', 'btn_height_plane',
            'btn_metric_blocks', 'btn_shared_plane', 'btn_top2_geo', 'btn_h_residual',
            'btn_rt_warp_view', 'btn_block_accuracy', 'btn_block_cancel', 'btn_block_mae',
            'text_region_u', 'text_region_v', 'btn_region_replay', 'btn_region_settings',
            'btn_height_profile', 'btn_sift_backend', 'btn_specular_settings',
            'btn_specular_v2', 'btn_specular_v2_settings']
        self.ns.update({name: FakeWidget() for name in widget_names})
        tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
        names = ['on_region_coordinate_replay', 'apply_region_settings',
                 'clear_block_accuracy_overlay', 'block_accuracy_control_widgets', 'lock_block_accuracy_controls',
                 'draw_block_accuracy_plan', 'pick_block_accuracy_corner', 'finish_block_accuracy',
                 'run_next_block_accuracy_point', 'on_block_accuracy', 'on_block_accuracy_close',
                 'on_block_mae_toggle']
        functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names}
        factory = ast.parse('def factory():\n    auto_calc_active = False\n').body[0]
        factory.body.extend(functions[name] for name in names)
        factory.body.append(ast.Return(ast.Dict(keys=[ast.Constant(name) for name in names],
                                               values=[ast.Name(name, ast.Load()) for name in names])))
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        exec(compile(module, str(SOURCE), 'exec'), self.ns)
        self.callbacks = self.ns['factory']()
        self.print_patch = patch('builtins.print')
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def start_preview(self):
        self.callbacks['on_block_accuracy'](None)
        for point in ((100, 100), (700, 100), (700, 500), (100, 500)):
            self.callbacks['pick_block_accuracy_corner'](*point)
        self.assertEqual(self.state['mode'], 'preview')
        self.assertEqual(len(self.state['plan']['samples']), 216)
        self.ns['do_measure'].assert_not_called()

    def test_preview_confirm_measure_cancel_and_restore_controls(self):
        self.start_preview()
        self.assertFalse(self.ns['c18'].active)
        self.assertFalse(self.ns['text_region_u'].active)
        self.assertFalse(self.ns['text_region_v'].active)
        self.assertFalse(self.ns['btn_region_replay'].active)
        self.assertFalse(self.ns['btn_region_settings'].active)
        self.assertFalse(self.ns['btn_height_profile'].active)
        self.callbacks['on_block_accuracy'](None)
        self.assertEqual(self.state['mode'], 'running')
        self.timer.start.assert_called_once()
        self.callbacks['run_next_block_accuracy_point']()
        self.ns['do_measure'].assert_called_once_with(125.0, 125.0, accuracy_batch=True)
        self.assertEqual(len(self.state['report'].records), 1)
        self.callbacks['finish_block_accuracy']('cancelled', 'test')
        self.assertEqual(self.state['mode'], 'idle')
        self.assertTrue(self.ns['c18'].active)
        self.assertTrue(self.ns['text_region_u'].active)
        self.assertTrue(self.ns['text_region_v'].active)
        self.assertTrue(self.ns['btn_region_replay'].active)
        self.assertTrue(self.ns['btn_region_settings'].active)
        self.assertTrue(self.ns['btn_height_profile'].active)
        self.assertTrue(self.state['report'].closed)
        self.assertTrue((self.state['report'].directory / 'blocks.csv').exists())

    def test_repeated_button_while_picking_does_not_lose_original_widget_states(self):
        self.callbacks['on_block_accuracy'](None)
        self.callbacks['on_block_accuracy'](None)
        self.callbacks['finish_block_accuracy']()
        self.assertTrue(self.ns['c1'].active)

    def test_region_coordinate_replay_uses_batch_pixel_rounding(self):
        self.ns['text_region_u'].text = '320.6'
        self.ns['text_region_v'].text = '123.4'
        self.callbacks['on_region_coordinate_replay'](None)
        self.ns['do_measure'].assert_called_once_with(321, 123)

    def test_region_settings_apply_is_atomic_and_blocked_during_batch(self):
        previous = self.ns['REGION_SIFT_CONFIG']
        updated = SimpleNamespace(group_balance_weight=0.5, use_masked_sift=True)
        self.state['mode'] = 'running'
        with self.assertRaises(ValueError):
            self.callbacks['apply_region_settings'](updated)
        self.assertIs(self.ns['REGION_SIFT_CONFIG'], previous)
        self.state['mode'] = 'idle'
        self.callbacks['apply_region_settings'](updated)
        self.assertIs(self.ns['REGION_SIFT_CONFIG'], updated)
        self.ns['btn_sift_backend'].label.set_text.assert_called_with('Descriptor: 自製')
        self.ns['do_measure'].assert_not_called()

    def test_region_coordinate_replay_rejects_invalid_mode_and_coordinates(self):
        self.ns['text_region_u'].text = '10'
        self.ns['text_region_v'].text = '20'
        self.state['mode'] = 'running'
        self.callbacks['on_region_coordinate_replay'](None)
        self.ns['do_measure'].assert_not_called()

        self.state['mode'] = 'idle'
        self.ns['view_state']['region_sift'] = False
        self.callbacks['on_region_coordinate_replay'](None)
        self.ns['do_measure'].assert_not_called()

        self.ns['view_state']['region_sift'] = True
        for u, v in (('', '20'), ('nan', '20'), ('800', '20'), ('10', '600')):
            self.ns['text_region_u'].text = u
            self.ns['text_region_v'].text = v
            self.callbacks['on_region_coordinate_replay'](None)
        self.ns['do_measure'].assert_not_called()

    def test_existing_settings_window_cannot_apply_during_profile(self):
        self.ns['height_profile'] = SimpleNamespace(enabled=True)
        previous = self.ns['REGION_SIFT_CONFIG']
        with self.assertRaises(ValueError):
            self.callbacks['apply_region_settings'](SimpleNamespace(group_balance_weight=.5))
        self.assertIs(self.ns['REGION_SIFT_CONFIG'], previous)

    def test_mae_toggle_colors_actual_polygons_without_remeasurement(self):
        self.start_preview()
        self.state['mode'] = 'idle'
        blocks, _ = summarize_block_records(self.state['plan'], [])
        for row, mae in zip(blocks, (0.49, 0.5, 1.0, 1.01)):
            row.update(valid_count=1, mean_height_mm=12, mean_height_error_mm=0, mae_mm=mae)
        self.state['summaries'] = blocks
        before = [dict(row) for row in blocks]
        self.callbacks['on_block_mae_toggle'](None)
        self.assertTrue(self.state['show_mae_colors'])
        self.assertEqual(len(self.axis.patches), 24)
        for polygon, color in zip(self.axis.patches, ('#B7E4C7', '#FFD6A5', '#FFD6A5', '#FFB3B3')):
            self.assertTrue(polygon.get_fill())
            np.testing.assert_allclose(polygon.get_facecolor(), to_rgba(color, 0.45))
        self.assertFalse(self.axis.patches[4].get_fill())
        self.assertEqual(len(self.axis.texts), 1)  # Only the color legend remains.
        self.assertTrue(self.axis.texts[0].get_text().startswith('MAE (mm):'))
        self.assertEqual(len(self.axis.collections), 0)
        self.callbacks['on_block_mae_toggle'](None)
        self.assertFalse(self.state['show_mae_colors'])
        self.assertEqual(len(self.axis.patches), 24)
        self.assertTrue(all(not polygon.get_fill() for polygon in self.axis.patches))
        self.assertEqual(len(self.axis.texts), 24)
        self.assertTrue(any('B01\nT=12.5mm\nH=12.00' in text.get_text()
                            for text in self.axis.texts))
        self.assertEqual(len(self.axis.collections), 1)
        self.assertEqual(len(self.axis.collections[0].get_offsets()), 216)
        self.assertEqual(before, self.state['summaries'])
        self.ns['do_measure'].assert_not_called()

    def test_mae_toggle_requires_finished_statistics_and_resets_for_new_run(self):
        self.callbacks['on_block_mae_toggle'](None)
        self.assertFalse(self.state['show_mae_colors'])
        self.start_preview()
        self.callbacks['on_block_accuracy'](None)
        self.callbacks['run_next_block_accuracy_point']()
        self.callbacks['finish_block_accuracy']('cancelled')
        self.assertIsNotNone(self.state['summaries'])
        self.callbacks['on_block_mae_toggle'](None)
        self.assertTrue(self.state['show_mae_colors'])
        self.callbacks['on_block_accuracy'](None)
        self.assertFalse(self.state['show_mae_colors'])
        self.assertIsNone(self.state['summaries'])
        # Keep MAE clickable during a run so it can explain why no coloring
        # is available instead of silently swallowing the mouse event.
        self.assertTrue(self.ns['btn_block_mae'].active)
        self.callbacks['finish_block_accuracy']()
        self.assertTrue(self.ns['btn_block_mae'].active)

    def test_real_button_press_release_after_run_switches_label_and_fill(self):
        canvas = FigureCanvasAgg(self.figure)
        button_ax = self.figure.add_axes([0.88, 0.77, 0.09, 0.026])
        button = Button(button_ax, 'MAE color: Off', useblit=False)
        self.ns['btn_block_mae'] = button
        button.on_clicked(self.callbacks['on_block_mae_toggle'])
        self.start_preview()
        self.callbacks['on_block_accuracy'](None)
        self.callbacks['run_next_block_accuracy_point']()
        self.callbacks['finish_block_accuracy']('cancelled')
        self.assertTrue(button.active)
        with patch('warnings.warn'):
            canvas.draw()
        x, y = button_ax.transAxes.transform((0.5, 0.5))
        for event_name in ('button_press_event', 'button_release_event'):
            canvas.callbacks.process(event_name, MouseEvent(event_name, canvas, x, y, button=1))
        self.assertTrue(self.state['show_mae_colors'])
        self.assertEqual(button.label.get_text(), 'MAE 著色: On')
        self.assertTrue(self.axis.patches[0].get_fill())

    def test_missing_summary_recovers_from_points_after_export_failure(self):
        self.start_preview()
        self.callbacks['on_block_accuracy'](None)
        self.callbacks['run_next_block_accuracy_point']()
        with patch.object(self.state['report'], 'finish', side_effect=OSError('file locked')):
            self.callbacks['finish_block_accuracy']('completed')
        self.assertEqual(self.state['mode'], 'idle')
        self.assertIsNotNone(self.state['summaries'])
        self.state['summaries'] = None
        self.callbacks['on_block_mae_toggle'](None)
        self.assertTrue(self.state['show_mae_colors'])
        self.assertEqual(self.state['summaries'][0]['valid_count'], 1)
        self.state['report'].finish('cancelled')

    def test_unavailable_button_click_has_visible_feedback(self):
        self.callbacks['on_block_mae_toggle'](None)
        self.assertIn('no statistics', self.ns['depth_text'].set_text.call_args.args[0])
        self.callbacks['on_block_accuracy'](None)
        self.callbacks['on_block_mae_toggle'](None)
        self.assertIn('finish/cancel', self.ns['depth_text'].set_text.call_args.args[0])
        self.callbacks['finish_block_accuracy']()

    def test_real_specular_control_disables_and_restores_without_triggering_toggle(self):
        FigureCanvasAgg(self.figure)
        specular = Button(self.figure.add_axes([0.42, 0.895, 0.13, 0.028]),
                          'V2: Off', useblit=False)
        on_change = Mock()
        specular.on_clicked(on_change)
        textbox = TextBox(self.figure.add_axes([0.78, 0.824, 0.08, 0.026]), '', initial='0')
        self.ns.update(btn_specular_v2=specular, text_box=textbox)
        self.callbacks['on_block_accuracy'](None)
        self.assertFalse(specular.active)
        self.assertFalse(textbox.active)
        self.assertEqual(specular.label.get_text(), 'V2: Off')
        self.callbacks['finish_block_accuracy']()
        self.assertTrue(specular.active)
        self.assertTrue(textbox.active)
        self.assertEqual(specular.label.get_text(), 'V2: Off')
        on_change.assert_not_called()

    def test_real_radio_full_measure_finish_and_mae_stays_usable(self):
        canvas = FigureCanvasAgg(self.figure)
        radio = RadioButtons(self.figure.add_axes([0.42, 0.836, 0.13, 0.12]),
                             ('Direct', 'Dedrift', 'Flow'), active=0)
        on_change = Mock()
        radio.on_clicked(on_change)
        button_ax = self.figure.add_axes([0.88, 0.77, 0.09, 0.026])
        button = Button(button_ax, 'MAE: Off', useblit=False)
        self.ns.update(radio_mode=radio, btn_block_mae=button)
        button.on_clicked(self.callbacks['on_block_mae_toggle'])
        self.start_preview()
        self.callbacks['on_block_accuracy'](None)
        for _ in range(216):
            self.callbacks['run_next_block_accuracy_point']()
        self.assertEqual(self.state['mode'], 'idle')
        self.assertEqual(radio.value_selected, 'Direct')
        self.assertTrue(radio.active)
        self.assertTrue(button.active)
        on_change.assert_not_called()
        with patch('warnings.warn'):
            canvas.draw()
        x, y = button_ax.transAxes.transform((0.5, 0.5))
        for name in ('button_press_event', 'button_release_event'):
            canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=1))
        self.assertTrue(self.state['show_mae_colors'])
        self.assertEqual(button.label.get_text(), 'MAE 著色: On')

    def test_measurement_exception_is_recorded_and_next_point_can_run(self):
        self.start_preview()
        self.callbacks['on_block_accuracy'](None)
        self.ns['do_measure'].side_effect = [ValueError('bad match'), {'height_display_mm': 12.0}]
        self.callbacks['run_next_block_accuracy_point']()
        self.callbacks['run_next_block_accuracy_point']()
        records = self.state['report'].records
        self.assertEqual([record['status'] for record in records], ['failed', 'valid'])
        self.assertIn('bad match', records[0]['failure_reason'])
        self.callbacks['on_block_accuracy_close'](None)
        self.assertTrue(self.state['report'].closed)

    def test_all_216_callbacks_complete_and_export_24_block_rows(self):
        self.start_preview()
        self.callbacks['on_block_accuracy'](None)
        for _ in range(216):
            self.callbacks['run_next_block_accuracy_point']()
        self.assertEqual(self.state['mode'], 'idle')
        self.assertEqual(self.ns['do_measure'].call_count, 216)
        self.assertEqual(len(self.state['report'].records), 216)
        self.assertTrue(self.state['report'].closed)
        self.timer.stop.assert_called_once()

    def test_csv_write_failure_stops_run_and_restores_controls(self):
        self.start_preview()
        self.callbacks['on_block_accuracy'](None)
        with patch.object(self.state['report'], 'append', side_effect=OSError('disk full')):
            self.callbacks['run_next_block_accuracy_point']()
        self.assertEqual(self.state['mode'], 'idle')
        self.assertTrue(self.ns['c18'].active)
        self.timer.stop.assert_called_once()

    def test_click_height_source_uses_best_pair_or_custom_fused_point(self):
        # Execute the unchanged height-readout section of do_measure, which
        # both ordinary clicks and the batch now return to their callers.
        tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
        measure = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == 'do_measure')
        assignments = [(i, ast.unparse(node.targets[0])) for i, node in enumerate(measure.body)
                       if isinstance(node, ast.Assign)]
        start = next(i for i, target in assignments if target == "res['shared_pattern_plane_diag']")
        end = next(i for i, target in assignments if target.startswith('measure_results['))
        code = compile(ast.fix_missing_locations(ast.Module(body=measure.body[start:end], type_ignores=[])),
                       str(SOURCE), 'exec')
        for custom, expected in ((False, 10.0), (True, 15.0)):
            result = {'p3d': np.array([0, 0, 115.]), 'p3d_best': np.array([0, 0, 112.])}
            context = dict(np=np, res=result, shared_height_plane_diag={}, custom_plane_fitted=custom,
                           custom_plane_n=np.array([0, 0, 1]), custom_plane_c=np.array([0, 0, 100]),
                           DEFAULT_WOUND_HEIGHT_OFFSET_MM=2.0,
                           get_selected_height_plane=lambda: (np.array([0, 0, 1]), np.array([0, 0, 100]),
                                                              'Shared Pattern Plane'))
            exec(code, context)
            self.assertEqual(result['height_display_mm'], expected)
            record = measurement_record(build_block_plan([[100, 100], [700, 100], [700, 500], [100, 500]],
                                                        (600, 800))['samples'][0], result, 1)
            self.assertEqual(record['error_mm'], expected - 12.5)

    def test_default_toggles_are_consistent_with_labels(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
        dictionaries = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('view_state', 'height_plane_state'):
                        dictionaries[target.id] = {ast.literal_eval(key): ast.literal_eval(value)
                                                  for key, value in zip(node.value.keys, node.value.values)
                                                  if isinstance(value, ast.Constant)}
        for option in ('precise', 'enforce_epi', 'grad_sift'):
            self.assertFalse(dictionaries['view_state'][option])
        self.assertTrue(dictionaries['view_state']['region_sift'])
        self.assertTrue(dictionaries['height_plane_state']['use_shared_plane'])
        labels = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == 'Button'
                    and len(node.value.args) > 1 and isinstance(node.value.args[1], ast.Constant)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        labels[target.id] = node.value.args[1].value
        self.assertEqual(labels['btn_shared_plane'], 'Shared: On')
        self.assertEqual(labels['c18'], '[X] Region-SIFT 匹配')
        for name in ('c1', 'c2', 'c3'):
            self.assertTrue(labels[name].startswith('[ ]'))


if __name__ == '__main__':
    unittest.main()
