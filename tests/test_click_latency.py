import ast
from pathlib import Path
import unittest
from Algorithm.click_latency import ClickLatency


class ClickLatencyTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.clock = ClickLatency(print_fn=self.lines.append)

    def test_stage_partition_includes_post_measure_wait_and_draw(self):
        ticket = self.clock.begin((100,200), 1.1, 1.)
        self.clock.ready(ticket, 1.8)
        result = self.clock.complete(ticket, 2., 2.1, 3., main_drawn=True)
        self.assertAlmostEqual(result['total_ms'], 2000.)
        expected = [100,700,200,100,900]
        for (_,actual), wanted in zip(result['stages'],expected):
            self.assertAlmostEqual(actual,wanted)
        self.assertAlmostEqual(sum(ms for _,ms in result['stages']),result['total_ms'])
        self.assertIsNone(self.clock.pending)
        self.assertIn('callback',self.lines[0])

    def test_partial_and_early_render_wait_for_main(self):
        ticket = self.clock.begin((1,2), 1.)
        self.assertIsNone(self.clock.complete(ticket,1.,2.,3.,main_drawn=True))
        self.clock.ready(ticket,2.)
        self.assertIsNone(self.clock.complete(ticket,1.,2.,3.,main_drawn=True))
        self.assertIsNone(self.clock.complete(ticket,2.,3.,4.,main_drawn=False))
        self.assertIsNotNone(self.clock.complete(ticket,2.,3.,4.,main_drawn=True))
        count = len(self.lines)
        self.assertIsNone(self.clock.complete(ticket,2.,3.,4.,main_drawn=True))
        self.assertEqual(len(self.lines),count)

    def test_superseded_ticket_cannot_finish_new_click(self):
        old = self.clock.begin((1,2),1.)
        new = self.clock.begin((3,4),2.)
        self.clock.ready(old,3.)
        self.clock.ready(new,3.)
        self.assertIsNone(self.clock.complete(old,3.,4.,5.,main_drawn=True))
        self.assertIs(self.clock.pending,new)
        self.assertIn('superseded',self.lines[0])

    def test_disabled_and_non_mouse_origin(self):
        disabled = ClickLatency(enabled=False,print_fn=self.lines.append)
        self.assertIsNone(disabled.begin((0,0),1.))
        disabled.ready(None,2.)
        self.assertIsNone(disabled.complete(None,2.,3.,4.,main_drawn=True))
        self.assertEqual(self.lines,[])
        ticket = self.clock.begin((0,0),1.)
        self.clock.ready(ticket,2.)
        self.clock.complete(ticket,2.,3.,4.,main_drawn=True)
        self.assertIn('量測入口',self.lines[0])

    def test_main_wiring_and_render_completion_order(self):
        source = Path(__file__).parents[1]/'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        functions = {n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
        release = functions['on_release']
        calls = [n for n in ast.walk(release) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Name) and n.func.id=='do_measure']
        self.assertEqual(len(calls),2)
        self.assertTrue(all(any(k.arg=='release_started_at' for k in n.keywords) for n in calls))
        render = ast.unparse(functions['render_pending'])
        self.assertLess(render.index('view_renderer.draw(pair)'),render.index('click_latency.complete'))
        measure = ast.unparse(functions['do_measure'])
        self.assertLess(measure.index('apply_measure_result(res'),measure.index('click_latency.ready'))
        self.assertLess(measure.index('click_latency.ready'),measure.index('click_timer.report'))


if __name__ == '__main__':
    unittest.main()
