"""Execute the real lower-panel callbacks; disabled artists raise on updates."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, ConnectionPatch
import numpy as np
import cv2

from Algorithm.Region_SIFT_Matching import run_region_sift_matching, with_config, DEFAULT_CONFIG
from Algorithm.paired_view_renderer import PairedViewRenderer
from tests.test_region_sift_matching import _identity_candidate

SOURCE = Path(__file__).parents[1]/'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
FUNCTIONS = {n.name:n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef)}


class LowerDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image=np.random.default_rng(56).integers(0,256,(180,220),dtype=np.uint8)
        cfg=with_config(DEFAULT_CONFIG,auto_scale_orientation=False,keypoint_size_px=3.2,
                        search_length_px=9,search_width_px=1,max_objective_score_ratio=None)
        result=run_region_sift_matching(cls.image,cls.image,(110,90),_identity_candidate(),np.eye(3),config=cfg)
        assert result['region_debug'] is not None
        cls.result=dict(region_debug=result['region_debug'],debug_left_gray=cls.image,
                        debug_right_gray=cls.image,u=110,v=90,pt=result['m_pt'])

    def harness(self,left,right,region=True):
        fig, axes=plt.subplots(2,2)
        self.addCleanup(plt.close,fig)
        info=[fig.add_axes([.1,.01,.3,.08]),fig.add_axes([.6,.01,.3,.08])]
        ns=dict(np=np,cv2=cv2,Rectangle=Rectangle,Polygon=Polygon,ConnectionPatch=ConnectionPatch,
                DEBUG_LEFT_PANEL_ENABLED=left,DEBUG_RIGHT_PANEL_ENABLED=right,
                ax_A=axes[0,0],ax_B=axes[0,1],ax_debug_A=axes[1,0],ax_debug_B=axes[1,1],
                ax_debug_info_A=info[0],ax_debug_info_B=info[1],
                debug_region_artists=[],debug_pair_artists=[],debug_audit_artists=[],
                debug_homography_residual_artists=[],debug_metric_block_artists=[],
                debug_region_state={},debug_audit_state={},debug_zoom_home={},
                region_diagnostic_latest={},block_accuracy_state={'mode':'idle'},
                auto_calc_active=False,height_profile=None,view_state={'region_sift':region},
                show_region_diagnostics=Mock(),region_diagnostics=SimpleNamespace(show=Mock()),
                request_blit_refresh=Mock(),current_cand=_identity_candidate(),KL=np.eye(3),w=220,h=180,
                LEFT_PATCH_SEARCH_RADIUS=20,RIGHT_PATCH_SEARCH_RADIUS=20,
                predict_right_seed_from_geometry=Mock(return_value=(np.array([110,90]),'test')),
                epipolar_line=Mock(return_value=((110,0),(110,180))))
        callbacks=['debug_axis_enabled','debug_artist_enabled','clear_region_sift_debug_artists',
                   'clear_metric_block_debug_artists','select_region_sift_debug_point',
                   'select_grad_descriptor_audit','update_region_sift_debug_views','update_grad_match_debug_views',
                   'render_metric_block_audit']
        nodes=[FUNCTIONS[name] for name in callbacks]
        names={n.id for fn in nodes for n in ast.walk(fn) if isinstance(n,ast.Name)}
        disabled=[]
        for name in names:
            if name.startswith('dbg_'):
                is_left=name.endswith('_A') or name=='dbg_hull_line'
                ax=axes[1,0 if is_left else 1]
                if name.startswith('dbg_info'): artist=info[0 if is_left else 1].text(0,0,'')
                elif name=='dbg_search_rect':
                    artist=Rectangle((0,0),1,1); ax.add_patch(artist)
                elif name in ('dbg_hull_line','dbg_hull_line_B','dbg_epi_line'): artist,=ax.plot([],[])
                else: artist=ax.scatter([],[])
                ns[name]=artist
                if not (left if is_left else right): disabled.append(artist)
        for side,enabled,ax in [('A',left,axes[1,0]),('B',right,axes[1,1])]:
            ns['im_debug_'+side]=ax.imshow(self.image)
            if not enabled: disabled.extend([ax,info[0 if side=='A' else 1],ns['im_debug_'+side]])
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),ns)
        # Match production exclusion policy. Raise before entering ANY graphics
        # update method, rather than trusting visible=False to suppress drawing.
        for ax in (axes[1,0],axes[1,1],*info):
            if not ns['debug_axis_enabled'](ax): ax.set_visible(False); ax.set_animated(True)
        for artist in disabled:
            for method in ('set_data','set_offsets','set_text','set_title','set_xlim','set_ylim',
                           'set_bounds','set_visible','plot','scatter','text','add_patch','add_artist','draw'):
                if hasattr(artist,method):
                    guard=patch.object(artist,method,side_effect=AssertionError('disabled '+method))
                    guard.start(); self.addCleanup(guard.stop)
        return ns,fig,axes,info

    def test_region_all_combinations_no_hidden_updates(self):
        for left,right in ((False,False),(True,False),(False,True),(True,True)):
            with self.subTest(left=left,right=right):
                ns,*_=self.harness(left,right)
                with patch.object(cv2,'cvtColor',wraps=cv2.cvtColor) as color:
                    ns['update_region_sift_debug_views'](self.result,110,90)
                self.assertEqual(color.call_count,int(left)+int(right))
                self.assertIs(ns['region_diagnostic_latest']['result'],self.result)
                ns['show_region_diagnostics'].assert_called_once()

    def test_region_failure_all_combinations(self):
        for left,right in ((False,False),(True,False),(False,True)):
            ns,*_=self.harness(left,right)
            result=dict(self.result,region_debug=None,fail_reason='test')
            ns['update_region_sift_debug_views'](result,110,90)

    def test_grad_all_combinations(self):
        for left,right in ((False,False),(True,False),(False,True),(True,True)):
            with self.subTest(left=left,right=right):
                ns,*_=self.harness(left,right,False)
                points=np.array([[100,80],[120,80],[110,100]],np.float32)
                result=dict(self.result,g_refA=points,g_refB=points,g_ptsA=points,g_ptsB=points)
                ns['update_grad_match_debug_views'](result,110,90)

    def test_full_and_navigation_render_never_draw_disabled_axes(self):
        for left,right in ((False,False),(True,False),(False,True)):
            ns,fig,axes,info=self.harness(left,right)
            renderer=PairedViewRenderer(fig,(tuple(axes[0]),tuple(ax for ax in axes[1] if ns['debug_axis_enabled'](ax))))
            renderer.draw()
            renderer.draw(0)
            renderer.draw(1)
            renderer.draw(1)

    def test_disabled_mouse_events_return_before_other_handlers(self):
        for name in ('on_press','on_motion','on_scroll','reset_debug_zoom','request_view_refresh'):
            node=FUNCTIONS[name]
            # Remove nonlocal declarations for standalone extraction; execution
            # must return before any other names/state are used.
            import copy
            node=copy.deepcopy(node)
            class Strip(ast.NodeTransformer):
                def visit_Nonlocal(self,node): return None
            node=Strip().visit(node)
            ns={'debug_axis_enabled':lambda axis:False}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
            ns[name](SimpleNamespace(inaxes=object()))


if __name__=='__main__':
    unittest.main()
