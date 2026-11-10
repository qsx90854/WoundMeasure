from dataclasses import replace
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from Algorithm.Region_SIFT_Matching import DEFAULT_CONFIG, run_region_sift_matching
from Algorithm.region_sift_center_refinement import (
    validate_center_config, sample_center, weighted_curve, assess_curve,
    refine_center, attach_center_heights, summary_for_log, CenterRefinementViewer)
from tests.test_region_sift_matching import _identity_candidate


class CenterRefinementTests(unittest.TestCase):
    def setUp(self):
        self.cfg=replace(DEFAULT_CONFIG, center_refine_enabled=True,
            auto_scale_orientation=False, keypoint_size_px=3.2,
            search_length_px=19,search_width_px=1,max_objective_score_ratio=None)
        self.image=cv2.GaussianBlur(np.random.default_rng(125).integers(0,256,(180,220),dtype=np.uint8),(3,3),.6)
        self.p=np.array([110.,90.],np.float32)

    def base(self, **changes):
        cfg=replace(self.cfg,center_refine_enabled=False,**changes)
        return run_region_sift_matching(self.image,self.image,self.p,_identity_candidate(),np.eye(3),config=cfg)

    def test_default_disabled_and_no_work(self):
        self.assertFalse(DEFAULT_CONFIG.center_refine_enabled)
        result={'m_pt':self.p.copy()}
        with patch('Algorithm.region_sift_center_refinement.sample_center') as call:
            refine_center(result,None,None,None,None,DEFAULT_CONFIG)
        call.assert_not_called()
        self.assertNotIn('center_refinement',result)

    def test_validation(self):
        for changes in (dict(local_plane_refine=True),dict(center_refine_step_px=0),
                        dict(center_refine_points=2),dict(center_refine_min_points=30),
                        dict(center_refine_min_group_fraction=0),dict(center_refine_radius_px=8),
                        dict(center_refine_max_support_radius_px=1)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_center_config(replace(self.cfg,**changes))

    def test_stratified_points_unique_spaced_and_p_retained(self):
        pts,roi=sample_center(self.image,self.p,self.cfg)
        self.assertEqual(pts.shape,(10,2)); self.assertEqual(len(np.unique(pts,axis=0)),10)
        np.testing.assert_array_equal(pts[-1],self.p)
        distances=np.linalg.norm(pts[:,None]-pts[None,:],axis=2); np.fill_diagonal(distances,np.inf)
        self.assertGreaterEqual(distances.min(),self.cfg.center_refine_min_point_distance_px)
        pts2,_=sample_center(self.image,self.p,replace(self.cfg,center_refine_points=16))
        self.assertEqual(len(pts2),17)

    def test_masked_points_and_boundary_fail_safely(self):
        mask=np.zeros_like(self.image);mask[89,108]=255
        pts,_=sample_center(self.image,self.p,replace(self.cfg,use_masked_sift=True),mask)
        self.assertFalse(np.any(mask[pts[:,1].astype(int),pts[:,0].astype(int)]))
        mask[90,110]=255
        with self.assertRaises(ValueError): sample_center(self.image,self.p,self.cfg,mask)
        with self.assertRaises(ValueError): sample_center(self.image,(1,1),self.cfg)

    def test_missing_points_not_rewarded_and_distribution_required(self):
        pts,_=sample_center(self.image,self.p,self.cfg)
        d=np.full((2,len(pts)),100.); ok=np.ones_like(d,bool);ok[1,0]=False
        scores,weights=weighted_curve(d,ok,pts,self.p,self.cfg)
        self.assertGreater(scores[1],scores[0]); self.assertAlmostEqual(weights.sum(),1.,places=6)
        ok[1]=pts[:,0]>self.p[0]
        scores,_=weighted_curve(d,ok,pts,self.p,self.cfg)
        self.assertTrue(np.isnan(scores[1]))

    def test_curve_decisions(self):
        x=np.arange(-2,3)
        self.assertEqual(assess_curve(x,[140,130,120,80,130],self.cfg)['action'],'apply')
        self.assertEqual(assess_curve(x,[140,130,120,100,80],self.cfg)['action'],'expand')
        self.assertEqual(assess_curve(x,[100]*5,self.cfg)['action'],'fallback')
        self.assertEqual(assess_curve(x,[140,np.nan,120,80,130],self.cfg)['action'],'fallback')
        self.assertEqual(assess_curve(x,[80,140,120,80,140],self.cfg)['action'],'expand')
        self.assertEqual(assess_curve(np.arange(-3,4),[140,80,130,120,80,130,140],self.cfg)['action'],'fallback')

    def test_rejected_never_rescued(self):
        r=dict(m_pt=None,reject_reason='ambiguous',region_debug={})
        with patch('Algorithm.region_sift_center_refinement.sample_center') as sample:
            refine_center(r,self.image,self.p,_identity_candidate(),np.eye(3),self.cfg)
        sample.assert_not_called();self.assertIsNone(r['m_pt']);self.assertEqual(r['reject_reason'],'ambiguous')

    def test_off_full_match_unchanged(self):
        off=self.base()
        with patch('Algorithm.region_sift_center_refinement.refine_center') as call:
            again=self.base()
        call.assert_not_called()
        np.testing.assert_array_equal(off['m_pt'],again['m_pt'])
        np.testing.assert_array_equal(off['region_debug']['candidate_objective_scores'],again['region_debug']['candidate_objective_scores'])

    def test_identity_keeps_original_for_both_backends(self):
        for masked in (False,True):
            with self.subTest(masked=masked):
                mask=np.zeros_like(self.image)
                cfg=replace(self.cfg,use_masked_sift=masked)
                r=run_region_sift_matching(self.image,self.image,self.p,_identity_candidate(),np.eye(3),
                    config=cfg,left_spec_mask=mask,right_spec_mask=mask)
                self.assertIsNotNone(r['m_pt'],r['reject_reason'])
                self.assertFalse(r['center_refinement']['applied'])
                self.assertGreater(r['center_refinement']['descriptor_rows'],0)
                np.testing.assert_allclose(r['m_pt'],self.p,atol=1e-5)

    def test_actual_local_search_expands_and_applies_without_changing_original_scores(self):
        r=self.base(); debug=r['region_debug']; tangent=debug['tangent']
        # Isolate center localization: move the initial rigid hypothesis by 2 px.
        r['m_pt']=r['m_pt']+2*tangent
        debug['best_center_warp']=debug['best_center_warp']+2*tangent
        debug['best_along_offset_px']+=2
        old_scores=debug['candidate_objective_scores'].copy()
        with patch('Algorithm.region_sift_center_refinement.triangulate_checked',return_value=np.array([0.,0.,200.])):
            refine_center(r,self.image,self.p,_identity_candidate(),np.eye(3),self.cfg)
        report=r['center_refinement']
        self.assertTrue(report['applied'],report['reason'])
        self.assertEqual(report['center_shift_px'],-2)
        self.assertGreater(len(report['history']),1)
        self.assertEqual(report['descriptor_rows'],90)  # 5 initial + 4 new positions, cache reused.
        np.testing.assert_allclose(r['m_pt'],self.p,atol=1e-5)
        np.testing.assert_array_equal(old_scores,debug['candidate_objective_scores'])
        json.dumps(summary_for_log(report))

    def test_bad_geometry_falls_back(self):
        r=self.base();d=r['region_debug'];r['m_pt']=r['m_pt']+d['tangent']
        d['best_center_warp']=d['best_center_warp']+d['tangent'];d['best_along_offset_px']+=1
        old=r['m_pt'].copy()
        with patch('Algorithm.region_sift_center_refinement.triangulate_checked',side_effect=ValueError('bad geometry')):
            refine_center(r,self.image,self.p,_identity_candidate(),np.eye(3),self.cfg)
        np.testing.assert_array_equal(old,r['m_pt']);self.assertFalse(r['center_refinement']['applied'])

    def test_two_stage_refines_once_after_final_acceptance(self):
        from Algorithm import region_sift_center_refinement as module
        cfg=replace(self.cfg,two_stage_search=True,fine_half_length_px=4,adaptive_max_expansions=0)
        with patch.object(module,'refine_center',wraps=module.refine_center) as call:
            r=run_region_sift_matching(self.image,self.image,self.p,_identity_candidate(),np.eye(3),config=cfg)
        self.assertEqual(call.call_count,1)
        self.assertIsNotNone(r['m_pt'],r['reject_reason'])
        self.assertAlmostEqual(r['elapsed_ms'],sum(r['timing_ms'].values()))

    def test_app_height_comparison_and_depth_limit_fallback(self):
        r=dict(m_pt=np.array([1.,0.]),method='original+CenterRefine',center_refinement=dict(status='applied',applied=True,
            original_center_right=np.array([0.,0.]),original_method='original'))
        tri=lambda q:np.array([0.,0.,200.+q[0]])
        attach_center_heights(r,tri,np.array([0.,0.,1.]),np.array([0.,0.,190.]))
        self.assertEqual(r['center_refinement']['original_height_mm'],10.)
        self.assertEqual(r['center_refinement']['final_height_mm'],11.)
        attach_center_heights(r,tri,np.array([0.,0.,1.]),np.zeros(3),max_depth=200.)
        self.assertFalse(r['center_refinement']['applied']);self.assertEqual(r['method'],'original')
        np.testing.assert_array_equal(r['m_pt'],[0.,0.])

    def test_main_ui_mutual_exclusion_and_lock(self):
        path=Path(__file__).parents[1]/'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('apply_region_settings','on_center_refine','on_region_refine')]
        button=lambda:SimpleNamespace(label=SimpleNamespace(set_text=Mock()))
        ns=dict(REGION_SIFT_CONFIG=replace(self.cfg,center_refine_enabled=False),height_profile=None,
            block_accuracy_state={'mode':'idle'},btn_sift_backend=button(),btn_region_refine=button(),
            btn_center_refine=button(),center_refinement_viewer=SimpleNamespace(close=Mock()),
            depth_text=SimpleNamespace(set_text=Mock()),request_blit_refresh=Mock())
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(path),'exec'),ns)
        ns['on_center_refine'](None);self.assertTrue(ns['REGION_SIFT_CONFIG'].center_refine_enabled)
        ns['on_region_refine'](None);self.assertFalse(ns['REGION_SIFT_CONFIG'].center_refine_enabled)
        self.assertTrue(ns['REGION_SIFT_CONFIG'].local_plane_refine)
        ns['block_accuracy_state']['mode']='running';ns['on_center_refine'](None)
        self.assertFalse(ns['REGION_SIFT_CONFIG'].center_refine_enabled)

    def test_view_disabled_does_not_draw(self):
        viewer=CenterRefinementViewer()
        with patch('matplotlib.pyplot.figure') as figure:
            viewer.show({},enabled=False)
        figure.assert_not_called()


if __name__=='__main__': unittest.main()
