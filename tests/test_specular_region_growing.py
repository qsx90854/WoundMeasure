import unittest
from dataclasses import replace
import cv2
import numpy as np
from Algorithm.specular_detection import (
    BlockSpatialSpecularConfig, _grow_specular_region,
    compute_specular_mask_bgr_block_adaptive, validate_block_spatial_specular_config)
from Algorithm.specular_settings import parse_config_values


class RegionGrowingTests(unittest.TestCase):
    def test_thin_line_radius_and_disconnected_candidate(self):
        image=np.full((13,15,3),180,np.uint8)
        seeds=np.zeros(image.shape[:2],bool);seeds[6,3]=True
        weak=np.zeros_like(seeds);weak[6,4:12]=True;weak[2,3]=True
        result=_grow_specular_region(image,seeds,weak,4,20)
        expected=seeds.copy();expected[6,4:8]=True
        np.testing.assert_array_equal(result,expected)
        self.assertEqual(seeds.sum(),1)

    def test_diagonal_and_no_color_drift(self):
        image=np.full((8,8,3),200,np.uint8)
        seeds=np.zeros((8,8),bool);seeds[1,1]=True
        weak=np.zeros_like(seeds)
        for i in range(2,7):
            weak[i,i]=True;image[i,i]=[200-(i-1)*15,200,200]
        result=_grow_specular_region(image,seeds,weak,6,40)
        self.assertTrue(result[2,2]);self.assertTrue(result[3,3])
        self.assertFalse(result[4,4]);self.assertFalse(result[5,5])

    def test_zero_radius_empty_seeds_and_boundaries(self):
        image=np.full((5,5,3),150,np.uint8);seeds=np.zeros((5,5),bool);weak=np.ones_like(seeds)
        self.assertFalse(_grow_specular_region(image,seeds,weak,3,60).any())
        seeds[0,0]=True
        np.testing.assert_array_equal(_grow_specular_region(image,seeds,weak,0,60),seeds)
        result=_grow_specular_region(image,seeds,weak,1,60)
        self.assertEqual(result.sum(),4)

    def test_integration_debug_and_default_off(self):
        image=np.full((41,41,3),40,np.uint8);image[20,20]=230;image[20,21:25]=160
        base=BlockSpatialSpecularConfig(v_min=200,v_max=200,rgb_min=200,rgb_max=200,
            gray_min=200,gray_below_v=0,open_kernel_px=1,close_kernel_px=1,dilate_px=0)
        old,d0=compute_specular_mask_bgr_block_adaptive(image,base,return_debug=True)
        self.assertFalse(base.enable_region_growing)
        grown,d=compute_specular_mask_bgr_block_adaptive(image,replace(base,enable_region_growing=True),return_debug=True)
        self.assertGreater(grown.sum(),old.sum())
        np.testing.assert_array_equal(d['region_growing_seed_mask'],d0['selected_before_morphology'])
        np.testing.assert_array_equal(grown,cv2.bitwise_or(old,d['region_growing_added_mask']))
        self.assertFalse(d0['region_growing_added_mask'].any())

    def test_settings_validation(self):
        c=parse_config_values(BlockSpatialSpecularConfig(),{'enable_region_growing':'True','grow_radius_px':'5'})
        self.assertTrue(c.enable_region_growing);self.assertEqual(c.grow_radius_px,5)
        for kw in ({'grow_radius_px':-1},{'grow_radius_px':33},{'grow_radius_px':1.5},
                   {'grow_color_max':float('nan')},{'grow_s_max':256},{'enable_region_growing':1}):
            with self.assertRaises(ValueError):validate_block_spatial_specular_config(replace(c,**kw))

if __name__=='__main__':unittest.main()
