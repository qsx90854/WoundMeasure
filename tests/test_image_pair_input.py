import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from Algorithm.image_pair_input import load_image_pair, create_pair_session


class ImagePairInputTests(unittest.TestCase):
    def test_pixels_unicode_dimensions_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root/'左圖.png', root/'右圖.png'
            a = np.random.default_rng(2).integers(0, 256, (32, 48, 3), dtype=np.uint8)
            b = np.flip(a, axis=1).copy()
            cv2.imencode('.png', a)[1].tofile(left)
            cv2.imencode('.png', b)[1].tofile(right)
            actual = load_image_pair(left, right)
            np.testing.assert_array_equal(actual[0], a)
            np.testing.assert_array_equal(actual[1], b)
            with self.assertRaises(ValueError):
                load_image_pair(left, left)
            m = create_pair_session(left, right, a.shape, root, root/'calibration.json')
            metadata = json.loads(Path(m).read_text(encoding='utf-8'))
            self.assertEqual(metadata['left_image'], str(left.resolve()))
            self.assertEqual(metadata['frame_mapping']['0'], 'right/A')
            self.assertNotEqual(m, create_pair_session(left, right, a.shape, root, root/'calibration.json'))
            cv2.imencode('.png', b[:20])[1].tofile(right)
            with self.assertRaises(ValueError):
                load_image_pair(left, right)

    def test_main_wrapper_fixed_pair_and_video_options(self):
        source = Path(__file__).parents[1]/'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'analyze_video_frames')
        # Execute the actual wrapper without launching main's Tk/ONNX UI.
        ns = {n.id: 1 for n in ast.walk(node) if isinstance(n, ast.Name) and n.id.isupper()}
        backend = Mock(return_value={'ok': True})
        ns.update(video_pose_algo=SimpleNamespace(analyze_video_frames=backend),
                  log_and_print=Mock(), configure_world_marker_sift_rt_refine=Mock(),
                  preprocess_gray=Mock(), average_rotations_svd=Mock(),
                  FRAME_PAIR_SELECTION_MODE='angle_guided')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
        args = ('pair.json', 20, 20, np.eye(3), np.zeros(5), np.eye(3), 20.)
        images = [np.zeros((20, 20, 3), np.uint8)]*2
        ns['analyze_video_frames'](*args, frames_override=images)
        call = backend.call_args
        self.assertEqual(call.args[1:3], (1, 1))
        self.assertEqual(call.args[7:9], ('best', 'fixed'))
        self.assertIs(call.kwargs['frames_override'], images)
        for name in ('pattern_guided', 'local_window', 'frames_are_undistorted'):
            self.assertFalse(call.kwargs[name])
        self.assertFalse(call.kwargs['angle_guided_config']['enabled'])
        ns['analyze_video_frames'](*args)
        self.assertTrue(backend.call_args.kwargs['angle_guided_config']['enabled'])
        self.assertTrue(backend.call_args.kwargs['local_window'])


if __name__ == '__main__':
    unittest.main()
