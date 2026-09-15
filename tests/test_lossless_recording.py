"""Exercise the actual local FFmpeg encoder/decoder, with no camera or GUI."""
from pathlib import Path
import ast
import os
import time
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from Algorithm.lossless_recording import open_lossless_writer


class LosslessRecordingTests(unittest.TestCase):
    def test_recording_loop_writes_clean_frames_and_finalizes_on_stop_or_disconnect(self):
        source = Path(__file__).resolve().parents[1] / 'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'record_video_from_camera')
        for disconnect in (False, True):
            with self.subTest(disconnect=disconnect), tempfile.TemporaryDirectory() as folder:
                cap, writer = Mock(), Mock()
                frame = np.full((240, 320, 3), 123, np.uint8)
                cap.read.side_effect = [(True, frame.copy()) for _ in range(4)] if not disconnect else [
                    (True, frame.copy()), (True, frame.copy()), (False, None)]
                cap.get.side_effect = lambda prop: {
                    cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 240,
                    cv2.CAP_PROP_FPS: 25.}.get(prop, 0)
                written = []
                writer.write.side_effect = lambda image: written.append(image.copy())
                fake_cv = Mock(wraps=cv2)
                # Constants need their actual numeric values, not child mocks.
                for name in ('CAP_PROP_FOURCC', 'CAP_PROP_FRAME_WIDTH', 'CAP_PROP_FRAME_HEIGHT',
                             'CAP_PROP_FPS', 'FONT_HERSHEY_SIMPLEX'):
                    setattr(fake_cv, name, getattr(cv2, name))
                fake_cv.error = cv2.error
                fake_cv.VideoCapture = Mock(return_value=cap)
                fake_cv.waitKey = Mock(side_effect=[ord('s'), 0, ord('s'), ord('q')])
                fake_cv.imshow = Mock()
                fake_cv.destroyAllWindows = Mock()
                factory = Mock(return_value=writer)
                ns = dict(cv2=fake_cv, np=np, os=os, time=time, ACTUAL_MARKER_SIZE_MM=10,
                          RECORD_SAVE_DIR=folder, CAMERA_WIDTH=320, CAMERA_HEIGHT=240,
                          camera_algo=Mock(), log_and_print=Mock(),
                          open_lossless_writer=factory, estimate_aruco_pattern_distances=Mock(return_value={}),
                          draw_rt_sift_roi_preview=lambda image: image.fill(0),
                          draw_high_contrast_preview_text=Mock())
                exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
                with patch('builtins.print'):
                    path = ns['record_video_from_camera']()
                self.assertTrue(path.endswith('_lossless.avi'))
                self.assertEqual(factory.call_args.args[1:], (25., (320, 240)))
                self.assertEqual(len(written), 1 if disconnect else 2)
                for image in written:
                    np.testing.assert_array_equal(image, frame)
                writer.release.assert_called_once()
                cap.release.assert_called_once()

    def test_1080p_bgr_roundtrip_is_pixel_exact_and_seekable(self):
        rng = np.random.default_rng(32)
        # High-frequency color tests expose chroma subsampling or lossy conversion.
        frames = [rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8),
                  np.full((1080, 1920, 3), (0, 127, 255), np.uint8),
                  np.zeros((1080, 1920, 3), np.uint8)]
        frames[2][::2, ::2] = (255, 0, 255)
        with tempfile.TemporaryDirectory(prefix='ffv1_test_') as folder:
            path = Path(folder) / 'test_lossless.avi'
            writer = open_lossless_writer(path, 25., (1920, 1080))
            try:
                for frame in frames:
                    writer.write(frame)
            finally:
                writer.release()
            cap = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(cap.isOpened())
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frames))
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 25.)
                for frame in frames:
                    ok, decoded = cap.read()
                    self.assertTrue(ok)
                    np.testing.assert_array_equal(decoded, frame)
                self.assertFalse(cap.read()[0])
                self.assertTrue(cap.set(cv2.CAP_PROP_POS_FRAMES, 1))
                ok, decoded = cap.read()
                self.assertTrue(ok)
                np.testing.assert_array_equal(decoded, frames[1])
            finally:
                cap.release()

    def test_encoder_unavailable_reports_failure_without_lossy_fallback(self):
        writer = Mock()
        writer.isOpened.return_value = False
        with patch('Algorithm.lossless_recording.cv2.VideoWriter', return_value=writer) as factory:
            with self.assertRaisesRegex(RuntimeError, 'FFV1'):
                open_lossless_writer('not_created.avi', 25., (1920, 1080))
            factory.assert_called_once()
        writer.release.assert_called_once()

    def test_invalid_dimensions_and_fps_are_rejected(self):
        for size, fps in (((1919, 1080), 25.), ((1920, 1079), 25.),
                          ((1920, 1080), float('nan')), ((1920, 1080), 0.)):
            with self.subTest(size=size, fps=fps), self.assertRaises(ValueError):
                open_lossless_writer('not_created.avi', fps, size)


if __name__ == '__main__':
    unittest.main()
