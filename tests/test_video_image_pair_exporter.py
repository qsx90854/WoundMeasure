import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from video_image_pair_exporter import read_frame, export_pair
from Algorithm.image_pair_input import load_image_pair


class VideoImagePairExporterTests(unittest.TestCase):
    def test_seek_export_and_measurement_input_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            video = folder/'test.avi'
            frames = [np.random.default_rng(i).integers(0, 256, (48, 64, 3), dtype=np.uint8) for i in range(8)]
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for frame in frames:
                writer.write(frame)
            writer.release()
            cap = cv2.VideoCapture(str(video))
            try:
                # Backward seek and non-consecutive frames must preserve exact pixels.
                left = read_frame(cap, 6, len(frames))
                right = read_frame(cap, 2, len(frames))
                np.testing.assert_array_equal(left, frames[6])
                np.testing.assert_array_equal(right, frames[2])
                with self.assertRaises(ValueError):
                    read_frame(cap, 8, len(frames))
            finally:
                cap.release()
            out = export_pair(video, (6, left), (2, right), folder/'中文輸出', 10.)
            a, b = load_image_pair(out/'left_F000006.png', out/'right_F000002.png')
            np.testing.assert_array_equal(a, frames[6])
            np.testing.assert_array_equal(b, frames[2])
            metadata = json.loads((out/'source.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['images']['left']['frame_index'], 6)
            self.assertEqual(metadata['images']['right']['nominal_time_s'], .2)
            second = export_pair(video, (6, left), (2, right), folder, 10.)
            self.assertNotEqual(out, second)
            with self.assertRaises(ValueError):
                export_pair(video, (6, left), (6, left), folder, 10.)


if __name__ == '__main__':
    unittest.main()
