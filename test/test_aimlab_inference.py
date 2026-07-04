import unittest

import cv2
import numpy as np

from inference_aimlab import detect_ball_bgr


class AimlabColorDetectorTests(unittest.TestCase):
    def setUp(self):
        self.lo = np.array([0, 30, 30], dtype=np.uint8)
        self.hi = np.array([180, 255, 255], dtype=np.uint8)

    def _detect(self, frame):
        return detect_ball_bgr(
            frame,
            self.lo,
            self.hi,
            min_area=20,
            morph_ksize=3,
            color_mode="red_wrap",
            ignore_center_margin_px=16.0,
            reject_if_only_center_blobs=True,
        )

    @staticmethod
    def _center(row):
        return (0.5 * (row[0] + row[2]), 0.5 * (row[1] + row[3]))

    def test_centered_ball_is_not_rejected_as_crosshair(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(frame, (64, 64), 10, (0, 0, 255), thickness=-1)

        row = self._detect(frame)

        self.assertIsNotNone(row)
        cx, cy = self._center(row)
        self.assertAlmostEqual(cx, 64.0, delta=1.0)
        self.assertAlmostEqual(cy, 64.0, delta=1.0)

    def test_thin_center_crosshair_is_not_a_ball(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.line(frame, (48, 64), (80, 64), (0, 0, 255), thickness=1)
        cv2.line(frame, (64, 48), (64, 80), (0, 0, 255), thickness=1)

        self.assertIsNone(self._detect(frame))

    def test_same_colour_crosshair_does_not_pull_ball_center(self):
        for expected_x in (56, 60, 64, 68, 72):
            with self.subTest(expected_x=expected_x):
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
                cv2.circle(frame, (expected_x, 64), 10, (0, 0, 255), thickness=-1)
                cv2.line(frame, (48, 64), (80, 64), (0, 0, 255), thickness=1)
                cv2.line(frame, (64, 48), (64, 80), (0, 0, 255), thickness=1)

                row = self._detect(frame)

                self.assertIsNotNone(row)
                cx, cy = self._center(row)
                self.assertAlmostEqual(cx, float(expected_x), delta=1.5)
                self.assertAlmostEqual(cy, 64.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
