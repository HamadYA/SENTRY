from pathlib import Path

import pytest


cv2 = pytest.importorskip("cv2")


def test_example_video_is_readable_and_matches_documented_frame_size():
    video = Path(__file__).resolve().parents[2] / "assets" / "example_video.mp4"
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
        assert ok
        assert frame.shape[:2] == (240, 320)
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 60
        blue, green, red = frame[100, 70]
        assert int(red) > int(blue) + 80
        assert int(red) > int(green) + 80
    finally:
        capture.release()
