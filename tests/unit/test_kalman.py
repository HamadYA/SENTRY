import numpy as np

from sentry_tracking.candidates import KalmanCandidate


def test_kalman_candidate_preserves_mask_shape():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[10:20, 15:25] = 1
    kalman = KalmanCandidate()
    kalman.initialize([15, 10, 10, 10], 50, 40, mask=mask)
    proposal = kalman.predict(50, 40)
    assert proposal is not None
    assert proposal.mask.sum() == mask.sum()
    assert proposal.kind == "kalman"
