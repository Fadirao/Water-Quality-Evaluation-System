# Only checks file produced by proxy NN curve.
import os

def test_loss_curve_exists():
    assert os.path.exists("visualization/graphs/loss_curve.png")
