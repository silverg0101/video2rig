import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'blender'))
from autorig_weights import segment_weights


class ProvisionalWeightTests(unittest.TestCase):
    def test_empty_heat_fallback_is_normalized_and_keeps_separated_limbs_local(self):
        points = [[-1, 0, .5], [1, 0, .5], [-1, 0, 0]]
        weights = segment_weights(points, [[-1, 0, 0], [1, 0, 0]], [[-1, 0, 1], [1, 0, 1]])
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_allclose(weights.sum(1), 1)
        self.assertGreater(weights[0, 0], .999)
        self.assertGreater(weights[1, 1], .999)

    def test_no_bones_is_an_explicit_error(self):
        with self.assertRaises(ValueError):
            segment_weights([[0, 0, 0]], [], [])


if __name__ == '__main__': unittest.main()
