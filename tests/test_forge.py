import unittest

import numpy as np

from video2rig.forge import export_observations, import_observations


class ObservationContract(unittest.TestCase):
    def setUp(self):
        self.j = {'points': [[0, 0, 0]] * 4, 'bones': [
            {'name': 'Hip', 'head': 0, 'tail': 1},
            {'name': 'Knee.L', 'head': 1, 'tail': 2},
            {'name': 'Foot.L', 'head': 2, 'tail': 3},
        ]}

    def test_normalized_pixel_round_trip_and_shared_head_identity(self):
        rng = np.random.default_rng(4)
        obs = dict(front=rng.random((5, 4, 2)), side=rng.random((5, 4, 2)),
                   vis_f=rng.random((5, 4)), vis_s=rng.random((5, 4)), detected=np.ones(4, bool))
        sizes = [(672, 768), (672, 768)]
        uv, conf, kps = import_observations(self.j, obs, sizes)
        self.assertEqual(kps[1], ('Knee.L', 'Knee.L', 0.0))
        back = export_observations(self.j, uv, conf, kps, sizes)
        for name, data in obs.items():
            np.testing.assert_allclose(back[name], data, atol=1e-7)

    def test_unknown_surface_landmarks_do_not_become_joint_targets(self):
        uv = np.full((3, 2, 4, 2), 15.)
        conf = np.ones((3, 2, 4))
        kps = [('Hip', 'Hip', 0), ('Knee.L', 'Knee.L', 0), ('Foot.L', 'Foot.L', 0),
               ('surface', 'Foot.L', [1, 2, 3])]
        obs = export_observations(self.j, uv, conf, kps, [(30, 30)] * 2)
        self.assertEqual(obs['detected'].tolist(), [True, True, True, False])
        self.assertTrue((obs['vis_f'][:, 3] == 0).all())

    def test_sparse_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'three'):
            import_observations(self.j, {'detected': [True, False, False, False]}, [(672, 768)] * 2)


if __name__ == '__main__':
    unittest.main()
