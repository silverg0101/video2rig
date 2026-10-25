"""Stage adapters for Asset Forge; generators stay outside video2rig.

Observations use named bone heads/tails, normalized front/side coordinates and
confidence. This allows either reader to feed either solver on the same video.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .common import Project, write_json

ADAPTER_VERSION = 1


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def attach_clip(project, video):
    """Each project is job-local: don't rewrite the shared trained reader project."""
    pr = Project(project)
    pr.cfg['clips'] = {'take': {'file': str(Path(video).resolve())}}
    write_json(pr.root / 'creature.json', pr.cfg)
    return pr


def forge_keypoints(joints, detected):
    """One named point per detected Forge index, with a real child head preferred over a tail."""
    by_point = {}
    for end, frac in (('head', 0.0), ('tail', 1.0)):
        for bone in joints['bones']:
            i = bone[end]
            if detected[i] and i not in by_point:
                by_point[i] = (bone['name'] + ('.tip' if frac else ''), bone['name'], frac)
    ids = sorted(by_point)
    if len(ids) < 3:
        raise ValueError('fewer than three named detected joints; cannot fit a skeleton')
    return ids, [by_point[i] for i in ids]


def import_observations(joints, observations, panel_sizes):
    """Forge normalized arrays -> video2rig's pixel-space, named keypoints."""
    ids, kps = forge_keypoints(joints, observations['detected'])
    uv = np.stack([observations['front'][:, ids], observations['side'][:, ids]], axis=1)
    uv = uv * np.asarray(panel_sizes)[None, :, None, :]
    conf = np.stack([observations['vis_f'][:, ids], observations['vis_s'][:, ids]], axis=1)
    return uv, conf, kps


def export_observations(joints, uv, conf, kps, panel_sizes):
    """Map exact bone heads/tails only; extra surface landmarks remain in reader.npz."""
    n, T = len(joints['points']), len(uv)
    tracks = np.full((2, T, n, 2), .5, dtype=np.float32)
    confidence = np.zeros((2, T, n), dtype=np.float32)
    found = np.zeros(n, bool)
    bones = {b['name']: b for b in joints['bones']}
    for k, (_, name, frac) in enumerate(kps):
        if name not in bones or not isinstance(frac, (int, float)) or frac not in (0, 1):
            continue
        i = bones[name]['head' if frac == 0 else 'tail']
        if found[i]:
            continue
        tracks[:, :, i] = uv[:, :, k].transpose(1, 0, 2) / np.asarray(panel_sizes)[:, None, :]
        confidence[:, :, i] = conf[:, :, k].T
        found[i] = True
    if found.sum() < 3:
        raise ValueError('video2rig reader has fewer than three Forge-compatible joints')
    return dict(front=tracks[0], side=tracks[1], vis_f=confidence[0], vis_s=confidence[1], detected=found)


def panel_sizes(pr, video):
    import cv2
    from .clip import Panel
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError('cannot decode input video')
    cams = json.loads((pr.start_dir / 'cameras.json').read_text())
    h, w = frame.shape[:2]
    panels = [Panel(v, (w, h), cams['frame_px']) for v in cams['views']]
    return [(p.cw, p.ch) for p in panels]


def read(project, video, joints, out, tta=8):
    from .reader.road import read_clip
    pr = attach_clip(project, video)
    uv, conf, kps, names = read_clip(pr, 'take', tta=tta)
    if names != ['front', 'side']:
        raise ValueError('Forge bridge requires front | side views')
    np.savez_compressed(out, **export_observations(joints, uv, conf, kps, panel_sizes(pr, video)))


def fit(project, video, joints, observations, out, native=False):
    from .reader.road import fit_reading
    pr = attach_clip(project, video)
    if native:
        data = np.load(pr.work('take') / 'reader.npz')
        uv, conf = data['uv'], data['conf']
        kps = json.loads(str(data['keypoints']))
    else:
        data = np.load(observations)
        uv, conf, kps = import_observations(joints, data, panel_sizes(pr, video))
    cfg = pr.clip_cfg('take')
    cfg.setdefault('solve', {})
    result = fit_reading(pr, 'take', cfg, uv, conf, kps, ['front', 'side'])
    result['creature'].save('take', result['params'], '', {'reader': True, 'forge_adapter': ADAPTER_VERSION})
    result['creature'].export_motion('take', result['params'], Path(out))
    write_json(Path(out).with_suffix('.report.json'), result['report'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='stage', required=True)
    for stage in ('read', 'fit'):
        p = sub.add_parser(stage)
        p.add_argument('project')
        p.add_argument('--video', required=True)
        p.add_argument('--joints', required=True)
        p.add_argument('--out', required=True)
        if stage == 'read':
            p.add_argument('--tta', type=int, choices=range(0, 17), default=8)
        else:
            p.add_argument('--observations', required=True)
            p.add_argument('--native', action='store_true')
    a = ap.parse_args()
    joints = json.loads(Path(a.joints).read_text())
    if a.stage == 'read':
        read(a.project, a.video, joints, a.out, a.tta)
    else:
        fit(a.project, a.video, joints, a.observations, a.out, a.native)


if __name__ == '__main__':
    main()
