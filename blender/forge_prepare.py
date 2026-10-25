"""Export an Asset Forge model through Forge's exact start-pose cameras.

blender -b -P forge_prepare.py -- MODEL JOINTS_JSON START_PNG PROJECT
The source model is never saved over. No image or video service is called.
"""
import json
import math
import runpy
import shutil
import sys
from pathlib import Path

import bpy


def main():
    model, joints, start, target = sys.argv[sys.argv.index('--') + 1:]
    model, target = Path(model).resolve(), Path(target).resolve()
    j = json.loads(Path(joints).read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if model.suffix.lower() in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=str(model))
    elif model.suffix.lower() == '.fbx':
        bpy.ops.import_scene.fbx(filepath=str(model))
    else:
        raise ValueError('Forge bridge accepts rigged GLB, glTF or FBX')
    arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    for o in bpy.context.scene.objects:
        if o.parent is None:
            o.rotation_euler.z += math.radians(j['turn'])
    bpy.context.view_layer.update()
    # Keep the imported current pose while dropping actions that would fight the fit.
    arm.animation_data_clear()
    target.mkdir(parents=True, exist_ok=True)
    (target / 'start').mkdir(exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(target / 'model.blend'))
    old = sys.argv
    try:
        sys.argv = ['export_rig.py', '--', '--out', str(target / 'rig')]
        runpy.run_path(str(Path(__file__).with_name('export_rig.py')))
    finally:
        sys.argv = old
    c = j['calib']
    cx, cy, cz = c['center']
    # Forge's front looks +Y; its side looks -X. Both share world +Z up.
    views = []
    for name, loc, right, forward, x0 in (
        ('front', [cx, cy - 100, cz], [1, 0, 0], [0, 1, 0], 0),
        ('side', [cx + 100, cy, cz], [0, 1, 0], [-1, 0, 0], 960),
    ):
        views.append(dict(name=name, type='ortho', location=loc, right=right, up=[0, 0, 1],
                          forward=forward, width_px=960, height_px=1080, x0_px=x0,
                          ortho_width=c['width'], ortho_height=c['height']))
    (target / 'start' / 'cameras.json').write_text(json.dumps({'frame_px': [1920, 1080], 'views': views}))
    shutil.copyfile(start, target / 'start' / 'start.png')
    (target / 'creature.json').write_text(json.dumps({'name': model.stem, 'facing': '-Y'}))


if __name__ == '__main__':
    main()
