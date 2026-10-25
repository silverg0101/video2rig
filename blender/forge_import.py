"""Import a fitted motion into its own prepared source and export a standalone GLB."""
import json
import math
import runpy
import sys
from pathlib import Path

import bpy


def main():
    project, motion, joints, output = sys.argv[sys.argv.index('--') + 1:]
    j = json.loads(Path(joints).read_text())
    bpy.ops.wm.open_mainfile(filepath=str(Path(project).resolve() / 'model.blend'))
    for o in bpy.context.scene.objects:
        if o.type == 'ARMATURE':
            o.animation_data_clear()
    old = sys.argv
    try:
        sys.argv = ['import_motion.py', '--', '--motion', str(Path(motion).resolve())]
        runpy.run_path(str(Path(__file__).with_name('import_motion.py')))
    finally:
        sys.argv = old
    # The prepared source carries Forge's filming turn. Export in the original model orientation.
    for o in bpy.context.scene.objects:
        if o.parent is None:
            o.rotation_euler.z -= math.radians(j['turn'])
    bpy.ops.export_scene.gltf(filepath=str(Path(output).resolve()), export_format='GLB',
                              export_animations=True, export_animation_mode='ACTIVE_ACTIONS',
                              export_force_sampling=True)


if __name__ == '__main__':
    main()
