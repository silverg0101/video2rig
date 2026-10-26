"""Blender integration regression for glTF-imported armature display widgets."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BLENDER = Path(os.environ.get(
    "BLENDER_EXE", r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"))


MAKE_FIXTURE = r'''
import bpy
import sys

args = sys.argv[sys.argv.index('--') + 1:]
out = args[args.index('--out') + 1]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# A .859 m box is a tiny stand-in for a skinned creature.
z = .859
verts = [(-.2, -.1, 0), (.2, -.1, 0), (.2, .1, 0), (-.2, .1, 0),
         (-.2, -.1, z), (.2, -.1, z), (.2, .1, z), (-.2, .1, z)]
faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
mesh_data = bpy.data.meshes.new('CreatureMesh')
mesh_data.from_pydata(verts, [], faces)
mesh_data.update()
mesh = bpy.data.objects.new('CreatureMesh', mesh_data)
bpy.context.collection.objects.link(mesh)

arm_data = bpy.data.armatures.new('CreatureSkeleton')
arm = bpy.data.objects.new('CreatureSkeleton', arm_data)
bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
root = arm_data.edit_bones.new('root')
root.head, root.tail = (0, 0, 0), (0, 0, z * .5)
top = arm_data.edit_bones.new('top')
top.head, top.tail, top.parent = (0, 0, z * .5), (0, 0, z), root
bpy.ops.object.mode_set(mode='OBJECT')

mesh.parent = arm
modifier = mesh.modifiers.new('Skin', 'ARMATURE')
modifier.object = arm
lower = mesh.vertex_groups.new(name='root')
upper = mesh.vertex_groups.new(name='top')
for vertex in mesh.data.vertices:
    (lower if vertex.co.z <= z * .5 else upper).add([vertex.index], 1.0, 'REPLACE')

bpy.ops.object.select_all(action='DESELECT')
arm.select_set(True)
mesh.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', use_selection=True,
                           export_skins=True, export_animations=False)
'''


INSPECT_IMPORT = r'''
import bpy
import json
import sys

model = sys.argv[sys.argv.index('--') + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=model)
widgets = []
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        for bone in obj.pose.bones:
            shape = bone.custom_shape
            if shape is not None:
                widgets.append({'name': shape.name, 'dimensions': list(shape.dimensions)})
print('IMPORT_WIDGETS_JSON ' + json.dumps(widgets), flush=True)
'''


class AutorigWidgetRegression(unittest.TestCase):
    @unittest.skipUnless(BLENDER.is_file(), f"Blender 5.1 executable not found: {BLENDER}")
    def test_gltf_armature_widgets_do_not_expand_autorig_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            model = work / "skinned-creature.glb"
            fixture = work / "make_fixture.py"
            inspector = work / "inspect_import.py"
            output = work / "autorig"
            fixture.write_text(MAKE_FIXTURE, encoding="utf-8")
            inspector.write_text(INSPECT_IMPORT, encoding="utf-8")

            generated = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup", "--python", str(fixture),
                 "--", "--out", str(model)], cwd=REPO, capture_output=True, text=True, timeout=180)
            self.assertEqual(generated.returncode, 0, generated.stdout[-4000:] + generated.stderr[-4000:])
            self.assertTrue(model.is_file())

            imported = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup", "--python", str(inspector), "--", str(model)],
                cwd=REPO, capture_output=True, text=True, timeout=180)
            self.assertEqual(imported.returncode, 0, imported.stdout[-4000:] + imported.stderr[-4000:])
            marker = next((line.removeprefix("IMPORT_WIDGETS_JSON ") for line in imported.stdout.splitlines()
                           if line.startswith("IMPORT_WIDGETS_JSON ")), "[]")
            widgets = json.loads(marker)
            self.assertTrue(widgets, "Blender import must reproduce the armature's custom display widget")
            self.assertGreater(max(max(widget["dimensions"]) for widget in widgets), 1.5, widgets)

            prepared = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup", "--python",
                 str(REPO / "blender" / "autorig_prepare.py"), "--", "--model", str(model),
                 "--out", str(output), "--voxel", "0.02"],
                cwd=REPO, capture_output=True, text=True, timeout=240)
            self.assertEqual(prepared.returncode, 0, prepared.stdout[-5000:] + prepared.stderr[-5000:])

            with np.load(output / "shell.npz", allow_pickle=False) as shell:
                height = float(shell["height"])
                mesh_names = shell["meshes"].tolist()
            self.assertAlmostEqual(height, .859, delta=.002)
            self.assertEqual(len(mesh_names), 1, mesh_names)


if __name__ == "__main__":
    unittest.main()
