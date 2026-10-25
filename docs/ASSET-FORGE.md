# Asset Forge stage adapter

This fork adds adapters; the original command-line workflow remains available.
It does not depend on Seedance: Forge supplies its own model, calibrated first
frame and MiniMax H3 video, or reuses a saved video.

Multi-GPU hosts pass `reader.prepare --engine CYCLES`. The training renderer
explicitly selects CUDA devices under `CUDA_VISIBLE_DEVICES` and reports their
IDs. Eevee's graphics device does not follow a CUDA mask and is unsuitable for
Forge's shared host allocation.
Cycles training strips skip denoising; reader augmentation already includes noise
and blur. This avoids a CPU denoising pass dominating GPU rendering time.

- `blender/forge_prepare.py`: imports a rigged GLB/FBX, applies Forge's filming
  turn, exports the rig and saves a separate Blender source with the exact Forge
  front/side calibration. It never overwrites the source asset.
- `python -m video2rig.forge read`: reads a clip using its creature's reader and
  writes normalized, named observations that Forge's solver can consume.
- `python -m video2rig.forge fit`: fits either those native readings (including
  surface landmarks) or Forge's detector observations on the same clip.
- `blender/forge_import.py`: imports the fitted NPZ on the prepared rig, removes
  the filming turn and exports an animated GLB.

Observation NPZ fields: `front`, `side` (frames, points, xy in panel 0..1),
`vis_f`, `vis_s` (frames, points, confidence), `detected` (points). Forge's
`joints.json` binds point indices to bone heads/tails. Surface landmarks that
have no exact Forge point remain available to the native solver in `reader.npz`.
Unmapped observations have zero confidence; fewer than three mapped points fail.

The reader and solver are independently selectable in Forge. The native solver
exports the complete fitted clip, with its own cleanup, rather than silently
applying Forge's loop cutting to a different fit. Creature clip jobs can apply
their usual downstream retarget/loop checks. A comparison must record these
different cleanup stages and must not call detector consistency an animation
quality score.

Targeted contract tests: `python -m unittest discover -s tests -p test_forge.py`.
