"""Finite provisional weights when both bone-heat attempts find no solution."""
import numpy as np


def segment_weights(points, heads, tails):
    points, heads, tails = (np.asarray(x, dtype=float) for x in (points, heads, tails))
    if len(heads) == 0 or not all(np.isfinite(x).all() for x in (points, heads, tails)):
        raise ValueError('finite vertices and deforming bones are required')
    distance = np.empty((len(points), len(heads)))
    for i, (head, tail) in enumerate(zip(heads, tails)):
        delta = tail - head
        along = np.clip((points - head) @ delta / max(delta @ delta, 1e-12), 0, 1)
        distance[:, i] = np.linalg.norm(points - head - along[:, None] * delta, axis=1)
    nearest = np.argsort(distance, axis=1)[:, :2]
    weights = np.zeros_like(distance)
    values = 1 / np.maximum(np.take_along_axis(distance, nearest, axis=1), 1e-6) ** 4
    values /= values.sum(axis=1, keepdims=True)
    np.put_along_axis(weights, nearest, values, axis=1)
    return weights
