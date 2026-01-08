import numpy as np

from environment import Environment
from registration.spatial.load_io_data.io_frame import IOFrame
from registration.spatial.transforms.matrix import MatrixStep
from registration.spatial.transforms.subset import SubsetStep
from registration.spatial.transforms.transforms import Transforms
from registration.spatial.utils.transforms_utils import apply_T, compose_T, invert_T


def _make_translation(tx, ty, tz):
    T = np.eye(4, dtype=float)
    T[:3, 3] = [tx, ty, tz]
    return T


def _make_z_rotation(theta_rad):
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    T = np.eye(4, dtype=float)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def main():
    env = Environment()

    raw_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    frame = IOFrame(env)
    frame.set_raw_points(raw_points)
    frame.transforms = Transforms()

    kept_idx = np.array([0, 2, 4], dtype=int)
    frame.active_indices = kept_idx.copy()
    frame.index_maps[("raw", "processed")] = kept_idx.copy()

    step_subset = SubsetStep(
        name="keep_subset",
        stage="processed",
        indices=kept_idx,
    )
    frame.transforms.add(step_subset)

    T_processed = _make_translation(1.0, -2.0, 0.5)
    frame.transforms.add(
        MatrixStep(
            name="translate_processed",
            stage="processed",
            T=T_processed,
        )
    )

    T_registered = _make_z_rotation(np.deg2rad(30.0))
    frame.transforms.add(
        MatrixStep(
            name="rotate_registered",
            stage="registered",
            T=T_registered,
        )
    )

    frame.points = frame.get_points("processed")
    frame.points_stage = "processed"

    expected_registered = apply_T(
        apply_T(raw_points[kept_idx], T_processed),
        T_registered,
    )

    composed = compose_T([T_processed, T_registered])
    composed_registered = apply_T(raw_points[kept_idx], composed)

    if not np.allclose(expected_registered, composed_registered):
        raise AssertionError("compose_T order mismatch")

    got_registered = frame.get_points("registered")
    if not np.allclose(expected_registered, got_registered):
        raise AssertionError("get_points('registered') mismatch")

    inv_registered = invert_T(T_registered)
    round_trip = apply_T(apply_T(raw_points[kept_idx], T_registered), inv_registered)
    if not np.allclose(round_trip, raw_points[kept_idx]):
        raise AssertionError("invert_T failed round-trip")

    frame.validate_points_consistency(strict=True)

    repeated_a = frame.get_points("registered")
    repeated_b = frame.get_points("registered")
    if not np.allclose(repeated_a, repeated_b):
        raise AssertionError("Repeated get_points calls drifted")

    active_registered = frame.get_active_points("registered")
    if not np.allclose(active_registered, expected_registered):
        raise AssertionError("Active points mismatch in registered stage")

    print("All transform consistency checks passed.")


if __name__ == "__main__":
    main()
