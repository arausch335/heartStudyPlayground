from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import numpy as np

from registration.spatial.utils.transforms_utils import apply_T, compose_T, invert_T


# ------------------------------------------------------------
# Individual transform step
# ------------------------------------------------------------

@dataclass
class TransformStep:
    """
    One step in a transformation chain.

    For reversible transforms:
      - matrix stores T_dst_from_src (4x4)
    For non-reversible steps (e.g., cropping), matrix can be None and
    we store metadata describing what was done.
    """
    name: str
    matrix: Optional[np.ndarray]  # 4x4 or None
    meta: Dict[str, Any]

    def has_matrix(self) -> bool:
        return self.matrix is not None

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "meta": self.meta,
        }
        if self.matrix is not None:
            out["matrix"] = self.matrix.tolist()
        else:
            out["matrix"] = None
        return out

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TransformStep":
        mat = d.get("matrix", None)
        if mat is not None:
            mat = np.array(mat, dtype=float)
        return TransformStep(
            name=d.get("name", "unnamed"),
            matrix=mat,
            meta=d.get("meta", {}),
        )


# ------------------------------------------------------------
# Chain of transforms
# ------------------------------------------------------------

class TransformChain:
    """
    Ordered list of transform steps applied to a point cloud.

    Conceptually:
        p_out = T_n ... T_2 T_1 p_in

    where each T_i is a 4x4 matrix mapping source -> destination for that step.
    Non-matrix steps (crop, mask) are stored as metadata only.
    """

    def __init__(self, steps: Optional[List[TransformStep]] = None):
        self.steps: List[TransformStep] = steps or []

    # ------------ adding steps ------------

    def add_step(self, step: TransformStep) -> "TransformChain":
        self.steps.append(step)
        return self

    def add_rigid_step(self,
                       name: str,
                       R: np.ndarray,
                       t: np.ndarray,
                       meta: Optional[Dict[str, Any]] = None) -> "TransformChain":
        """
        Add a rigid transform step with rotation R (3x3) and translation t (3,).
        """
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        step = TransformStep(
            name=name,
            matrix=T,
            meta=meta or {},
        )
        return self.add_step(step)

    def add_matrix_step(self,
                        name: str,
                        T: np.ndarray,
                        meta: Optional[Dict[str, Any]] = None) -> "TransformChain":
        """
        Add a general 4x4 transform step.
        """
        T = np.asarray(T, dtype=float)
        assert T.shape == (4, 4)
        step = TransformStep(
            name=name,
            matrix=T,
            meta=meta or {},
        )
        return self.add_step(step)

    def add_crop_step(self,
                      name: str,
                      bbox: Dict[str, float],
                      meta: Optional[Dict[str, Any]] = None) -> "TransformChain":
        """
        Store a cropping step. This is non-reversible (no matrix).
        bbox example:
          {"xmin": ..., "xmax": ..., "ymin": ..., "ymax": ..., "zmin": ..., "zmax": ...}
        """
        combined_meta = {"bbox": bbox}
        if meta:
            combined_meta.update(meta)
        step = TransformStep(
            name=name,
            matrix=None,
            meta=combined_meta,
        )
        return self.add_step(step)

    def add_normalization_step(self,
                               center: np.ndarray,
                               scale: float,
                               name: str = "normalize",
                               meta: Optional[Dict[str, Any]] = None) -> "TransformChain":
        """
        Add a normalization step: p_norm = (p - center) / scale
        This *is* reversible, so we store the corresponding 4x4 matrix.
        """
        center = np.asarray(center, dtype=float).reshape(3)
        S = 1.0 / float(scale)
        R = np.eye(3) * S
        t = -center * S

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        full_meta = {"center": center.tolist(), "scale": float(scale)}
        if meta:
            full_meta.update(meta)

        step = TransformStep(
            name=name,
            matrix=T,
            meta=full_meta,
        )
        return self.add_step(step)

    # ------------ composition & application ------------

    def composite_matrix(self) -> np.ndarray:
        """
        Compose all *matrix* steps in order into a single 4x4 transform.

        Steps without matrices (crop, etc.) are ignored in the matrix product,
        but their metadata remains available for bookkeeping.
        """
        matrices = [step.matrix for step in self.steps if step.matrix is not None]
        return compose_T(matrices)

    def inverse_composite_matrix(self) -> np.ndarray:
        """
        Inverse of the composite matrix (assuming all matrix steps are invertible).
        """
        T = self.composite_matrix()
        return invert_T(T)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Apply all matrix transforms in sequence to (N,3) points.
        Non-matrix steps are ignored here (they affect which points exist, not
        their coordinates).
        """
        T = self.composite_matrix()
        return apply_T(points, T)

    def apply_inverse(self, points: np.ndarray) -> np.ndarray:
        """
        Apply inverse of the composite transform.
        """
        T_inv = self.inverse_composite_matrix()
        return apply_T(points, T_inv)

    # ------------ serialization ------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.as_dict() for s in self.steps],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TransformChain":
        steps_data = d.get("steps", [])
        steps = [TransformStep.from_dict(sd) for sd in steps_data]
        return TransformChain(steps=steps)

    # ------------ convenience ------------

    def describe(self) -> None:
        """
        Print a human-readable summary of the chain.
        """
        print("TransformChain:")
        for i, step in enumerate(self.steps):
            has_mat = "matrix" if step.matrix is not None else "NO matrix"
            print(f"  [{i}] {step.name:15s}  ({has_mat})  meta={step.meta}")
