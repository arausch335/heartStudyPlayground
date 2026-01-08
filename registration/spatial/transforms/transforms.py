import numpy as np

from registration.spatial.transforms.matrix import MatrixStep
from registration.spatial.transforms.subset import SubsetStep
from registration.spatial.transforms.annotation import AnnotationStep, SegmentationStep
from registration.spatial.utilities.transforms_utils import compose_T, invert_T


class Transforms:
    def __init__(self):
        self.steps = []

    ### --- PROPERTIES --- ###
    @property
    def collapsed(self):
        """
        Property returning (collapsed_T, collapsed_indices) for the full Transforms chain
        """
        c = self.collapse(n_raw_points=None)
        return c["collapsed_T"], c["collapsed_indices"]

    @property
    def inverted_collapsed_transform(self):
        return invert_T(self.collapsed_transform())

    ### --- UPDATE --- ###
    def add(self, step):
        self.steps.append(step)
        return self

    ### --- SUMMARY --- ###
    def summary(self):
        lines = [f"Transforms(n_steps={len(self.steps)})"]
        for i, s in enumerate(self.steps):
            if s.kind == "matrix":
                lines.append(f"  {i:02d}. MATRIX  {s.name}")
            else:
                lines.append(f"  {i:02d}. SUBSET  {s.name}  n={len(s.indices)}")
        return "\n".join(lines)

    def collapse(self, start=None, end=None, stages=None, n_raw_points=None):
        """
        Collapse selected steps into:
          - collapsed_indices: indices into the ORIGINAL starting set (raw)
          - collapsed_T: composed 4x4 matrix (matrix steps only)
        """
        # get initial number of raw points
        if n_raw_points is None:
            n_raw = self._infer_n_raw_points(start=start, end=end, stages=stages)
        else:
            n_raw = int(n_raw_points)
            if n_raw <= 0:
                raise ValueError("n_raw_points must be > 0")

        # initialize active indices array using n
        active = np.arange(n_raw, dtype=int)

        # iterate through transform steps
        for s in self._iter_steps(start=start, end=end, stages=stages):
            # get transform step kind (subset, matrix)
            kind = getattr(s, "kind", None)

            # iterate through subset steps only
            if kind == "subset":
                # get indices of subset step
                idx = np.asarray(s.indices, dtype=int).reshape(-1)

                if len(active) == 0:
                    raise ValueError(
                        f"Cannot collapse: subset step '{getattr(s, 'name', '')}' "
                        f"applied after active set became empty."
                    )

                if idx.size > 0:
                    if idx.min() < 0 or idx.max() >= len(active):
                        raise ValueError(
                            f"Cannot collapse: subset step '{getattr(s, 'name', '')}' indices out of range "
                            f"(expects N={len(active)} but got idx in [{idx.min()},{idx.max()}])."
                        )

                active = active[idx]

            elif kind == "matrix":
                continue

            else:
                raise ValueError(
                    f"Cannot collapse: unknown step kind '{kind}' for step '{getattr(s, 'name', None)}'"
                )

        T = self.collapsed_transform(start=start, end=end, stages=stages)

        return {
            "collapsed_indices": active,
            "collapsed_T": T,
            "n_raw": int(n_raw),
            "n_kept": int(len(active)),
            "start": start,
            "end": end,
            "stages": list(stages) if stages is not None else None,
        }

    def collapsed_transform(self, start=None, end=None, stages=None):
        """
        Property returning sum of all matrix transforms for the full Transforms chain
        """
        steps = list(self._iter_steps(start=start, end=end, stages=stages))
        matrices = [np.asarray(s.T, dtype=float) for s in steps if getattr(s, "kind", None) == "matrix"]
        return compose_T(matrices)

    ### --- HELPER FUNCTIONS --- ###
    def _iter_steps(self, start=None, end=None, stages=None):
        steps = self.steps

        if start is not None or end is not None:
            s = 0 if start is None else int(start)
            e = len(steps) if end is None else int(end)
            steps = steps[s:e]

        if stages is None:
            for st in steps:
                yield st
            return

        stage_set = set(stages)
        for st in steps:
            if getattr(st, "stage", None) in stage_set:
                yield st

    def _infer_n_raw_points(self, start=None, end=None, stages=None):
        """
        Infer the starting active count from step metadata.

        We look for the earliest subset step in the selected range that has:
          step.metadata["previous_index_count"]

        This should be present for your initial processed subset step.
        """
        steps = list(self._iter_steps(start=start, end=end, stages=stages))
        indices = [self.steps.index(step) for step in steps]
        stage_set = set(stages) if stages is not None else None
        start, end = min(indices), max(indices)

        while sum([step.kind == "subset" for step in steps]) == 0:
            start -= 1
            if start < 0:
                raise IndexError("No subset steps found in transform")
            steps = list(self._iter_steps(start=start, end=end))
            steps = [
                step
                for step in steps
                if (
                    stage_set is None
                    or step.stage in stage_set
                    or start <= self.steps.index(step) <= min(indices)
                )
            ]

        for s in steps:
            if getattr(s, "kind", None) != "subset":
                continue
            md = getattr(s, "metadata", None) or {}
            if "previous_index_count" in md:
                n = int(md["previous_index_count"])
                if n <= 0:
                    raise ValueError("metadata['previous_index_count'] must be > 0")
                return n

        raise RuntimeError(
            "Could not infer n_raw_points. Provide n_raw_points explicitly, "
            "or ensure at least one subset step in the selected range has "
            "metadata['previous_index_count'] set (typically the first processed subset)."
        )


### --- HELPER FUNCTIONS --- ###
