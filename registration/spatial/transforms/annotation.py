import numpy as np


class AnnotationStep:
    """
    Non-mutating step that records information about a subset/segment
    without changing the active point set.
    """
    def __init__(self, name, *, stage=None):
        self.kind = "annotation"
        self.name = name
        self.stage = stage

        self.metadata = None


class SegmentationStep(AnnotationStep):
    """
    Annotation step representing a segmentation.

    Core attributes (identity):
      - label
      - subset_raw_indices

    Metadata captures:
      - index semantics
      - provenance / safety checks
      - algorithm details
    """

    def __init__(
        self,
        name,
        *,
        label,
        subset_raw_indices,
        stage="segmentation",
        subset_object=None,
    ):
        super().__init__(name=name, stage=stage)

        self.kind = "segmentation"
        self.label = str(label)

        self.subset_raw_indices = np.asarray(subset_raw_indices, dtype=int).reshape(-1)
        self.subset_object = subset_object

    # ------------------------------------------------------------
    # Metadata setter
    # ------------------------------------------------------------
    def set_metadata(
        self,
        *,
        source_space,
        raw_indices_count=None,
        active_indices_at_time=None,
        previous_index_count=None,
        algorithm=None,
        params=None,
        metrics=None,
    ):

        if source_space not in {"raw", "active", "processed"}:
            raise ValueError(f"Invalid source_space '{source_space}'")

        if raw_indices_count is None:
            raw_indices_count = int(self.subset_raw_indices.size)

        md = {
            # --- index semantics ---
            "raw_index_space": "raw",
            "indices_format": "indices",
            "raw_indices_count": int(raw_indices_count),

            # --- provenance / safety ---
            "source_space": str(source_space),
            "active_indices_at_time": (
                np.asarray(active_indices_at_time, dtype=int).tolist()
                if active_indices_at_time is not None
                else None
            ),
            "previous_index_count": (
                int(previous_index_count)
                if previous_index_count is not None
                else None
            ),

            # --- algorithm ---
            "algorithm": str(algorithm) if algorithm is not None else None,
            "params": dict(params) if params is not None else {},
            "metrics": dict(metrics) if metrics is not None else {},
        }

        self.metadata = md
        return self

