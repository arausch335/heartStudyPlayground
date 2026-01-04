import numpy as np


class SubsetStep:
    def __init__(self, name, indices, stage=None):
        self.kind = "subset"
        self.name = str(name)
        self.stage = stage
        self.indices = np.asarray(indices, dtype=int).reshape(-1)

        self.metadata = None

    def __repr__(self):
        return f"SubsetStep(name={self.name!r}, n={len(self.indices)})"

    def set_metadata(self,
                     description,
                     rule,
                     previous_index_count,
                     params=None):
        self.metadata = {
            "description": str(description),
            "rule": str(rule),
            "previous_index_count": int(previous_index_count),
            "params": dict(params) if params is not None else {}
        }
