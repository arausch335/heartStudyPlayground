import numpy as np


class MatrixStep:
    def __init__(self, name, T, stage=None):
        self.kind = "matrix"
        self.name = str(name)
        self.stage = stage  # e.g. "preprocess", "register"
        self.T = np.asarray(T, dtype=float)
        if self.T.shape != (4, 4):
            raise ValueError("MatrixStep requires (4,4)")

        self.metadata = None

    def __repr__(self):
        return f"MatrixStep(name={self.name!r})"

    def set_metadata(self,
                     description,
                     from_space,
                     to_space,
                     method,
                     params=None,
                     metrics=None):
        self.metadata = {
            "description": str(description),
            "from_space": str(from_space),
            "to_space": str(to_space),
            "method": str(method),
            "params": dict(params) if params is not None else {},
            "metrics": dict(metrics) if metrics is not None else {},
        }
