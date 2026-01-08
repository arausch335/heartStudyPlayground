from registration.spatial.frame_registrar import FrameRegistrar
from registration.spatial.load_po_data.load_po_data import load_po_series, load_po_frames
from registration.spatial.load_io_data.load_io_data import load_io_data


class Scene:
    """
    High-level container for a case.

    Key idea (no ORCoordinateGrid):
    ------------------------------
        - OR space is defined implicitly as the output of stage="registered".
        - The target frame defines OR when we add its first MatrixStep(stage="registered").
        - Other frames add their own registered-stage MatrixStep (e.g., ICP result).
    """

    def __init__(self, env):
        self.env = env

        self.io_frames = None
        self.target_frame = None
        self.registrar = None

        self.po_series = None
        self.po_frames = None

    ### --- IO LOADING --- ###
    def load_io_data(self, data_dir=None, target_index=0):
        """
        Load intraoperative scan (IO) frames only.

        Expected result:
          - frames have raw_points
          - preprocessing already ran (or was run during load):
              frame.points, frame.active_indices, frame.transforms (stage='processed')
        """
        data_dir = data_dir if data_dir is not None else self.env.IO_DATA_PATH
        self.io_frames = load_io_data(self.env, data_dir)

        if not self.io_frames:
            raise RuntimeError("No IO frames loaded")

        self.target_frame = self.io_frames[target_index]
        return self

    ### --- PO LOADING --- ###
    def load_po_data(self, data_dir=None, load_view="SAX", replace=False):
        data_dir = data_dir if data_dir is not None else self.env.PO_DATA_PATH

        self.po_series = load_po_series(data_dir, load_view, replace=replace)
        if not self.po_series:
            raise RuntimeError("No PO series loaded")

        self.po_frames = load_po_frames(self.po_series[0])
        return self

    ### --- SEGMENT --- ###
    def segment_io_frames(self, which="all"):
        """
        Run segmentation (e.g., retractor) on frames.

        `which`:
          - "all"
          - "target"
          - list of indices
        """
        if self.io_frames is None:
            raise RuntimeError("IO frames not loaded")

        if which == "all":
            frames = self.io_frames
        elif which == "target":
            if self.target_frame is None:
                raise RuntimeError("target_frame not set")
            frames = [self.target_frame]
        elif isinstance(which, (list, tuple)):
            frames = [self.io_frames[int(i)] for i in which]
        else:
            raise ValueError("which must be 'all', 'target', or a list of indices")

        for f in frames:
            if getattr(f, "segment", None) is None:
                raise RuntimeError(f"Frame {getattr(f, 'n', '?')} has no segment() method")
            f.segment()

        return self

    ### --- DEFINE OR (SEPARATE) --- ###
    def define_or_from_target(self):
        """
        Define OR space by aligning the target frame to its retractor plane and
        recording that alignment as the first registered-stage MatrixStep.

        Important:
        - This is NOT segmentation.
        - This is NOT ICP to another frame.
        - This produces the canonical OR axes/origin for the case.
        """
        if self.target_frame is None:
            raise RuntimeError("target_frame not set")
        if getattr(self.target_frame, "retractor", None) is None:
            raise RuntimeError("Target frame has no retractor. Run segment_io_frames('target') first.")
        if not getattr(self.target_frame.retractor, "segmented", False):
            raise RuntimeError("Target frame retractor not segmented.")

        # align_to_retractor_plane should:
        #  - compute T (processed->registered) from retractor info
        #  - add MatrixStep(stage='registered') to frame.transforms
        self.target_frame.align_to_retractor_plane(stage="registered")

        return self

    ### --- REGISTRATION (SEPARATE) --- ###
    def register_io_frames(self, **registrar_kwargs):
        """
        Register all non-target IO frames into OR (ICP or other method).

        Pre-reqs:
          - target_frame defined
          - OR defined on target via define_or_from_target()
          - each frame has a retractor segmentation (raw indices)
        """
        if self.io_frames is None:
            raise RuntimeError("IO frames not loaded")
        if self.target_frame is None:
            raise RuntimeError("target_frame not set")

        self.registrar = FrameRegistrar(
            frames=self.io_frames,
            target_frame=self.target_frame,
            **registrar_kwargs,
        )
        self.registrar.register_all()
        return self

    ### --- CONVENIENCE --- ###
    def select_frame(self, data_type, **kwargs):
        if data_type == "io":
            frames = self.io_frames
        elif data_type == "po":
            frames = self.po_frames
        else:
            raise ValueError(f"{data_type} is not supported")

        if frames is None:
            raise RuntimeError(f"No frames loaded for data_type='{data_type}'")

        n = kwargs.get("n", None)
        phase_index = kwargs.get("phase_index", None)
        cine_phase_index = kwargs.get("cine_phase_index", None)

        if n is not None:
            return frames[int(n)]

        if phase_index is not None:
            for f in frames:
                if getattr(f, "phase_index", None) == phase_index:
                    return f

        if cine_phase_index is not None:
            for f in frames:
                if getattr(f, "cine_phase_index", None) == cine_phase_index:
                    return f

        return None

    @staticmethod
    def apply(objs, func=None, rel_func=None, **kwargs):
        if objs is None:
            return

        if func is not None:
            for obj in objs:
                func(obj, **kwargs)

        if rel_func is not None:
            for obj in objs:
                method = getattr(obj, rel_func, None)
                if method is None:
                    raise AttributeError(f"{obj} has no method '{rel_func}'")
                method(**kwargs)
