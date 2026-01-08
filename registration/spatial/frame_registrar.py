import numpy as np

from registration.spatial.icp.icp import point_to_plane_icp
from registration.spatial.transforms.transforms import MatrixStep
from registration.spatial.utilities.visualization import plot_two_registered_retractors


class FrameRegistrar:
    """
    Registers IO frames into OR space using the retractor as the anchor.

    Modern design (Transforms-driven)
    ---------------------------------
    - No ORCoordinateGrid.
    - OR space is defined as the output of stage="registered".
    - The target_frame already has a registered-stage MatrixStep that defines OR
      (typically from retractor alignment).
    - For each other frame, we estimate a PROCESSED->REGISTERED transform via ICP and
      record it as MatrixStep(stage="registered") on that frame.

    Assumptions
    -----------
    - Each frame has:
        frame.raw_points
        frame.transforms
        frame.retractor with subset indices in RAW space
    - Segmentation runs on active/preprocessed points, but subset indices are stored as RAW indices.
    """

    def __init__(
        self,
        frames,
        target_frame,
        *,
        max_iterations=75,
        max_correspondence_distance=15.0,
        normal_radius=25.0,
        visualize=False,
    ):
        self.frames = list(frames) if frames is not None else []
        self.target_frame = target_frame

        self.max_iterations = int(max_iterations)
        self.max_correspondence_distance = float(max_correspondence_distance)
        self.normal_radius = float(normal_radius)
        self.visualize = bool(visualize)

    ### --- INTERNAL HELPERS --- ###
    def _require_target_registration_T(self):
        """
        Returns the target frame's PROCESSED->REGISTERED transform from its transforms chain.
        """
        if self.target_frame is None:
            raise RuntimeError("FrameRegistrar.target_frame is None")

        if getattr(self.target_frame, "transforms", None) is None:
            raise RuntimeError("Target frame has no transforms")

        if not hasattr(self.target_frame.transforms, "collapse"):
            raise RuntimeError(
                "Target frame transforms missing collapse()"
            )

        collapsed = self.target_frame.transforms.collapse(
            stages=["registered"],
            n_raw_points=len(self.target_frame.raw_points),
        )
        T = collapsed["collapsed_T"]

        has_registration = any(
            getattr(step, "kind", None) == "matrix" and getattr(step, "stage", None) == "registered"
            for step in self.target_frame.transforms.steps
        )
        if not has_registration:
            raise RuntimeError(
                "Could not obtain target PROCESSED->REGISTERED transform (registration stage missing?)"
            )

        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(
                f"Target registration transform must be (4,4), got {T.shape}"
            )

        return T

    def _get_retractor_raw_indices(self, frame):
        """
        Get retractor indices stored in RAW index space.
        """
        r = getattr(frame, "retractor", None)
        if r is None:
            raise RuntimeError(f"Frame {getattr(frame, 'n', '?')} has no retractor object")

        idx = getattr(r, "subset_indices", None)
        if idx is None:
            idx = getattr(r, "raw_indices", None)  # if renamed later
        if idx is None:
            raise RuntimeError(
                f"Frame {getattr(frame, 'n', '?')} retractor has no subset/raw indices"
            )

        idx = np.asarray(idx, dtype=int).reshape(-1)
        return idx

    def _get_target_retractor_registered(self):
        """
        Return target retractor points expressed in registered coordinates.

        Since target defines OR, we:
          - take target retractor RAW points (processed stage)
          - apply target registration transform (processed->registered)
        """
        idx = self._get_retractor_raw_indices(self.target_frame)
        pts_registered = self.target_frame.get_points("registered")
        idx_registered = self.target_frame.map_raw_indices_to_stage(idx, "registered")
        return pts_registered[idx_registered]

    ### --- PUBLIC API --- ###
    def register_all(self):
        """
        Register all frames (except target) into OR space via ICP.

        For each non-target frame:
          - source = frame retractor points in PROCESSED space (RAW points subset)
          - target = target retractor points in REGISTERED space
          - ICP estimates T_registered_from_processed_for_frame
          - record as MatrixStep(stage="registered") on the frame.transforms
          - store rmse in frame.quality_metrics["registration_rmse"]
        """
        if self.target_frame is None:
            raise RuntimeError("target_frame must be provided")

        # Ensure target has registration defined
        _ = self._require_target_registration_T()

        target_retractor_registered = self._get_target_retractor_registered()

        for frame in self.frames:
            if frame is self.target_frame:
                frame.is_registered = True
                continue

            if getattr(frame, "raw_points", None) is None:
                raise RuntimeError(f"Frame {getattr(frame, 'n', '?')} raw_points is None")

            if getattr(frame, "transforms", None) is None:
                raise RuntimeError(f"Frame {getattr(frame, 'n', '?')} has no transforms")

            idx = self._get_retractor_raw_indices(frame)
            source_processed = frame.get_points("processed")
            idx_processed = frame.map_raw_indices_to_stage(idx, "processed")
            source_retractor_processed = source_processed[idx_processed]

            # ICP estimates SOURCE(processed) -> TARGET(registered)
            T_registered_from_processed, rmse, info = point_to_plane_icp(
                source=source_retractor_processed,
                target=target_retractor_registered,
                max_correspondence_distance=self.max_correspondence_distance,
                normal_radius=self.normal_radius,
                max_iterations=self.max_iterations,
            )

            step = MatrixStep(
                name="register/icp_to_or",
                stage="registered",
                T=T_registered_from_processed,
            )
            step.set_metadata(
                description="Register frame into OR space using point-to-plane ICP on retractor points",
                from_space="processed",
                to_space="registered",
                method="icp_point_to_plane",
                params={
                    "max_correspondence_distance": float(self.max_correspondence_distance),
                    "normal_radius": float(self.normal_radius),
                    "max_iterations": int(self.max_iterations),
                    "anchor_frame_n": getattr(self.target_frame, "n", None),
                },
                metrics={
                    "rmse": float(rmse),
                },
            )

            # Optional: attach raw info payload (can be large) if you want it
            # If you DO want this standardized too, we can add a MatrixStep.set_debug_info(...)
            if info is not None:
                # keep inside metadata under a consistent key
                md = step.metadata if hasattr(step, "metadata") else None
                if isinstance(md, dict):
                    md["debug"] = {"icp_info": info}

            frame.transforms.add(step)

            # Quality metrics on frame
            if getattr(frame, "quality_metrics", None) is None:
                frame.quality_metrics = {}
            frame.quality_metrics["registration_rmse"] = float(rmse)

            frame.is_registered = True
            frame.validate_points_consistency(strict=False)

        if self.visualize and len(self.frames) >= 2:
            plot_two_registered_retractors(self.frames[0], self.frames[1])

        return self
