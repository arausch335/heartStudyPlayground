from registration.spatial.frame import IOFrame
from registration.spatial.load_io_data.preprocessing import *

import open3d as o3d
import numpy as np
import os


def load_io_data(env, data_dir=None):
    """

    """
    frames = []

    # find data directory with .ply files
    data_dir = env.IO_DATA_PATH if data_dir is None else data_dir
    frame_files = os.listdir(data_dir)
    if len(frame_files) == 0:
        raise IOError("No IO frames found in %s" % data_dir)

    frame_index = 0
    for frame_file in frame_files:
        # filter for .ply files
        if not frame_file.endswith(".ply"):
            continue

        # convert points to np array
        frame_path = os.path.join(data_dir, frame_file)
        pcd = o3d.io.read_point_cloud(frame_path)
        raw_points = np.array(pcd.points)

        # create IO Frame object
        frame = IOFrame(env)

        # set raw (original) points and metadata, which includes frame index and path
        frame.set_raw_points(raw_points)
        frame.set_metadata(frame_index, frame_path)

        # preprocess frame through outlier removal and normalization
        frame.preprocessing_pipeline = PreprocessingPipeline(raw_points)
        frame.preprocessing_pipeline.preprocess()

        frame.transforms = frame.preprocessing_pipeline.transforms
        frame.active_indices = frame.preprocessing_pipeline.active_indices
        frame.index_maps[("raw", "processed")] = frame.active_indices.copy()
        frame.points = frame.preprocessing_pipeline.processed_points
        frame.points_stage = "processed"
        frame.validate_points_consistency(strict=True)

        # add frame to list and increase frame index counter
        frames.append(frame)
        frame_index += 1

    return frames
