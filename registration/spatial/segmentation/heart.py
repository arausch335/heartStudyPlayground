def heart_segmentation_algorithm(frame, **kwargs) -> dict:
    """
    Segment epicardium from a point cloud.
    Returns a dict like: { "indices": np.ndarray, "mesh": mesh_obj, "metrics": {...} }
    """
    return {'indices': []}