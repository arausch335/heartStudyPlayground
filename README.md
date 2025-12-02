# Heart Study Playground

Utilities to mask rectangular retractors within point clouds and align clouds via point-to-plane ICP. The `spatial_registration` package includes loaders for `.npy`, ASCII `.ply`, and `.obj` files, rectangle detection, masking, and visualization helpers.

## Sample data

Synthetic test clouds are generated with two parallel rectangles and a perpendicular rectangle. Run:

```bash
python data/generate_sample_data.py
```

This produces `sample_target` and `sample_source` files in `.npy`, `.ply`, and `.obj` formats under `data/`.

## Running the pipeline

With dependencies installed (`numpy`, `scipy`, and `matplotlib`), call the pipeline directly from Python. Defaults point to the
sample pair and write outputs to `data/outputs`:

```python
from pathlib import Path
from spatial_registration.main import main

main(
    target_path=Path("data/sample_target.ply"),
    source_path=Path("data/sample_source.ply"),
)
```

Outputs include masked point clouds, rectangle metadata, and PNG overlays for masks, rectangle fits, and the aligned point
clouds. Parameters such as `distance_threshold`, `icp_iterations`, and `correspondence_threshold` can be overridden via the
function call.
