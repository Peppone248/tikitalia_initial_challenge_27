# Notebooks

Exploration lives here. Name notebooks `NN_initials_topic.ipynb` (e.g. `01_gp_ball_visibility.ipynb`)
so they sort in order and everyone knows whose they are.

- Clear large outputs before committing (videos frames, big tables) — keep diffs reviewable.
- Anything you reuse goes in `src/tikitalia/`.
- Path setup at the top of each notebook:

```python
import sys; sys.path[:0] = ["..", "../src"]
from games import asset_paths
```

The official guided tours are in `starter/` (01 smart tagging, 02 tracking, 03 first submission).
