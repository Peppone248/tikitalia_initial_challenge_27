# utils/

This folder holds the modules shared by the other folders: they are not run directly, but this is
where default paths and settings are changed.

| File | Content |
|---|---|
| `config.py` | Default paths (video, homography, weights, output), list of models, CSV paths |
| `io.py` | Reading of `detections.csv`, `.progress` and `track.csv` |
| `pitch.py` | Homography pixels ↔ metres, tile grid, physical parameters, gap filling |

What to change and where:

- match data in another folder: `MTAH_DATA_DIR` environment variable (as in `games.py`);
- default models of the ensemble: `DEFAULT_MODELS` in `config.py`;
- new model: add it to `ALL_MODELS` and handle it in `inference/detect_ball.py`.
