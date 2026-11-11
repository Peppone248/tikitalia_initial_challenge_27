# analysis/

This folder holds the script that compares the models, useful to understand how they behave and
which combination to use in the ensemble.

`compare_models.py` compares the models without re-running inference. Prerequisites: steps 1 and 2
of `inference/` (`detections.csv` and `track.csv` of each model).

```bash
python -m analysis.compare_models
python -m analysis.compare_models --models tikitalia_tiles roboflow
```

Tables produced (without ground truth: behaviour and agreement, not accuracy):

- **A.** raw detections: coverage, off pitch, impossible jumps (> 40 m/s)
- **B.** post-processed tracks: ball present, segments, impossible jumps
- **C.** agreement between pairs of models (within 15 px)
- **D.** ensemble: confirmed / tracked / recovered / synthetic

Output in `runs/ball_pipeline/<video>/`: `comparison.md` (or `comparison_<models>.md`) and the
`ball.csv` of every possible ensemble of the given models.
