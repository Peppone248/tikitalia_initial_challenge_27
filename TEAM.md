# Tikitalia: More than a Hack 2027 (Phase 1: Ball Possession)

The submission is produced by **[changepoint_model/](changepoint_model/README.md)**: setup,
the command that writes the CSV, the training data and the validation are described there;
[changepoint_model/HOW_IT_WORKS.md](changepoint_model/HOW_IT_WORKS.md) explains the model.

```
python -m venv venv
venv\Scripts\pip install -r changepoint_model\requirements.txt
venv\Scripts\python.exe changepoint_model\predict.py --out outputs\tikitalia.csv
venv\Scripts\python.exe evaluation\validate_submission.py outputs\tikitalia.csv
```

The match data is gated and is not in this repository: put the bundle in `data/` (layout in
[DATA.md](DATA.md)) or point `MTAH_DATA_DIR` at it. `ffmpeg` must be on the PATH (the audio of
the tactical video is read for the whistle).
