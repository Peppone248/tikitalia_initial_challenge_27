# runs

Results of `scripts/run_lomo.py` go here as `lomo_<models>_<seeds>.md` (tables are generated, one file per launch).

Full comparison (decision criterion tcn_change vs frame):

    sequence_model/.venv/Scripts/python.exe sequence_model/scripts/run_lomo.py --models baseline50 frame tcn tcn_change --seeds 2027 2028

The (beta, w) of tcn_change is chosen in-sample on each fold's 6 training matches (no inner retraining).
