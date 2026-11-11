# `tikitalia_tiles` on the BAR-ATH video

*11 November 2026, whole video (199,319 frames).*

`tikitalia_tiles` = YOLOv8m-P2 trained on 640 tiles (`weights/tikitalia_tiles.pt`, epoch 12),
compared with `tikitalia_model` (YOLOv8s-P2, full frames at 1280).

| | `tikitalia_model` | `tikitalia_tiles` |
|---|---|---|
| Frames with an on-pitch detection | 54.1% | **66.3%** |
| Off-pitch detections | 18.6% | 21.6% |
| Impossible jumps (raw) | 6.3% | 7.3% |
| Ball present in the track | 66.0% | **75.8%** |
| Impossible jumps (track) | 0.9% | **0.7%** |
| Agreement with `roboflow` (raw) | 92.7% | 84.8% |
| Frames agreeing with `roboflow` (raw) | 49.6% | **55.0%** |

| Ensemble | Confirmed | Ball present |
|---|---|---|
| `tikitalia_model` + `roboflow` (default) | 48.9% | 70.2% |
| **`tikitalia_tiles` + `roboflow`** | **54.3%** | **75.4%** |
| All three (2 votes out of 3) | 60.7% | 77.5% |

Without ground truth these are measures of coverage and consistency, not of accuracy. The default
ensemble and the ATD have not been changed: to adopt `tikitalia_tiles` change `DEFAULT_MODELS` in
`utils/config.py`.

Full reports in `runs/ball_pipeline/BAR-ATH_tactical/`: `comparison.md`,
`comparison_tikitalia_tiles+roboflow.md`, `comparison_tikitalia_model+tikitalia_tiles+roboflow.md`.
