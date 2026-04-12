# computer-vision-pipeline-for-multi-object-detection-tracking-and-spatial-analysis

This repo contains small, runnable scripts for a CV pipeline:

- Extract a short clip from a longer video
- Detect objects
- Track objects with **persistent IDs** (BoT-SORT / ByteTrack)
- Enforce a **scene boundary / active area** so detections are restricted to the main scene

## 1) Extract a 5-minute clip

```powershell
.\venv\Scripts\python.exe .\extract_clip.py --input videoplayback.mp4 --output output\clip_5min.mp4 --start 0 --duration 300
```

## 2) Detection + Tracking + Scene ROI filtering (CRITICAL)

Runs YOLO detection per-frame, then tracks with BoT-SORT (default) to keep IDs stable, while filtering to a single
"active area" (scene boundary).

```powershell
.\venv\Scripts\python.exe .\track_cv.py --source output\clip_5min.mp4 --output-video output\tracked.mp4 --output-csv output\tracks.csv
```

Notes:

- If you do not pass `--roi`, the script auto-detects a scene boundary from the first frame and saves it to
	`output/scene_roi.json`.
- To force a specific ROI:

```powershell
.\venv\Scripts\python.exe .\track_cv.py --source output\clip_5min.mp4 --roi output\scene_roi.json
```

Outputs:

- `output/tracked.mp4`: annotated video with ROI + track IDs
- `output/tracks.csv`: per-frame track log (frame/time/id/class/conf/bbox)

