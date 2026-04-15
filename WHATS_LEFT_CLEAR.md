# WHAT'S LEFT - CLEAR STATEMENT

**Date:** April 15, 2026  
**Target Score:** 8.2/10 → 9.2+/10

---

## Current Status: 84% Complete

| Category | Status | Work Remaining | Time Est. |
|----------|--------|-----------------|-----------|
| Detection Quality | ✅ Done (95%) | Test YOLOv9 model | 1 hr |
| Tracking Stability | 🔴 **BLOCKED** | **Integrate DeepSORT** | **2-3 hrs** |
| Field Segmentation | ✅ Done (90%) | Fine-tune parameters | 30 min |
| Geometric Calibration | ✅ Done (95%) | Validation only | None |
| Speed Estimation | ✅ Done (90%) | Export smoothed CSV | 30 min |
| Heatmaps | 🟡 Partial (70%) | Semantic zones + events | 1-2 hrs |
| Trajectory Viz | ✅ Done (85%) | Pitch markings overlay | 1 hr |
| CSV Output | ✅ Done (95%) | Per-track metrics | 30 min |

---

## 🔴 CRITICAL BLOCKER #1: DeepSORT Integration

### What's Missing
- DeepSORT wrapper **exists** but **not connected** to main tracking loop
- Currently using: ByteTrack (motion-only) instead of DeepSORT (motion + appearance)
- This directly impacts **ID stability score** (most critical metric)

### What Needs to Be Done

**File:** [track_cv.py](track_cv.py)

**Current State (Lines ~200-300):**
```python
# Currently using:
if args.tracker_type == "bytetrack":
    tracker = BYTETracker(...)
elif args.tracker_type == "botsort":
    tracker = BOTSORT(...)
# DeepSORT NOT integrated here ❌
```

**Required Change:**
```python
# Need to add:
elif args.tracker_type == "deepsort":
    tracker = DeepSORTTracker(...)  # From deepsort_wrapper.py
```

**Dependencies:**
1. Download ReID model: `osnet_x1_0_imagenet.pth` (~25 MB)
   - Source: OSNet or torchvision
   - Location: Place in `models/` folder

2. Update [tracking_config.yaml](tracking_config.yaml):
   ```yaml
   tracker_type: "deepsort"  # Change from "botsort"
   deepsort:
     feature_path: "models/osnet_x1_0_imagenet.pth"  # Ensure path correct
   ```

### Expected Impact
- ✅ ID switches reduced by 50-70% in crowded scenes
- ✅ Score improvement: +2-3 points
- ✅ Tracking stability metric: From 3/10 → 8/10

### Completion Criteria
- [ ] ReID model downloaded
- [ ] DeepSORTTracker integrated in main loop
- [ ] Test on dense scene (set piece)
- [ ] Verify ID consistency > 80% over 5-minute clip

---

## 🟡 SECONDARY BLOCKER #2: YOLOv9 Model Testing

### What's Missing
- YOLOv9 configuration **ready** but model **not tested**
- Config says: `model: "yolov9c.pt"` but still using YOLOv8n
- Will improve detection accuracy by 3-5%

### What Needs to Be Done

**File:** [detection_config.yaml](detection_config.yaml)

**Current:**
```yaml
model: "yolov9c.pt"  # ← Configured but model not available
```

**Required Actions:**
1. Download yolov9c.pt (~25 MB)
   - Source: Ultralytics/YOLOv9 GitHub releases
   - Command: `pip install yolov9` OR download .pt file directly

2. Keep/verify in detection_config.yaml:
   ```yaml
   model: "yolov9c.pt"
   confidence_threshold: 0.60
   iou_threshold: 0.45
   ```

3. Run benchmark test:
   ```bash
   python track_cv.py \
     --source output/clip_5min.mp4 \
     --output-csv output/tracks_yolov9_test.csv \
     --use-enhanced-detection
   ```

### Expected Impact
- Detection accuracy improvement: +3-5%
- False positive reduction: 10-20%
- Score improvement: +1-2 points

### Completion Criteria
- [ ] YOLOv9c.pt downloaded and in working directory
- [ ] Test run completes successfully
- [ ] Compare detection count vs YOLOv8n (should be similar or better)

---

## 🟡 PARTIAL: Heatmap Refinement (Not Critical)

### What's Working
- ✅ Basic presence heatmaps generated
- ✅ World-coordinate projection applied
- ✅ Pixel→meters conversion working
- ✅ Multiple heatmap types available

### What's Missing
- Zone-based segmentation (e.g., attacking/midfield/defensive thirds)
- Event detection specificity (what counts as an "event"?)
- Team-separated heatmaps
- Visualization clarity (color schemes, overlays)

### Files Involved
- [world_space_heatmap.py](world_space_heatmap.py) - Main generation
- [heatmaps.py](heatmaps.py) - Pixel-space heatmaps

### What Needs to Be Done
**Option A (Core):** Define zones semantically
```python
# Add to world_space_heatmap.py:
def zone_heatmap_semantic(self, num_zones=3):  # Attacking/Mid/Defensive
    # Divide field into thirds based on pitch length (105m)
    # Zone 1: 0-35m (defensive)
    # Zone 2: 35-70m (midfielder)
    # Zone 3: 70-105m (attacking)
    pass
```

**Option B (Polish):** Better visualizations
```python
# Add overlay features:
# - Color gradient (cold→hot)
# - Grid lines
# - Zone labels
# - Team separation
```

### Completion Criteria
- [ ] 3 semantic zones defined on pitch
- [ ] Separate heatmaps per team (if multi-team detection)
- [ ] Visual clarity improved (readable labels)

---

## ✅ DONE - Don't Touch

| Item | Status | Quality |
|------|--------|---------|
| Detection Config | ✅ Done | Production-ready |
| Calibration Engine | ✅ Done | Production-ready |
| Homography Transform | ✅ Done | Production-ready |
| Trajectory Smoothing | ✅ Done | Production-ready |
| Validation Suite | ✅ Done | Production-ready |
| CSV Schema | ✅ Done | Production-ready |
| CSV Output Generation | ✅ Done | Production-ready |
| ROI Management | ✅ Done | Production-ready |
| Trajectory Viz (basic) | ✅ Done | Production-ready |

---

## EXECUTION ROADMAP (Next 3 Days)

### Day 1: Critical Fixes (Must Do)
- [ ] **2-3 hours:** Complete DeepSORT integration
  - Download ReID model
  - Integrate into track_cv.py main loop
  - Test on sample video
  
- [ ] **1 hour:** YOLOv9 setup and test
  - Download yolov9c.pt
  - Run benchmark comparison
  - Verify improvements

### Day 2: Output Generation
- [ ] **15-20 min:** Run full pipeline with DeepSORT
  ```bash
  python track_cv.py \
    --source output/clip_5min.mp4 \
    --tracker deepsort \
    --use-enhanced-detection \
    --use-homography-calibration
  ```
  
- [ ] **5-10 min:** Generate heatmaps and visualizations
  ```bash
  python world_space_heatmap.py output/tracks_5min_enhanced.csv
  python enhanced_trajectory_viz.py output/tracks_5min_enhanced.csv
  ```

### Day 3: Polish & Validation
- [ ] Verify all outputs (video, CSV, heatmaps)
- [ ] Heatmap zone refinement (if time permits)
- [ ] Pitch markings on trajectory viz (if time permits)
- [ ] Final validation report generation
- [ ] Documentation update

---

## File Modification Checklist

**To Modify:**
- [ ] [track_cv.py](track_cv.py) - Add DeepSORT-tracker case (~5 lines)
- [ ] [tracking_config.yaml](tracking_config.yaml) - Verify feature_path correct
- [ ] [detection_config.yaml](detection_config.yaml) - Keep yolov9c.pt (already set)

**To Download:**
- [ ] `osnet_x1_0_imagenet.pth` (ReID model)
- [ ] `yolov9c.pt` (Detection model)

**To Test:**
- [ ] Run full pipeline with DeepSORT
- [ ] Verify ID switches < 1 per minute
- [ ] Check video output ~300MB
- [ ] Verify CSV has 10k+ rows

---

## Metrics to Hit

| Metric | Target | How to Verify |
|--------|--------|---------------|
| **Detection Confidence** | ≥ 0.60 | Min value in CSV conf column |
| **ID Switches** | < 1 per min | Manual review of video or tracking_id column |
| **Calibration Accuracy** | ±5% | Compare pixel↔meter conversions |
| **Speed Bounds** | < 10 m/s | Max value in speed_mps column |
| **CSV Schema** | 13+ columns | Count columns: frame_index, track_id, x_m, y_m, speed_mps, etc. |
| **Video Framerate** | 25 FPS | Check video properties |

---

## Quick Decision Matrix

```
Do I need to make code changes?
├─ YES: DeepSORT integration (track_cv.py)
├─ YES: Download models (2 files)
├─ NO: Detection config (already correct)
├─ NO: Calibration (already working)
└─ NO: Smoothing (already working)

Will this improve score?
├─ DeepSORT: +2-3 points (ID stability)
├─ YOLOv9: +1-2 points (detection)
├─ Heatmap zones: +0-1 points (polish)
└─ Pitch markings: +0 points (nice-to-have)
```

---

## Summary: What's Left to Complete

### 🔴 **MUST DO** (Blocking)
1. **DeepSORT Integration** - 2-3 hours (scores +2-3)
2. **YOLOv9 Download** - 1 hour (scores +1-2)

### 🟡 **SHOULD DO** (Polish)
3. **Heatmap Zones** - 1-2 hours (scores +0-1)
4. **Pitch Markings** - 1 hour (scores +0)

### ✅ **ALREADY DONE** (Don't touch)
- Detection quality improvements
- Geometric calibration
- Trajectory smoothing
- CSV validation
- All 7 core modules

**Total Time to Complete:** 4-7 hours (critical path: 3 hours)  
**Expected Score Gain:** +3-5 points (from 8.2 → 9.2+)

