import json, csv, os, sys
from collections import defaultdict

tracks_path = os.path.join('output','tracks_5min.csv')
roi_path = os.path.join('output','scene_roi.json')

if not os.path.exists(tracks_path):
    print(f"ERROR: Missing {tracks_path}")
    sys.exit(2)
if not os.path.exists(roi_path):
    print(f"ERROR: Missing {roi_path}")
    sys.exit(2)

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def to_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

def point_in_poly_ray(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1)%n]
        # boundary check
        cross = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
        if abs(cross) < 1e-9 and min(x1,x2)-1e-9 <= x <= max(x1,x2)+1e-9 and min(y1,y2)-1e-9 <= y <= max(y1,y2)+1e-9:
            return True
        if (y1 > y) != (y2 > y):
            dy = (y2-y1)
            if abs(dy) < 1e-12:
                continue
            x_int = (x2-x1)*(y-y1)/dy + x1
            if x_int >= x:
                inside = not inside
    return inside

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    union = area_a + area_b - inter
    return inter/union if union > 0 else 0.0

# ROI
with open(roi_path,'r',encoding='utf-8') as f:
    roi_obj = json.load(f)

fps = None
poly = None
if isinstance(roi_obj, dict):
    for k in ['fps','FPS','frame_rate','frameRate']:
        if k in roi_obj:
            try: fps = float(roi_obj[k])
            except Exception: pass
    for k in ['roi','ROI','polygon','poly','points','vertices']:
        if k in roi_obj:
            poly = roi_obj[k]; break
    if poly is None:
        for v in roi_obj.values():
            if isinstance(v, list) and v and isinstance(v[0], (list,tuple,dict)):
                poly = v; break
elif isinstance(roi_obj, list):
    poly = roi_obj

if not poly or len(poly) < 3:
    print('ERROR: Could not find ROI polygon points in scene_roi.json')
    sys.exit(2)

poly2 = []
for p in poly:
    if isinstance(p, dict) and 'x' in p and 'y' in p:
        poly2.append((float(p['x']), float(p['y'])))
    elif isinstance(p, (list,tuple)) and len(p) >= 2:
        poly2.append((float(p[0]), float(p[1])))
poly = poly2

_use_cv2 = False
try:
    import cv2
    import numpy as np
    cnt = np.array(poly, dtype=np.float32).reshape((-1,1,2))
    _use_cv2 = True
except Exception:
    cnt = None

def inside_roi(x, y):
    if _use_cv2:
        return cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0
    return point_in_poly_ray(float(x), float(y), poly)

# CSV
with open(tracks_path, 'r', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = [c.strip() for c in (reader.fieldnames or [])]
    rows = list(reader)

if not rows:
    print('ERROR: tracks CSV has no rows')
    sys.exit(2)

frame_candidates = ['frame','frame_id','frameid','frame_idx','frameindex','frame_index','idx','f']
id_candidates = ['track_id','trackid','id','track']
time_candidates = ['time','t','timestamp','seconds','sec','time_seconds','time_s']

col_frame = next((c for c in fieldnames if c.lower() in frame_candidates), None)
col_id = next((c for c in fieldnames if c.lower() in id_candidates), None)
col_time = next((c for c in fieldnames if c.lower() in time_candidates), None)

bbox_map = {}
for key in fieldnames:
    kl = key.lower()
    if kl in ['x1','left','xmin']: bbox_map['x1'] = key
    elif kl in ['y1','top','ymin']: bbox_map['y1'] = key
    elif kl in ['x2','right','xmax']: bbox_map['x2'] = key
    elif kl in ['y2','bottom','ymax']: bbox_map['y2'] = key

missing = []
if not col_frame: missing.append('frame')
if not col_id: missing.append('track_id')
for k in ['x1','y1','x2','y2']:
    if k not in bbox_map: missing.append(k)
if missing:
    print('ERROR: Missing expected columns:', ', '.join(missing))
    print('Columns found:', fieldnames)
    sys.exit(2)

records = []
for r in rows:
    fr = to_int(r.get(col_frame))
    tid = to_int(r.get(col_id))
    x1 = to_float(r.get(bbox_map['x1']))
    y1 = to_float(r.get(bbox_map['y1']))
    x2 = to_float(r.get(bbox_map['x2']))
    y2 = to_float(r.get(bbox_map['y2']))
    if fr is None or tid is None or None in (x1,y1,x2,y2):
        continue
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    tm = to_float(r.get(col_time)) if col_time else None
    records.append((fr, tid, x1, y1, x2, y2, tm))

if not records:
    print('ERROR: No valid parsed records from CSV')
    sys.exit(2)

frames = [fr for fr, *_ in records]
track_ids = [tid for _, tid, *_ in records]
unique_frames = sorted(set(frames))
unique_tracks = sorted(set(track_ids))

by_frame = defaultdict(list)
by_track = defaultdict(list)
for fr, tid, x1, y1, x2, y2, tm in records:
    by_frame[fr].append((tid, (x1,y1,x2,y2), tm))
    by_track[tid].append(fr)

n_rows = len(records)
n_frames = len(unique_frames)
n_tracks = len(unique_tracks)

max_time = None
if col_time:
    times = [tm for *_, tm in records if tm is not None]
    max_time = max(times) if times else None
elif fps and fps > 0:
    max_time = max(unique_frames)/fps

avg_tracks_per_frame = (n_rows / n_frames) if n_frames else 0.0
max_tracks_in_a_frame = max((len(v) for v in by_frame.values()), default=0)

out_center = 0
out_corner = 0
for fr, tid, x1, y1, x2, y2, tm in records:
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    if not inside_roi(cx, cy):
        out_center += 1
    corners = [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
    if any(not inside_roi(px,py) for px,py in corners):
        out_corner += 1

pct_center_out = 100.0*out_center/n_rows if n_rows else 0.0
pct_corner_out = 100.0*out_corner/n_rows if n_rows else 0.0

track_stats = []
tracks_le_5 = 0
for tid, frs in by_track.items():
    s = sorted(set(frs))
    start, end = s[0], s[-1]
    length = len(s)
    gaps = 0
    max_gap = 0
    for a,b in zip(s, s[1:]):
        d = b-a
        if d > 1:
            gaps += 1
            max_gap = max(max_gap, d-1)
    if length <= 5:
        tracks_le_5 += 1
    track_stats.append((tid, start, end, length, gaps, max_gap))

if track_stats:
    lengths = [t[3] for t in track_stats]
    gaps_list = [t[4] for t in track_stats]
    maxg_list = [t[5] for t in track_stats]
    avg_len = sum(lengths)/len(lengths)
    med_len = sorted(lengths)[len(lengths)//2]
    tracks_with_gaps = sum(1 for g in gaps_list if g>0)
    worst_max_gap = max(maxg_list)
else:
    avg_len = med_len = tracks_with_gaps = worst_max_gap = 0

# Approx ID switches
unique_frames_set = set(unique_frames)
iou_thr = 0.3
switches = 0
total_matches = 0

for f0 in unique_frames:
    f1 = f0 + 1
    if f1 not in unique_frames_set:
        continue
    a = by_frame.get(f0, [])
    b = by_frame.get(f1, [])
    if not a or not b:
        continue
    pairs = []
    for i,(id_a, box_a, _) in enumerate(a):
        for j,(id_b, box_b, _) in enumerate(b):
            v = iou(box_a, box_b)
            if v >= iou_thr:
                pairs.append((v,i,j))
    if not pairs:
        continue
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_i = set(); used_j = set()
    for v,i,j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j)
        total_matches += 1
        if a[i][0] != b[j][0]:
            switches += 1

switch_rate = (switches/total_matches) if total_matches else 0.0

print('\n=== Tracks / ROI Analysis (output/tracks_5min.csv) ===')
print(f"rows\t{n_rows}")
print(f"unique_frames\t{n_frames}")
print(f"duration_s\t{('NA' if max_time is None else f'{max_time:.3f}')}")
print(f"unique_track_ids\t{n_tracks}")
print(f"avg_tracks_per_frame\t{avg_tracks_per_frame:.3f}")
print(f"max_tracks_in_frame\t{max_tracks_in_a_frame}")
print(f"roi_center_out_pct\t{pct_center_out:.2f}")
print(f"roi_any_corner_out_pct\t{pct_corner_out:.2f}")

print('\n=== Tracking stability (per track_id) ===')
print(f"avg_track_length_frames\t{avg_len:.2f}")
print(f"median_track_length_frames\t{med_len}")
print(f"tracks_with_gaps\t{tracks_with_gaps}")
print(f"worst_max_gap_frames\t{worst_max_gap}")
print(f"tracks_len_le_5\t{tracks_le_5}")

print('\n=== Approx ID switch (IoU greedy matching, thr=0.3) ===')
print(f"total_matches\t{total_matches}")
print(f"switches\t{switches}")
print(f"switch_rate\t{switch_rate:.4f}")

track_stats.sort(key=lambda t: (t[4], t[5], t[3]), reverse=True)
print('\nTop 5 gappy tracks: tid start end len gaps max_gap')
for tid, start, end, length, gaps, max_gap in track_stats[:5]:
    print(f"{tid}\t{start}\t{end}\t{length}\t{gaps}\t{max_gap}")
