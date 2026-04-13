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

# (content truncated in this variant; keep original logic if you still need this exploratory script)
print("This helper script has been moved into evaluation/scripts for archival.")
