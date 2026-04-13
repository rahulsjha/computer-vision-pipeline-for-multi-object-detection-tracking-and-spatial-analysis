import csv, os
from collections import Counter

path = os.path.join('output','tracks_5min.csv')
if not os.path.exists(path):
    raise SystemExit(f"Missing: {path}")

with open(path, 'r', newline='', encoding='utf-8-sig') as f:
    r = csv.DictReader(f)
    fns = r.fieldnames or []
    def pick(cands):
        m = {c.lower(): c for c in fns}
        for c in cands:
            if c in m:
                return m[c]
        return None

    col_id = pick(['cls_id','class_id','class','cls'])
    col_name = pick(['cls_name','class_name','label','name'])

    if not col_id and not col_name:
        print('No class columns found. Columns:', ', '.join(fns))
        raise SystemExit(0)

    cnt = Counter()
    for row in r:
        cid = (row.get(col_id) if col_id else '')
        cname = (row.get(col_name) if col_name else '')
        cid = '' if cid is None else str(cid).strip()
        cname = '' if cname is None else str(cname).strip()
        cnt[(cid, cname)] += 1

print('cls_id\tcls_name\tcount')
for (cid, cname), n in cnt.most_common():
    print(f"{cid}\t{cname}\t{n}")
