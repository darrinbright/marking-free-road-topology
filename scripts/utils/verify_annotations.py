import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = ROOT / 'data' / 'annotations'

def main():
    issues = []
    for jp in sorted(ANN_DIR.glob('*.json')):
        try:
            with open(jp) as f:
                ann = json.load(f)
        except Exception as e:
            issues.append((jp.name, str(e)))
            continue
        shapes = ann.get('shapes', [])
        labels = [s.get('label') for s in shapes]
        if 'passable_surface' not in labels:
            issues.append((jp.name, 'Missing passable_surface'))
        for s in shapes:
            pts = s.get('points', [])
            if s.get('shape_type') == 'polygon' and len(pts) < 3:
                issues.append((jp.name, f'Polygon {s.get('label')} has <3 points'))

if __name__ == '__main__':
    main()