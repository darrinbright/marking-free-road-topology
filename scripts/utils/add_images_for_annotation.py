import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = ROOT / 'data' / 'annotations'
IDD_ROOT = ROOT / 'data' / 'idd_raw'
LEFT_IMG = IDD_ROOT / 'leftImg8bit'

def main():
    ap = argparse.ArgumentParser(description='List unannotated IDD images for annotation')
    ap.add_argument('--count', type=int, default=100, help='Max images to suggest')
    ap.add_argument('--prefer-known-seq', action='store_true', help='Prefer sequences we already annotate')
    ap.add_argument('--output', type=str, default=None, help='Write list to file (one path per line)')
    args = ap.parse_args()
    annotated = set()
    for jp in ANN_DIR.glob('*.json'):
        annotated.add(jp.stem)
    candidates = []
    for split in ('val', 'train', 'test'):
        if not (LEFT_IMG / split).exists():
            continue
        for seq_dir in sorted((LEFT_IMG / split).iterdir()):
            if not seq_dir.is_dir():
                continue
            for img in sorted(seq_dir.glob('*_leftImg8bit.png')):
                frame_id = img.stem.replace('_leftImg8bit', '')
                stem = f'{seq_dir.name}_{frame_id}'
                if stem in annotated:
                    continue
                candidates.append((str(img), seq_dir.name, frame_id))
    if not candidates:
        return
    if args.prefer_known_seq:
        known_seqs = {s.split('_', 1)[0] for s in annotated}

        def score(item):
            path, seq, _ = item
            return (0 if seq in known_seqs else 1, path)
        candidates.sort(key=score)
    selected = candidates[:args.count]
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            for path, seq, frame in selected:
                f.write(f'{path}\t{seq}_{frame}\n')
if __name__ == '__main__':
    main()