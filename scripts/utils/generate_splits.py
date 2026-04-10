import random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = ROOT / 'data' / 'annotations'
SPLIT_DIR = ROOT / 'data' / 'splits'

def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    jsons = sorted(ANN_DIR.glob('*.json'))
    random.seed(42)
    random.shuffle(jsons)
    n = len(jsons)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val
    train = jsons[:n_train]
    val = jsons[n_train:n_train + n_val]
    test = jsons[n_train + n_val:]
    for name, paths in [('train', train), ('val', val), ('test', test)]:
        out = SPLIT_DIR / f'{name}.txt'
        out.write_text('\n'.join((str(p.name) for p in paths)))
if __name__ == '__main__':
    main()