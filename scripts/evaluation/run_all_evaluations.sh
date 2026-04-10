#!/bin/bash
# Run all evaluations: edge sweep, baselines, temporal.
# Activate venv if present.
cd "$(dirname "$0")/.."
[[ -d .venv ]] && source .venv/bin/activate

echo "=== 1. Edge threshold sweep (0.2, 0.3, 0.4, 0.5) ==="
python scripts/evaluate_with_gnn_edges.py --sweep

echo ""
echo "=== 2. Baseline comparison (test set, n=15) ==="
python scripts/evaluate_baselines.py

echo ""
echo "=== 3. Temporal evaluation (sequences, n=140) ==="
python scripts/evaluate_temporal.py

echo ""
echo "Done. Results in outputs/baselines.json"
echo ""
echo "Training pipeline: Phase 1 (backbone) -> Phase 2 (GNN) -> Phase 3 (joint)"
