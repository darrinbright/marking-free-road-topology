# Temporal Topology Extraction on Marking-Free Roads

This codebase extracts road network topologies (junctions and traversable segments) purely from monocular RGB video frames, without HD maps, lane markings, or GPS.

## Key Features & Novelty

* **Passable-Surface as Graph Primitive:** Replaces traditional lane-center lines with the mathematical skeleton of the drivable surface. This enables topology prediction on completely unmarked and unpaved roads.
* **Occlusion-Robust Temporal Graph Memory:** Uses a per-node GRU to maintain tracking state across frames. When large vehicles occlude the road, the system explicitly heals the topology layout by bridging spatial gaps based on projected temporal confidence.
* **IDD-Topo Benchmark:** Introduces the first topology-level annotation set for unstructured roads (based on the Indian Driving Dataset) to evaluate marking-free topology generation.

## Setup and Installation

### Prerequisites
* Python 3.8+
* PyTorch (PyTorch nightly `cu128` recommended for Blackwell GPUs such as RTX 5070)

### Installation
1. Clone the repository and navigate into the project root.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset and Output Structure

* `data/`: Place your IDD segmentation and IDD-Topo datasets here.
* `checkpoints/`: Model weights and training checkpoints are saved here (ignored by git).
* `outputs/`: Output topologies, evaluation results, and graphical plots are saved here.

## Execution

* **Training:** Run the backbone and GNN training pipelines from `scripts/training/`.
* **Evaluation & Visualization:** Benchmark your model on IDD-Topo or visualize tracking continuity using scripts in `scripts/eval/` and `scripts/visualization/`.
