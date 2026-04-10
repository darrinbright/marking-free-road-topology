import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    labels = ['Baseline A\n(Raw Skeleton)', 'Baseline C\n(Static GNN)', 'Proposed System\n(Temporal Graph Memory)']
    precisions = [0.754, 0.725, 0.835]
    recalls = [0.746, 0.717, 0.812]
    f1_scores = [0.750, 0.721, 0.823]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))

    color_p = '#A0C4FF'
    color_r = '#BDB2FF'
    color_f1 = '#4361EE'

    rects1 = ax.bar(x - width, precisions, width, label='Edge Precision', color=color_p, edgecolor='black', linewidth=1.2)
    rects2 = ax.bar(x, recalls, width, label='Edge Recall', color=color_r, edgecolor='black', linewidth=1.2)
    rects3 = ax.bar(x + width, f1_scores, width, label='Edge F1-Score', color=color_f1, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Score', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Topological Extraction Performance (IDD)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.set_ylim(0.65, 0.90)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    out_dir = Path("/home/darrin/Desktop/Automotive2/outputs")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "final_metrics_chart.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {out_path}")

if __name__ == '__main__':
    main()
