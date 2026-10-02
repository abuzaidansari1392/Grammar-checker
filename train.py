"""
Standalone training script.
Run: python train.py
"""

# import os, sys
# sys.path.insert(0, os.path.dirname(__file__))

# from nlp.error_classifier import train_classifier
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# Must set backend BEFORE importing pyplot — avoids Tk/display errors
import matplotlib
matplotlib.use("Agg")   # non-interactive file renderer, no display needed

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from nlp.error_classifier import train_classifier

print("Training grammar error classifier…\n")
metrics = train_classifier(verbose=True)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(9, 7))
cm  = metrics["conf_matrix"]
labels = metrics["label_names"]
cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

sns.heatmap(
    cm_pct, annot=True, fmt=".0f", cmap="Blues",
    xticklabels=labels, yticklabels=labels, ax=ax,
    linewidths=0.5, cbar=False,
)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual",    fontsize=11)
ax.set_title("Confusion Matrix (% of actual class)", fontsize=13, pad=12)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

out_path = "models/confusion_matrix.png"
os.makedirs("models", exist_ok=True)
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nConfusion matrix saved to {out_path}")

print(f"\nModel saved to models/error_classifier.joblib")
print("Done! ✓")
