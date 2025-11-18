import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# -----------------------------------------------------
# 1. Create DataFrame exactly matching the LaTeX table
# -----------------------------------------------------
data = [
    ("vid_1", 6.000, 4.045, "FPS_ADJUST"),
    ("vid_10", 1.000, 0.000, "TRIM"),
    ("vid_11", 5.000, 0.000, "UNCERTAIN"),
    ("vid_12", 8.000, 2.457, "UNCERTAIN"),
    ("vid_13", 11.000, 0.000, "UNCERTAIN"),
    ("vid_14", 44.000, 0.000, "UNCERTAIN"),
    ("vid_15", 48.000, 0.000, "UNCERTAIN"),
    ("vid_16", 8.000, 0.016, "UNCERTAIN"),
    ("vid_17", 4.000, 2.622, "FPS_ADJUST"),
    ("vid_18", 11.000, 0.000, "UNCERTAIN"),
    ("vid_20", 77.000, 0.000, "UNCERTAIN"),
    ("vid_22", 0.000, 0.000, "TRIM"),
    ("vid_23", 21.000, 0.000, "UNCERTAIN"),
    ("vid_24", 20.000, 0.000, "UNCERTAIN"),
    ("vid_25", 6.000, 0.000, "UNCERTAIN"),
    ("vid_26", 11.000, 0.470, "UNCERTAIN"),
    ("vid_3", 1.000, 0.000, "TRIM"),
    ("vid_30", 26.000, 0.000, "UNCERTAIN"),
    ("vid_31", 39.000, 0.000, "UNCERTAIN"),
    ("vid_32", 3.000, 0.000, "UNCERTAIN"),
    ("vid_33", 23.000, 0.000, "UNCERTAIN"),
    ("vid_34", 1.000, 0.000, "TRIM"),
    ("vid_35", 2.000, 0.601, "TRIM"),
    ("vid_37", 11.000, 0.704, "UNCERTAIN"),
    ("vid_38", 34.000, 0.000, "UNCERTAIN"),
    ("vid_39", 8.000, 0.979, "UNCERTAIN"),
    ("vid_4", 2.000, 0.000, "TRIM"),
    ("vid_40", 0.000, 0.000, "TRIM"),
    ("vid_41", 42.000, 0.000, "UNCERTAIN"),
    ("vid_42", 1.000, 0.000, "TRIM"),
    ("vid_43", 5.000, 0.000, "UNCERTAIN"),
    ("vid_44", 9.000, 1.865, "UNCERTAIN"),
    ("vid_45", 11.000, 0.000, "UNCERTAIN"),
    ("vid_46", 35.000, 0.000, "UNCERTAIN"),
    ("vid_47", 10.000, 0.000, "UNCERTAIN"),
    ("vid_48", 5.000, 0.000, "UNCERTAIN"),
    ("vid_49", 12.000, 0.000, "UNCERTAIN"),
    ("vid_5", 4.000, 0.000, "UNCERTAIN"),
    ("vid_8", 2.000, 0.000, "TRIM"),
    ("vid_9", 35.000, 0.000, "UNCERTAIN")
]

df = pd.DataFrame(data, columns=["seq_id", "delta_fr", "remain_fr", "decision"])


# -----------------------------------------------------
# 2. Decision color mapping
# -----------------------------------------------------
colors = df["decision"].map({
    "TRIM": "green",
    "FPS_ADJUST": "blue",
    "UNCERTAIN": "red"
})


# -----------------------------------------------------
# 3. Plot: delta_fr & remain_fr as grouped bars
# -----------------------------------------------------
plt.figure(figsize=(18, 6))

x = range(len(df))
bar_width = 0.35

plt.bar([i - bar_width/2 for i in x], df["delta_fr"],
        width=bar_width, color="gray", label="delta_fr")

plt.bar([i + bar_width/2 for i in x], df["remain_fr"],
        width=bar_width, color="silver", label="remain_fr")

# Decision markers above bars
marker_y = df[["delta_fr", "remain_fr"]].max(axis=1) + 2
plt.scatter(x, marker_y, c=colors, s=90)


# -----------------------------------------------------
# 4. Formatting
# -----------------------------------------------------
plt.xticks(x, df["seq_id"], rotation=90, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

plt.xlabel("Sequence ID", fontsize=14, fontweight='bold')
plt.ylabel("Value", fontsize=14, fontweight='bold')
plt.title("Edge Lag Difference & Trimming Decision (CHROM)",
          fontsize=14, fontweight='bold')


# -----------------------------------------------------
# 5. Legend with circle markers
# -----------------------------------------------------
legend_handles = [
    Line2D([0], [0], marker='o', color='green', markersize=10,
           linewidth=0, label='TRIM'),
    Line2D([0], [0], marker='o', color='blue', markersize=10,
           linewidth=0, label='FPS_ADJUST'),
    Line2D([0], [0], marker='o', color='red', markersize=10,
           linewidth=0, label='UNCERTAIN'),
    Patch(color="gray", label="delta_fr (drift frames)"),
    Patch(color="silver", label="remain_fr (unused frames)")
]

legend = plt.legend(handles=legend_handles, fontsize=14)
for text in legend.get_texts():
    text.set_fontweight('bold')

plt.tight_layout()
plt.show()
