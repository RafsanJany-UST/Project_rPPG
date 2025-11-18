import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------------------------------------
# 1. PURE Edge-lag + decision data
# -----------------------------------------------------
data = [
("01-01",0,0.039,"TRIM"), ("01-02",1,0.407,"TRIM"),
("01-03",23,0.067,"UNCERTAIN"), ("01-04",22,0.068,"UNCERTAIN"),
("01-05",10,0.163,"UNCERTAIN"), ("01-06",0,0.273,"TRIM"),

("02-01",14,0.586,"UNCERTAIN"), ("02-02",22,0.000,"UNCERTAIN"),
("02-03",0,0.000,"TRIM"), ("02-04",35,0.000,"UNCERTAIN"),
("02-05",1,0.132,"TRIM"), ("02-06",2,0.691,"TRIM"),

("03-01",0,0.000,"TRIM"), ("03-02",1,0.000,"TRIM"),
("03-03",1,0.000,"TRIM"), ("03-04",0,0.190,"TRIM"),
("03-05",1,0.000,"TRIM"), ("03-06",2,0.000,"TRIM"),

("04-01",13,0.000,"UNCERTAIN"), ("04-02",10,0.000,"UNCERTAIN"),
("04-03",1,0.098,"TRIM"), ("04-04",9,0.000,"UNCERTAIN"),
("04-05",11,0.000,"UNCERTAIN"), ("04-06",11,0.000,"UNCERTAIN"),

("05-01",2,0.187,"TRIM"), ("05-02",1,0.000,"TRIM"),
("05-03",15,0.000,"UNCERTAIN"), ("05-04",62,0.000,"UNCERTAIN"),
("05-05",33,0.352,"UNCERTAIN"), ("05-06",21,0.149,"UNCERTAIN"),

("06-01",15,0.000,"UNCERTAIN"), ("06-03",0,0.000,"TRIM"),
("06-04",36,0.320,"UNCERTAIN"), ("06-05",0,0.236,"TRIM"),
("06-06",1,0.000,"TRIM"),

("07-01",0,0.219,"TRIM"), ("07-02",55,0.000,"UNCERTAIN"),
("07-03",13,0.376,"UNCERTAIN"), ("07-04",0,0.000,"TRIM"),
("07-05",20,0.000,"UNCERTAIN"), ("07-06",7,0.079,"UNCERTAIN"),

("08-01",1,0.000,"TRIM"), ("08-02",1,0.000,"TRIM"),
("08-03",0,0.366,"TRIM"), ("08-04",0,0.000,"TRIM"),
("08-05",0,0.008,"TRIM"), ("08-06",1,0.000,"TRIM"),

("09-01",1,0.001,"TRIM"), ("09-02",20,0.000,"UNCERTAIN"),
("09-03",0,0.000,"TRIM"), ("09-04",11,0.000,"UNCERTAIN"),
("09-05",0,0.000,"TRIM"), ("09-06",0,0.000,"TRIM"),

("10-01",1,0.000,"TRIM"), ("10-02",0,0.000,"TRIM"),
("10-03",22,0.034,"UNCERTAIN"), ("10-04",1,0.248,"TRIM"),
("10-05",1,0.000,"TRIM"), ("10-06",0,0.000,"TRIM"),
]

df = pd.DataFrame(data, columns=["seq_id", "delta_fr", "remain_fr", "decision"])


# -----------------------------------------------------
# 2. Decision colors
# -----------------------------------------------------
colors = df["decision"].map({
    "TRIM": "green",
    "FPS_ADJUST": "blue",   # (none here, but kept for consistency)
    "UNCERTAIN": "red"
})


# -----------------------------------------------------
# 3. Bar plot (delta_fr vs remain_fr)
# -----------------------------------------------------
plt.figure(figsize=(22, 7))

x = range(len(df))
w = 0.35

plt.bar([i - w/2 for i in x], df["delta_fr"],
        width=w, color="gray", label="delta_fr")

plt.bar([i + w/2 for i in x], df["remain_fr"],
        width=w, color="silver", label="remain_fr")

# Decision markers
marker_y = df[["delta_fr", "remain_fr"]].max(axis=1) + 2
plt.scatter(x, marker_y, c=colors, s=90)


# -----------------------------------------------------
# 4. Formatting (same as previous)
# -----------------------------------------------------
plt.xticks(x, df["seq_id"], rotation=90, fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")

plt.xlabel("Sequence ID", fontsize=14, fontweight="bold")
plt.ylabel("Value", fontsize=14, fontweight="bold")
plt.title("Edge Lag + Decision Summary (PURE, CHROM)",
          fontsize=16, fontweight="bold")


# -----------------------------------------------------
# 5. Legend with circular markers
# -----------------------------------------------------
legend_items = [
    Patch(color="gray", label="delta_fr (lag frames)"),
    Patch(color="silver", label="remain_fr (unused frames)"),
    Line2D([0],[0], marker='o', color='green', markersize=10,
           linewidth=0, label="TRIM"),
    Line2D([0],[0], marker='o', color='red', markersize=10,
           linewidth=0, label="UNCERTAIN")
]

legend = plt.legend(handles=legend_items, fontsize=14)
for t in legend.get_texts():
    t.set_fontweight("bold")

plt.tight_layout()
plt.show()
