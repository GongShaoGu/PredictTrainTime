import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

df = pl.read_csv("reslut.csv")
diff = df["diff"].to_list()
max_val = np.max(diff)
min_val = np.min(diff)
mean_val = np.mean(diff)
std_val = np.std(diff)

lengend_labels = [
    f"Max: {max_val:.2f}",
    f"Min: {min_val:.2f}",
    f"Mean: {mean_val:.2f}",
    f"Std: {std_val:.2f}",
]

sns.violinplot(y=diff)
plt.legend(lengend_labels, loc="upper right")
plt.savefig("diff_data.png")
