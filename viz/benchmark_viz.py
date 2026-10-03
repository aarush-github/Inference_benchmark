import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

engines = ["vLLM", "SGLang"]
colors = ["#1f77b4", "#ff7f0e"]

# 1. Throughput Comparison (TPS)
ax = axes[0]
throughputs = [379.96, 369.74]
bars = ax.bar(engines, throughputs, color=colors, width=0.55, edgecolor="black", alpha=0.85)
ax.set_title("Request Throughput (TPS)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Requests / Second")
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold")

# 2. Time to First Token (TTFT in ms)
ax = axes[1]
ttft_p50 = [14.49, 15.52]
ttft_p95 = [31.11, 32.36]
x = np.arange(len(engines))
width = 0.35

b1 = ax.bar(x - width/2, ttft_p50, width, label="Median (P50)", color="#4C72B0", edgecolor="black")
b2 = ax.bar(x + width/2, ttft_p95, width, label="P95", color="#C44E52", edgecolor="black")
ax.set_title("Time to First Token (TTFT)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Milliseconds (ms)")
ax.set_xticks(x)
ax.set_xticklabels(engines)
ax.legend(frameon=True)

# 3. End-to-End Latency (in ms)
ax = axes[2]
lat_p50 = [17.53, 18.57]
lat_p95 = [33.91, 35.16]

b3 = ax.bar(x - width/2, lat_p50, width, label="Median (P50)", color="#55A868", edgecolor="black")
b4 = ax.bar(x + width/2, lat_p95, width, label="P95", color="#8172B3", edgecolor="black")
ax.set_title("End-to-End Request Latency", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Milliseconds (ms)")
ax.set_xticks(x)
ax.set_xticklabels(engines)
ax.legend(frameon=True)

plt.suptitle("LLM Inference Benchmark: vLLM vs. SGLang (Qwen3-8B Baseline)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("baseline_comparison.png", bbox_inches="tight")
print("Saved benchmark chart as baseline_comparison.png")