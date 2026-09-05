"""Plot the corrected v9 four-architecture comparison without altering data."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ["all_sram", "sram_mram_mram", "sram_osfet_osfet", "optimized"]
LABELS = ["S-S-S (SRAM baseline)", "S-M-M", "S-O-O", "S-A-O"]
COLORS = ["#D2D2D2", "#245882", "#A5D5E8", "#E45D85"]


def load_data():
    data = {}
    for workload in ("attention", "ffn"):
        raw = json.loads((ROOT / f"results/scope_v9_{workload}.json").read_text())
        assert raw["schema_version"] == 9
        cases = {item["case"]: item for item in raw["architecture_comparison"]}
        assert set(cases) == set(CASES)
        for item in cases.values():
            assert item["power_metric"] == "system_average"
            assert math.isclose(item["fom_per_ns_mw"],
                                1 / (item["average_latency_ns"] * item["average_power_mw"]))
            assert all(0 <= rate <= 1 for rate in item["conditional_hit_rates"])
        data[workload] = cases
    return data


def main():
    data = load_data()
    plt.rcParams.update({"font.family": "Arial", "font.size": 18,
                         "axes.labelsize": 21, "axes.titleweight": "bold",
                         "axes.titlesize": 28, "axes.linewidth": 1.8,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                         "svg.fonttype": "none", "svg.hashsalt": "SCOPE-v9"})
    fig, axes = plt.subplots(2, 2, figsize=(18, 9.4))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.11, top=0.83,
                        wspace=0.27, hspace=0.60)

    def format_axis(ax, title, ylabel):
        ax.set_title(title, pad=14)
        ax.set_ylabel(ylabel, labelpad=10, fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DBDFE3", linestyle=":", linewidth=0.8)
        ax.tick_params(axis="both", length=4, width=1.2)

    def metric(ax, key, title, ylabel, scale=1, decimals=2):
        x = np.array([0.0, 1.25])
        width = 0.20
        top = 0
        groups = []
        for i, case in enumerate(CASES):
            values = [data[w][case][key] * scale for w in ("attention", "ffn")]
            top = max(top, *values)
            bars = ax.bar(x + (i - 1.5) * width, values, width * 0.94,
                          color=COLORS[i], edgecolor="#202020", linewidth=1.35)
            groups.append((bars, values))
        ax.set_xticks(x, ["Attention", "FFN"], fontsize=22, fontweight="bold")
        ax.set_ylim(0, top * 1.24)
        ax.set_xlim(-0.62, 1.87)
        format_axis(ax, title, ylabel)
        unit_per_point = top * 1.24 / (ax.get_position().height * fig.get_figheight() * 72)
        previous_y = {}
        for bars, values in groups:
            for index, (bar, value) in enumerate(zip(bars, values)):
                label_y = value + 5 * unit_per_point
                if index in previous_y and abs(label_y - previous_y[index]) < 17 * unit_per_point:
                    label_y = previous_y[index] + 17 * unit_per_point
                previous_y[index] = label_y
                ax.text(bar.get_x() + bar.get_width()/2, label_y,
                        f"{value:.{decimals}f}", ha="center", va="bottom",
                        fontsize=14, fontweight="bold")

    metric(axes[0, 0], "average_latency_ns", "(a) Latency ↓", "Average latency (ns)")
    metric(axes[0, 1], "average_power_mw", "(b) Power ↓", "Power (mW)")
    metric(axes[1, 1], "fom_per_ns_mw", "(d) FoM ↑",
           r"FoM ($10^{-3}$ / (ns·mW))", scale=1000, decimals=3)

    ax = axes[1, 0]
    x = np.array([0, 1, 2, 3.55, 4.55, 5.55])
    width = 0.19
    previous_label_y = {}
    for i, case in enumerate(CASES):
        values = [100 * v for w in ("attention", "ffn")
                  for v in data[w][case]["conditional_hit_rates"]]
        bars = ax.bar(x + (i - 1.5) * width, values, width * 0.94,
                      color=COLORS[i], edgecolor="#202020", linewidth=1.15)
        for k, (bar, value) in enumerate(zip(bars, values)):
            if not k % 3:
                continue
            label_y = value + 3.3
            if k in previous_label_y and abs(label_y - previous_label_y[k]) < 7:
                label_y = previous_label_y[k] + 7
            previous_label_y[k] = label_y
            ax.text(bar.get_x()+bar.get_width()/2, label_y, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=11)
    for position, workload in [(0, "attention"), (3.55, "ffn")]:
        rates = [100 * data[workload][case]["conditional_hit_rates"][0] for case in CASES]
        assert all(math.isclose(value, rates[0]) for value in rates)
        ax.text(position, rates[0] + 2.3, f"{rates[0]:.1f} (all)",
                ha="center", fontsize=13)
    ax.set_xticks(x, ["L1", "L2", "L3"] * 2, fontsize=17, fontweight="bold")
    ax.set_ylim(0, 106)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlim(-0.65, 6.2)
    ax.axvline(2.775, color="#C0C0C0", linewidth=1, linestyle="--")
    for position, text in [(1, "Attention"), (4.55, "FFN")]:
        ax.text(position, -0.18, text, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=21, fontweight="bold")
    format_axis(ax, "(c) Cache hit rate ↑", "Conditional hit rate (%)")

    handles = [Patch(facecolor=color, edgecolor="#202020", label=label)
               for color, label in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.975),
               ncol=4, frameon=False, fontsize=23, handlelength=1.4,
               columnspacing=2.0, prop={"size": 23, "weight": "bold"})
    output = ROOT / "results" / "figures"
    output.mkdir(exist_ok=True, parents=True)
    for extension in ("png", "svg"):
        path = output / f"scope_v9_four_configuration.{extension}"
        fig.savefig(path, dpi=300, facecolor="white",
                    metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
