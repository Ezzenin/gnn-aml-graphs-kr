"""Generate the remaining thesis visuals from saved metrics and local IBM data.

The script does not train models. It only reads existing result JSON/CSV files and
the local IBM AML CSV/pattern file, then writes publication-ready PNG figures to
results/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IBM_TRANS = ROOT / "data" / "ibm_aml" / "HI-Small_Trans.csv"
IBM_PATTERNS = ROOT / "data" / "ibm_aml" / "HI-Small_Patterns.txt"

BLUE = "#2f5f9f"
RED = "#c64f4f"
GREEN = "#3f8f6b"
GOLD = "#d59b2d"
GRAY = "#6f7782"
DARK = "#1f2933"
LIGHT = "#eef2f6"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save(fig: plt.Figure, name: str) -> None:
    path = RESULTS / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path.relative_to(ROOT)}")


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#d8dde5", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def plot_r3_feature_ablation() -> None:
    hybrid = load_json(RESULTS / "ibm_hybrid_gnn_xgb_metrics.json")
    raw = hybrid["comparison"]["xgboost_raw_edge"]
    full = load_json(RESULTS / "ibm_xgboost_notime_metrics.json")["test_metrics"]
    pna = load_json(RESULTS / "ibm_pna_fulldata_metrics.json")["test_metrics"]

    rows = [
        ("raw-edge XGBoost", raw["auc_pr"], raw["f1"]),
        ("full XGBoost\n(raw + node)", full["auc_pr"], full["f1"]),
        ("graph-only\nnode features", 0.067, 0.140),
        ("best standalone GNN\n(PNA)", pna["auc_pr"], pna["f1"]),
    ]
    labels = [r[0] for r in rows]
    auc = np.array([r[1] for r in rows])
    f1 = np.array([r[2] for r in rows])

    y = np.arange(len(rows))
    h = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.barh(y - h / 2, auc, h, color=BLUE, label="AUC-PR")
    ax.barh(y + h / 2, f1, h, color=GOLD, label="F1")
    for yi, v in zip(y - h / 2, auc):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9)
    for yi, v in zip(y + h / 2, f1):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.46)
    ax.set_xlabel("test score")
    ax.set_title("R3. Feature ablation: IBM AML HI-Small")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    save(fig, "r3_feature_ablation_xgb.png")


def plot_r7_hybrid_lift() -> None:
    hybrid = load_json(RESULTS / "ibm_hybrid_gnn_xgb_metrics.json")
    raw = hybrid["comparison"]["xgboost_raw_edge"]["auc_pr"]
    # This base-GINe hybrid number is preserved in docs/findings_summary.md from
    # the earlier hybrid run; the final Kaggle JSON contains the Multi-GNN run.
    base = 0.310
    multi = hybrid["comparison"]["hybrid_embedding_xgb"]["auc_pr"]

    labels = ["raw-edge\nXGBoost", "+ base-GINe\nembedding", "+ Multi-GNN\nembedding"]
    values = np.array([raw, base, multi])
    x = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(x, values, color=BLUE, linewidth=2.8, marker="o", markersize=9)
    ax.fill_between(x, raw, values, color=BLUE, alpha=0.10)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.0045, f"{v:.3f}", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.annotate(
        f"+{base - raw:.3f}",
        xy=(0.5, (raw + base) / 2),
        xytext=(0.5, raw + 0.007),
        ha="center",
        color=GREEN,
        arrowprops=dict(arrowstyle="-", color=GREEN, linewidth=1.4),
    )
    ax.annotate(
        f"+{multi - raw:.3f} vs raw",
        xy=(2, multi),
        xytext=(1.26, multi + 0.013),
        color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.4),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.27, 0.350)
    ax.set_ylabel("test AUC-PR")
    ax.set_title("R7. Hybrid lift grows with embedding structure")
    style_axes(ax)
    save(fig, "r7_hybrid_lift.png")


def edge_degree_score_arrays() -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(IBM_TRANS, usecols=[0, 1, 2, 3, 4, 10], dtype=str)
    src_key = df["From Bank"] + "_" + df["Account"]
    dst_key = df["To Bank"] + "_" + df["Account.1"]
    codes, _ = pd.factorize(pd.concat([src_key, dst_key], ignore_index=True))
    n = len(df)
    src = codes[:n].astype(np.int64)
    dst = codes[n:].astype(np.int64)
    num_nodes = int(codes.max()) + 1
    y = df["Is Laundering"].astype(int).to_numpy()
    ts = (
        pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
        .to_numpy()
        .astype("datetime64[s]")
        .astype(np.int64)
    )

    q_test = np.quantile(ts, 0.8)
    q_val = np.quantile(ts, 0.65)
    train_mask = ts < q_val
    test_mask = ts >= q_test

    out_deg = np.bincount(src[train_mask], minlength=num_nodes).astype(float)
    in_deg = np.bincount(dst[train_mask], minlength=num_nodes).astype(float)
    fan_out = np.log1p(out_deg[src])
    fan_in = np.log1p(in_deg[dst])

    def norm(a: np.ndarray) -> np.ndarray:
        rng = a.max() - a.min()
        return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

    score = norm(fan_out) + norm(fan_in)
    pos = test_mask & (y == 1)
    neg = test_mask & (y == 0)
    return score[pos], score[neg]


def plot_r11_degree_distribution() -> None:
    illicit_score, legit_score = edge_degree_score_arrays()
    rng = np.random.default_rng(42)
    legit_sample = rng.choice(legit_score, size=min(len(legit_score), 120_000), replace=False)

    def ecdf(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.sort(a)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios": [0.9, 1.25]})

    violin = axes[0].violinplot(
        [legit_sample, illicit_score],
        showmeans=False,
        showmedians=True,
        widths=0.78,
    )
    for body, color in zip(violin["bodies"], [GRAY, RED]):
        body.set_facecolor(color)
        body.set_alpha(0.42)
        body.set_edgecolor("none")
    violin["cmedians"].set_color(DARK)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels([f"legit test edges\nn={len(legit_score):,}", f"illicit test edges\nn={len(illicit_score):,}"])
    axes[0].set_ylabel("degree heuristic score")
    axes[0].set_title("Fan-in/fan-out score by edge class")
    axes[0].grid(axis="y", color="#d8dde5", alpha=0.8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    x_l, y_l = ecdf(legit_score)
    x_i, y_i = ecdf(illicit_score)
    axes[1].plot(x_l, y_l, color=GRAY, linewidth=2.2, label="legit edges")
    axes[1].plot(x_i, y_i, color=RED, linewidth=2.4, label="illicit edges")
    med_l = np.median(legit_score)
    med_i = np.median(illicit_score)
    axes[1].axvline(med_l, color=GRAY, linestyle="--", alpha=0.8)
    axes[1].axvline(med_i, color=RED, linestyle="--", alpha=0.8)
    axes[1].text(med_l, 0.08, f"median {med_l:.2f}", color=GRAY, rotation=90, va="bottom")
    axes[1].text(med_i, 0.08, f"median {med_i:.2f}", color=RED, rotation=90, va="bottom")
    axes[1].set_xlabel("degree heuristic score")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title("High-degree score is anti-informative")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(color="#d8dde5", alpha=0.8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[1].text(
        0.04,
        0.92,
        "degree heuristic ROC-AUC = 0.216",
        transform=axes[1].transAxes,
        fontsize=9,
        color=DARK,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#ccd3dc"),
    )
    fig.suptitle("R11. IBM AML degree diagnostics: mules, not hubs", y=1.03, fontsize=13)
    save(fig, "r11_degree_distribution_illicit_vs_legit.png")


def plot_r12_roc_vs_pr() -> None:
    rows = []
    with (RESULTS / "ibm_all_variants.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["family"].startswith("GNN"):
                continue
            if not row["roc_auc"] or not row["auc_pr"]:
                continue
            rows.append(row)

    auc = np.array([float(r["auc_pr"]) for r in rows])
    roc = np.array([float(r["roc_auc"]) for r in rows])
    f1 = np.array([float(r["f1"]) for r in rows])
    labels = [r["variant"] for r in rows]
    regimes = [r["regime"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    sizes = 180 + 1500 * f1
    colors = [RED if "PNA" in l else BLUE if "full-data" in r else GRAY for l, r in zip(labels, regimes)]
    ax.scatter(roc, auc, s=sizes, c=colors, alpha=0.76, edgecolor="white", linewidth=1.0)
    for r, x, y in zip(rows, roc, auc):
        key = (
            r["variant"] == "PNA"
            or (r["variant"] == "Multi-GNN" and r["regime"] == "full-data/no-time")
            or (r["variant"] == "GINe" and r["regime"] == "full-data/no-time")
            or r["variant"] == "Multi-GNN + EU big-nbr"
        )
        if key:
            ax.text(x + 0.002, y + 0.0015, r["variant"], fontsize=8.5)
    ax.axhspan(0, 0.06, color=RED, alpha=0.08)
    ax.annotate(
        "High ROC-AUC does not imply\na useful top-risk PR ranking",
        xy=(0.9639, 0.0591),
        xytext=(0.875, 0.066),
        arrowprops=dict(arrowstyle="->", color=DARK, linewidth=1.2),
        fontsize=9,
    )
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("AUC-PR")
    ax.set_xlim(0.78, 0.975)
    ax.set_ylim(0.0, 0.072)
    ax.set_title("R12. Standalone GNN: good ranking, weak precision-recall")
    ax.grid(color="#d8dde5", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "r12_gnn_roc_auc_vs_auc_pr.png")


def add_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, text: str, fc: str, ec: str = DARK) -> None:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = DARK, rad: float = 0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def plot_r14_representations_schema() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("Account graph: node = account, edge = transaction", fontsize=11)
    pos = {"A": (0.18, 0.62), "B": (0.48, 0.78), "C": (0.78, 0.58), "D": (0.52, 0.28)}
    edges = [("A", "B", "t1"), ("B", "C", "t2"), ("A", "D", "t3"), ("D", "C", "t4")]
    for u, v, t in edges:
        add_arrow(ax, pos[u], pos[v], BLUE)
        mid = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
        ax.text(mid[0], mid[1] + 0.035, t, color=BLUE, fontsize=9, ha="center")
    for name, (x, y) in pos.items():
        circ = plt.Circle((x, y), 0.055, color=LIGHT, ec=DARK, lw=1.4)
        ax.add_patch(circ)
        ax.text(x, y, name, ha="center", va="center", weight="bold")
    ax.text(0.5, 0.08, "GNN predicts labels on transaction edges", ha="center", fontsize=9, color=GRAY)

    ax = axes[1]
    ax.set_title("Line graph: node = transaction", fontsize=11)
    pos2 = {"t1": (0.18, 0.55), "t2": (0.43, 0.76), "t3": (0.43, 0.33), "t4": (0.73, 0.55)}
    adj = [("t1", "t2", "shared B"), ("t1", "t3", "shared A"), ("t3", "t4", "shared D"), ("t2", "t4", "shared C")]
    for u, v, label in adj:
        add_arrow(ax, pos2[u], pos2[v], GREEN, rad=0.05)
        mid = ((pos2[u][0] + pos2[v][0]) / 2, (pos2[u][1] + pos2[v][1]) / 2)
        ax.text(mid[0], mid[1], label, fontsize=7.6, color=GREEN, ha="center", va="center")
    for name, (x, y) in pos2.items():
        circ = plt.Circle((x, y), 0.062, color="#fff5df", ec=GOLD, lw=1.5)
        ax.add_patch(circ)
        ax.text(x, y, name, ha="center", va="center", weight="bold")
    ax.text(0.5, 0.08, "Alternative representation; left one is implemented", ha="center", fontsize=9, color=GRAY)

    fig.suptitle("R14. Two graph representations for AML transactions", y=1.02, fontsize=13)
    save(fig, "r14_graph_representations_schema.png")


def plot_r15_hybrid_dataflow() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ((0.03, 0.58), 0.16, 0.22, "Temporal\ntransaction graph", LIGHT),
        ((0.25, 0.58), 0.16, 0.22, "GNN message\npassing", "#e8f2ff"),
        ((0.47, 0.58), 0.16, 0.22, "Edge embedding\n[h_u || h_v || e]", "#eaf7ef"),
        ((0.47, 0.19), 0.16, 0.18, "Raw edge\nattributes", "#fff5df"),
        ((0.69, 0.49), 0.15, 0.22, "XGBoost\nclassifier", "#f5e9e9"),
        ((0.89, 0.49), 0.09, 0.22, "Risk\nscore", "#f1f3f5"),
    ]
    for xy, w, h, text, fc in boxes:
        add_box(ax, xy, w, h, text, fc)
    add_arrow(ax, (0.19, 0.69), (0.25, 0.69), BLUE)
    add_arrow(ax, (0.41, 0.69), (0.47, 0.69), BLUE)
    add_arrow(ax, (0.63, 0.69), (0.69, 0.62), GREEN)
    add_arrow(ax, (0.63, 0.28), (0.69, 0.55), GOLD)
    add_arrow(ax, (0.84, 0.60), (0.89, 0.60), RED)
    ax.text(0.36, 0.86, "context for test embeddings: train + val edges only", ha="center", fontsize=8.5, color=GRAY)
    ax.text(0.76, 0.36, "threshold selected on validation", ha="center", fontsize=8.5, color=GRAY)
    ax.set_title("R15. Hybrid GNN -> XGBoost data flow", fontsize=13)
    save(fig, "r15_hybrid_gnn_xgboost_dataflow.png")


def parse_first_pattern(pattern_name: str = "CYCLE") -> list[list[str]]:
    rows: list[list[str]] = []
    inside = False
    with IBM_PATTERNS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                inside = pattern_name in line
                rows = [] if inside else rows
            elif line.startswith("END LAUNDERING ATTEMPT"):
                if inside and rows:
                    return rows
                inside = False
            elif inside and line:
                rows.append(line.split(","))
    return rows


def short_account(bank: str, account: str) -> str:
    return f"{bank}\\n{account[-4:]}"


def plot_r13_chain_highlight() -> None:
    rows = parse_first_pattern("CYCLE")
    nodes = []
    edges = []
    for r in rows:
        src = (r[1], r[2])
        dst = (r[3], r[4])
        if src not in nodes:
            nodes.append(src)
        if dst not in nodes:
            nodes.append(dst)
        edges.append((src, dst, float(r[7]), r[8]))
    n = len(nodes)
    theta = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    pos = {node: (0.5 + 0.34 * np.cos(t), 0.52 + 0.34 * np.sin(t)) for node, t in zip(nodes, theta)}

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for src, dst, amount, cur in edges:
        add_arrow(ax, pos[src], pos[dst], RED, rad=0.05)
    for node, (x, y) in pos.items():
        circ = plt.Circle((x, y), 0.052, color="#fff5f5", ec=RED, lw=1.4)
        ax.add_patch(circ)
        ax.text(x, y, short_account(*node), ha="center", va="center", fontsize=7.2)
    ax.text(0.5, 0.09, f"Real IBM AML pattern block: CYCLE, {len(edges)} laundering transactions", ha="center", fontsize=10)
    ax.set_title("R13. Highlighted suspicious transaction chain", fontsize=13)
    save(fig, "r13_suspicious_chain_highlight.png")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    plot_r3_feature_ablation()
    plot_r7_hybrid_lift()
    plot_r11_degree_distribution()
    plot_r12_roc_vs_pr()
    plot_r13_chain_highlight()
    plot_r14_representations_schema()
    plot_r15_hybrid_dataflow()


if __name__ == "__main__":
    main()
