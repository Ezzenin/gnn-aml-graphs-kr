"""Сводное сравнение моделей на Elliptic: таблица + bar-chart по results/.

Использование:
    python -m src.compare --run     # прогнать все конфиги, затем собрать сводку
    python -m src.compare           # только собрать сводку из готовых results/*.json

Собирает test-метрики из results/*_metrics.json в таблицу (CSV + Markdown) и
строит сравнительный график AUC-PR / F1. Главные метрики для несбалансированной
задачи — AUC-PR и F1 по позитивному классу.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# Конфиги бейзлайнов и GNN (порядок = порядок прогона).
BASELINE_CONFIGS = [
    "configs/elliptic_xgb.yaml",
    "configs/elliptic_logreg.yaml",
]
GNN_CONFIGS = [
    "configs/elliptic_gcn.yaml",
    "configs/elliptic_sage.yaml",
    "configs/elliptic_gat.yaml",
    "configs/elliptic_gin.yaml",
    "configs/elliptic_pna.yaml",
]

METRIC_KEYS = ["auc_pr", "f1", "recall_at_precision_90", "roc_auc", "precision", "recall"]

# ── IBM AML edge-classification: бейзлайн + ablation-сетка Multi-GNN (Фаза D) ──
IBM_XGB_CONFIG = "configs/ibm_xgb.yaml"
IBM_GNN_CONFIGS = [
    "configs/ibm_gine.yaml",        # base (все флаги выкл)
    "configs/ibm_gine_rev.yaml",    # + reverse MP
    "configs/ibm_gine_port.yaml",   # + port numbering
    "configs/ibm_gine_ego.yaml",    # + ego-IDs
    "configs/ibm_multignn.yaml",    # full (reverse + port + ego)
]
# (output_name, человекочитаемый ярлык) — порядок строк сводки/графика.
# Два режима признаков (P1.6): с norm_time и без (time при temporal split вредит).
IBM_VARIANTS = [
    ("ibm_xgboost", "XGBoost"),
    ("ibm_gine", "GINe (base)"),
    ("ibm_gine_rev", "+reverse"),
    ("ibm_gine_port", "+port"),
    ("ibm_gine_ego", "+ego"),
    ("ibm_multignn", "Multi-GNN (full)"),
]
IBM_VARIANTS_NOTIME = [
    ("ibm_xgboost_notime", "XGBoost"),
    ("ibm_gine_notime", "GINe (base)"),
    ("ibm_gine_rev_notime", "+reverse"),
    ("ibm_gine_port_notime", "+port"),
    ("ibm_gine_ego_notime", "+ego"),
    ("ibm_multignn_notime", "Multi-GNN (full)"),
]
IBM_VARIANTS_FULLDATA = [
    ("ibm_xgboost_notime", "XGBoost"),
    ("ibm_xgboost_fan", "XGBoost+fan"),
    ("ibm_hybrid_gnn_xgb", "Hybrid GNN→XGBoost"),  # лучшая модель: граф-эмбеддинг + дерево
    ("ibm_gine_fulldata", "GINe (base)"),
    ("ibm_gine_rev_fulldata", "+reverse"),
    ("ibm_gine_port_fulldata", "+port"),
    ("ibm_gine_ego_fulldata", "+ego"),
    ("ibm_multignn_fulldata", "Multi-GNN (full)"),
]
# GIN+EU (edge updates) поверх full-data/no-time — самый дешёвый буст по Egressy
# (+19пп). Показывает вклад edge-updates на base и на полном Multi-GNN.
IBM_VARIANTS_EU = [
    ("ibm_xgboost_notime", "XGBoost"),
    ("ibm_gine_fulldata", "GINe (base)"),
    ("ibm_gine_eu_fulldata", "GINe+EU"),
    ("ibm_multignn_fulldata", "Multi-GNN"),
    ("ibm_multignn_eu_fulldata", "Multi-GNN+EU"),
]
# Лестница перехода «слабый → сильный режим» (главный нарратив работы): по одному
# рычагу за шаг при ФИКСИРОВАННОМ протоколе. L0–L2 — слабый режим ([10,10]/2 слоя),
# L3 — те же адаптации при больших окрестностях/глубине, L4–L6 — смена агрегатора
# на PNA (Egressy 2024) вплоть до целевого Multi-PNA+EU. XGBoost(+fan) — референс.
IBM_VARIANTS_LADDER = [
    ("ibm_xgboost_notime", "XGBoost"),
    ("ibm_xgboost_fan", "XGBoost+fan"),
    ("ibm_gine_fulldata", "L0 GINe"),
    ("ibm_multignn_fulldata", "L1 Multi-GNN"),
    ("ibm_multignn_eu_fulldata", "L2 +EU"),
    ("ibm_multignn_big_fulldata", "L3 big-nbr"),
    ("ibm_pna_fulldata", "L4 PNA"),
    ("ibm_multipna_fulldata", "L5 Multi-PNA"),
    ("ibm_multipna_eu_fulldata", "L6 Multi-PNA+EU"),
]
# Табличные референсы лестницы — рисуются горизонтальными линиями, не шагами.
LADDER_REFS = {"XGBoost", "XGBoost+fan"}

IBM_METRIC_KEYS = ["auc_pr", "f1", "recall_at_precision_90", "recall"]
IBM_ALL_METRIC_KEYS = ["auc_pr", "f1", "roc_auc", "recall_at_precision_90", "precision", "recall"]

# Единый реестр всех IBM-вариантов, которые уже встречаются в results/.
# Режим нужен для группировки в общей таблице и графиках.
IBM_ALL_VARIANTS = [
    ("ibm_xgboost", "XGBoost", "Tabular", "with norm_time"),
    ("ibm_xgboost_notime", "XGBoost no-time", "Tabular", "no-time"),
    ("ibm_xgboost_fan", "XGBoost+fan", "Tabular", "fan features"),
    ("hybrid_xgb_raw_edge", "XGBoost raw-edge", "Hybrid test", "raw edge attrs"),
    ("hybrid_gine_emb_xgb", "GINe emb -> XGBoost", "Hybrid", "GNN embedding"),
    ("ibm_hybrid_gnn_xgb", "Multi-GNN emb -> XGBoost", "Hybrid", "GNN embedding"),
    ("ibm_gine", "GINe", "GNN: GINe", "with norm_time"),
    ("ibm_gine_rev", "GINe + reverse", "GNN: GINe", "with norm_time"),
    ("ibm_gine_port", "GINe + port", "GNN: GINe", "with norm_time"),
    ("ibm_gine_ego", "GINe + ego", "GNN: GINe", "with norm_time"),
    ("ibm_multignn", "Multi-GNN", "GNN: GINe", "with norm_time"),
    ("ibm_gine_notime", "GINe", "GNN: GINe", "no-time"),
    ("ibm_gine_rev_notime", "GINe + reverse", "GNN: GINe", "no-time"),
    ("ibm_gine_port_notime", "GINe + port", "GNN: GINe", "no-time"),
    ("ibm_gine_ego_notime", "GINe + ego", "GNN: GINe", "no-time"),
    ("ibm_multignn_notime", "Multi-GNN", "GNN: GINe", "no-time"),
    ("ibm_gine_fulldata", "GINe", "GNN: GINe", "full-data/no-time"),
    ("ibm_gine_rev_fulldata", "GINe + reverse", "GNN: GINe", "full-data/no-time"),
    ("ibm_gine_port_fulldata", "GINe + port", "GNN: GINe", "full-data/no-time"),
    ("ibm_gine_ego_fulldata", "GINe + ego", "GNN: GINe", "full-data/no-time"),
    ("ibm_multignn_fulldata", "Multi-GNN", "GNN: GINe", "full-data/no-time"),
    ("ibm_gine_eu_fulldata", "GINe + EU", "GNN: GINe+EU", "full-data/no-time"),
    ("ibm_multignn_eu_fulldata", "Multi-GNN + EU", "GNN: GINe+EU", "full-data/no-time"),
    ("ibm_multignn_big_fulldata", "Multi-GNN + EU big-nbr", "GNN: GINe+EU", "big/full-data"),
    ("ibm_pna_fulldata", "PNA", "GNN: PNA", "big/full-data"),
    ("ibm_multipna_fulldata", "Multi-PNA", "GNN: PNA", "big/full-data"),
    ("ibm_multipna_eu_fulldata", "Multi-PNA + EU", "GNN: PNA", "big/full-data"),
    ("ibm_heuristics", "Degree heuristics", "Heuristics", "no-time"),
]

IBM_ABLATION_HEATMAP = [
    ("GINe", ["ibm_gine", "ibm_gine_notime", "ibm_gine_fulldata"]),
    ("+reverse", ["ibm_gine_rev", "ibm_gine_rev_notime", "ibm_gine_rev_fulldata"]),
    ("+port", ["ibm_gine_port", "ibm_gine_port_notime", "ibm_gine_port_fulldata"]),
    ("+ego", ["ibm_gine_ego", "ibm_gine_ego_notime", "ibm_gine_ego_fulldata"]),
    ("Multi-GNN", ["ibm_multignn", "ibm_multignn_notime", "ibm_multignn_fulldata"]),
]
IBM_ABLATION_HEATMAP_COLS = ["with time", "no-time", "full-data"]
# Опубликованные F1-minority (%) на AML Small HI — для справки, НЕ сравнивать в одну
# колонку с нашими (другой сплит 60/20/20, обучение на всех рёбрах). См. docs/lit_benchmarks.md.
LITERATURE_F1_HI = [
    ("XGBoost+GF (Altman 2023)", 63.23, "+ подграфовые fan/cycle-фичи (GFP)"),
    ("LightGBM+GF (Altman 2023)", 62.86, ""),
    ("GIN base (Egressy 2024)", 28.70, "2 слоя, все train-рёбра, class weights"),
    ("GIN+Ports", 54.85, "port numbering"),
    ("GIN+ReverseMP", 46.79, "reverse MP — у нас не переносится при [10,10]"),
    ("Multi-GIN (rev+port+ego)", 57.15, "ego поверх rev+port почти не добавляет"),
    ("Multi-PNA+EU (SOTA)", 68.16, "единственный обошёл GBT+GF на всех AML"),
    ("XGBoost без GF (Blanuša 2024)", 24.50, "≈ наш XGBoost 19.0 по порядку"),
]
BASE_LABEL = "GINe (base)"
GNN_ORDER = [BASE_LABEL, "+reverse", "+port", "+ego", "Multi-GNN (full)"]

# Per-pattern (RQ3): три семейства + эвристики (4-е, если посчитаны). 8 паттернов.
CANONICAL_PATTERNS = [
    "fan_out", "fan_in", "gather_scatter", "scatter_gather",
    "cycle", "random", "bipartite", "stack",
]
# Per-pattern собирается в финальном (лучшем) режиме full-data/no-time: лучшая
# версия каждого семейства, согласовано с финальным нарративом (XGBoost no-time —
# выбранный baseline). Режим явно указан в заголовке таблицы.
PER_PATTERN_FAMILIES = [
    ("ibm_xgboost_notime", "XGBoost"),
    ("ibm_xgboost_fan", "XGBoost+fan"),
    ("ibm_gine_fulldata", "GINe (base)"),
    ("ibm_multignn_fulldata", "Multi-GNN"),
    ("ibm_multipna_eu_fulldata", "Multi-PNA+EU"),  # целевой L6 сильного режима
    ("ibm_heuristics", "Эвристики"),
]


def run_all() -> None:
    """Прогнать все бейзлайны и GNN по их конфигам."""
    from src.train import run as run_gnn
    from src.train_baseline import run as run_baseline
    from src.utils import load_config

    for cfg_path in BASELINE_CONFIGS:
        print(f"\n===== RUN baseline: {cfg_path} =====")
        run_baseline(load_config(cfg_path))
    for cfg_path in GNN_CONFIGS:
        print(f"\n===== RUN gnn: {cfg_path} =====")
        run_gnn(load_config(cfg_path))


def collect(results_dir: str = "results") -> list[dict]:
    """Собрать test-метрики ТОЛЬКО Elliptic (elliptic_*_metrics.json).

    P1.4: после появления IBM-результатов нельзя грести все *_metrics.json в
    одну сводку — Elliptic и IBM это разные задачи (node- vs edge-classification).
    IBM собирается отдельным collect_ibm().
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "elliptic_*_metrics.json"))):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        test = d.get("test_metrics", {})
        model = d.get("model_type") or os.path.basename(path).replace("_metrics.json", "")
        row = {"model": model, "file": os.path.basename(path)}
        row.update({k: test.get(k) for k in METRIC_KEYS})
        rows.append(row)
    # Сортировка по AUC-PR убыванию (главная метрика).
    rows.sort(key=lambda r: (r.get("auc_pr") is not None, r.get("auc_pr") or 0), reverse=True)
    return rows


def write_table(rows: list[dict], results_dir: str = "results") -> None:
    """Записать сводку в CSV и Markdown, напечатать в консоль."""
    import csv

    csv_path = os.path.join(results_dir, "comparison.csv")
    md_path = os.path.join(results_dir, "comparison.md")
    cols = ["model"] + METRIC_KEYS

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join([str(r["model"])] + [fmt(r.get(k)) for k in METRIC_KEYS]) + " |")
    md = "# Сравнение моделей на Elliptic (test)\n\nГлавные метрики: AUC-PR и F1 (позитив = illicit).\n\n" + "\n".join(lines) + "\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "\n".join(lines))
    print(f"\n[saved] {csv_path}\n[saved] {md_path}")


def plot(rows: list[dict], results_dir: str = "results") -> None:
    """Сравнительный bar-chart по AUC-PR и F1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = [r["model"] for r in rows]
    auc_pr = [r.get("auc_pr") or 0 for r in rows]
    f1 = [r.get("f1") or 0 for r in rows]

    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.1), 4.5))
    ax.bar(x - width / 2, auc_pr, width, label="AUC-PR", color="#4c72b0")
    ax.bar(x + width / 2, f1, width, label="F1", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Elliptic (test): сравнение моделей")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(results_dir, "comparison_models.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


# ─────────────────────────── IBM AML (Фаза D) ───────────────────────────
def run_ibm() -> None:
    """Прогнать IBM-сетку: XGBoost-бейзлайн (если нет) + 5 edge-GNN вариантов.

    Тяжёлая часть (CUDA) — выполняется на ПК. XGBoost считается один раз и
    переиспользуется; GNN-варианты — это ablation (base + 3 одиночные + full).
    """
    from src.train_baseline import run as run_baseline
    from src.train_edge import run as run_edge
    from src.utils import load_config

    if not os.path.exists("results/ibm_xgboost_metrics.json"):
        print(f"\n===== RUN ibm baseline: {IBM_XGB_CONFIG} =====")
        run_baseline(load_config(IBM_XGB_CONFIG))
    for cfg_path in IBM_GNN_CONFIGS:
        print(f"\n===== RUN ibm edge-GNN: {cfg_path} =====")
        run_edge(load_config(cfg_path))


def collect_ibm(results_dir: str = "results", variants=IBM_VARIANTS) -> list[dict]:
    """Собрать test-метрики IBM-вариантов в заданном порядке (пропуская отсутствующие)."""
    rows = []
    for name, label in variants:
        path = os.path.join(results_dir, f"{name}_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        test = d.get("test_metrics", {})
        row = {"variant": label, "name": name}
        row.update({k: test.get(k) for k in IBM_METRIC_KEYS})
        rows.append(row)
    return rows


def _metrics_from_json(path: str) -> dict:
    """Прочитать JSON результата и вернуть test_metrics (если есть)."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("test_metrics", {}) or {}


def _hybrid_embedded_rows(results_dir: str = "results") -> dict[str, dict]:
    """Строки из comparison-блока гибридного эксперимента.

    В `ibm_hybrid_gnn_xgb_metrics.json` несколько сравнений лежат внутри одного
    файла: raw-edge XGBoost, GINe-embedding hybrid и Multi-GNN-embedding hybrid.
    Для общей таблицы представляем их как отдельные строки.
    """
    path = os.path.join(results_dir, "ibm_hybrid_gnn_xgb_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        comparison = json.load(f).get("comparison", {}) or {}
    return {
        "hybrid_xgb_raw_edge": comparison.get("xgboost_raw_edge", {}),
        "hybrid_gine_emb_xgb": comparison.get("hybrid_gine_emb_xgb", {}),
        "ibm_hybrid_gnn_xgb": (
            comparison.get("hybrid_multignn_emb_xgb", {})
            or comparison.get("hybrid_embedding_xgb", {})
        ),
    }


def collect_ibm_all(results_dir: str = "results") -> list[dict]:
    """Единая таблица всех IBM-вариантов: режимы, сильная лестница и гибриды."""
    hybrid_rows = _hybrid_embedded_rows(results_dir)
    rows = []
    for name, label, family, regime in IBM_ALL_VARIANTS:
        path = os.path.join(results_dir, f"{name}_metrics.json")
        if name in hybrid_rows and hybrid_rows[name]:
            test = hybrid_rows[name]
            source = "ibm_hybrid_gnn_xgb_metrics.json"
        elif os.path.exists(path):
            test = _metrics_from_json(path)
            source = os.path.basename(path)
        else:
            continue
        row = {
            "name": name,
            "variant": label,
            "family": family,
            "regime": regime,
            "source": source,
        }
        row.update({k: test.get(k) for k in IBM_ALL_METRIC_KEYS})
        rows.append(row)
    return rows


def _best_by(rows: list[dict], family: str, metric: str = "auc_pr") -> dict | None:
    vals = [r for r in rows if r["family"] == family and isinstance(r.get(metric), (int, float))]
    if not vals:
        return None
    return max(vals, key=lambda r: r[metric])


def write_ibm_all_table(rows: list[dict], results_dir: str = "results") -> None:
    """Записать общий рейтинг всех IBM-экспериментов в CSV и Markdown."""
    import csv

    if not rows:
        return

    csv_path = os.path.join(results_dir, "ibm_all_variants.csv")
    md_path = os.path.join(results_dir, "ibm_all_variants.md")
    cols = ["family", "regime", "variant"] + IBM_ALL_METRIC_KEYS + ["source"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ranked = sorted(
        rows,
        key=lambda r: (isinstance(r.get("auc_pr"), (int, float)), r.get("auc_pr") or 0),
        reverse=True,
    )

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    header = "| " + " | ".join(cols[:-1]) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) - 1)) + "|"
    lines = [header, sep]
    for r in ranked:
        lines.append("| " + " | ".join(
            [str(r[c]) if c in ("family", "regime", "variant") else fmt(r.get(c)) for c in cols[:-1]]
        ) + " |")

    best_overall = ranked[0]
    best_gnn = max(
        [r for r in rows if r["family"].startswith("GNN") and isinstance(r.get("auc_pr"), (int, float))],
        key=lambda r: r["auc_pr"],
    )
    best_tab = _best_by(rows, "Tabular")
    best_hybrid = _best_by(rows, "Hybrid")
    intro = [
        "# IBM AML HI-Small: все рассмотренные варианты (test)",
        "",
        "Единая таблица для отчета: табличные бейзлайны, все режимы GINe/Multi-GNN,",
        "edge-updates, сильная PNA-лестница, эвристики и гибрид GNN→XGBoost.",
        "Главная сортировка ниже — по AUC-PR на test.",
        "",
        "## Короткий вывод",
        "",
        f"- Лучший общий вариант: **{best_overall['variant']}** "
        f"({best_overall['family']}, AUC-PR {best_overall['auc_pr']:.3f}, F1 {fmt(best_overall.get('f1'))}).",
        f"- Лучший standalone GNN: **{best_gnn['variant']}** "
        f"({best_gnn['regime']}, AUC-PR {best_gnn['auc_pr']:.3f}, F1 {fmt(best_gnn.get('f1'))}).",
    ]
    if best_tab is not None:
        intro.append(f"- Лучший табличный baseline из отдельных прогонов: **{best_tab['variant']}** "
                     f"(AUC-PR {best_tab['auc_pr']:.3f}).")
    if best_hybrid is not None:
        intro.append(f"- Лучший гибрид: **{best_hybrid['variant']}** "
                     f"(AUC-PR {best_hybrid['auc_pr']:.3f}); это показывает, что графовый эмбеддинг "
                     "полезен как представление, хотя standalone GNN слабее XGBoost.")
    intro += [
        "",
        "## Рейтинг",
        "",
        *lines,
        "",
        "Примечание: строки `Multi-PNA` и `Multi-PNA + EU` восстановлены из run log,",
        "поэтому для них отсутствуют `precision`, `recall`, `ROC-AUC` и threshold.",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(intro))

    print(f"\n[saved] {csv_path}\n[saved] {md_path}")


def write_ibm_table(rows: list[dict], results_dir: str = "results",
                    name: str = "ibm_comparison", regime: str = "",
                    title: str = "бейзлайн + ablation Multi-GNN",
                    intro: str | None = None) -> None:
    """Сводная таблица IBM (CSV + Markdown): XGBoost vs base GNN vs +адаптации.

    title/intro переопределяются для лестницы перехода (иной нарратив, чем ablation).
    """
    import csv

    cols = ["variant"] + IBM_METRIC_KEYS
    csv_path = os.path.join(results_dir, f"{name}.csv")
    md_path = os.path.join(results_dir, f"{name}.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join([str(r["variant"])] + [fmt(r.get(k)) for k in IBM_METRIC_KEYS]) + " |")
    default_intro = ("Главные метрики: AUC-PR и F1 (позитив = laundering). Ablation: вклад\n"
                     "адаптаций reverse / port / ego поверх базовой GINe (RQ2).")
    md = (f"# IBM AML HI-Small (test): {title}{regime}\n\n"
          + (intro if intro is not None else default_intro) + "\n\n"
          + "\n".join(lines) + "\n"
          + _literature_block())
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "\n".join(lines))
    print(f"\n[saved] {csv_path}\n[saved] {md_path}")


def _literature_block() -> str:
    """Справочный блок опубликованных F1 (НЕ в одну колонку с нашими — другой режим)."""
    lines = ["", "## Reference results (literature) — F1-minority, %",
             "",
             "> Внимание: другой сплит (60/20/20) и режим обучения (все train-рёбра, class",
             "> weights). НЕ сравнивать напрямую с нашими AUC-PR/F1 выше; приведено для",
             "> ориентира масштаба и анализа расхождений (см. docs/lit_benchmarks.md).",
             "", "| Модель (источник) | F1-minority % | примечание |",
             "|---|---|---|"]
    for name, f1, note in LITERATURE_F1_HI:
        lines.append(f"| {name} | {f1:.1f} | {note} |")
    lines += ["",
              "Наш режим ослаблен (сабсэмпл негативов, окрестности [10,10], XGBoost без",
              "подграфовых GF-фич), поэтому абсолютные числа ниже. Устойчивого переноса",
              "эффекта адаптаций (reverse/port/ego) не наблюдается ни в одном из трёх",
              "режимов (с временем / без времени / full-data): знак и порядок дельт к base",
              "меняются между режимами, full-data не воспроизводит направление Egressy.",
              "См. docs/lit_benchmarks.md §1.2.", ""]
    return "\n".join(lines)


def plot_ablation(rows: list[dict], results_dir: str = "results",
                  out_name: str = "ablation", regime: str = "") -> None:
    """Bar-chart ablation: F1 и AUC-PR по GNN-вариантам + пунктир уровня base.

    Показывает вклад каждой адаптации относительно базовой GINe (главный график
    отчёта по RQ2). XGBoost — как горизонтальная референс-линия, если есть.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by = {r["variant"]: r for r in rows}
    variants = [v for v in GNN_ORDER if v in by]
    if BASE_LABEL not in variants:
        print("[ablation] нет базовой GINe — график пропущен")
        return

    f1 = [by[v].get("f1") or 0 for v in variants]
    auc = [by[v].get("auc_pr") or 0 for v in variants]
    base_f1 = by[BASE_LABEL].get("f1") or 0
    base_auc = by[BASE_LABEL].get("auc_pr") or 0

    x = np.arange(len(variants))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(variants) * 1.3), 4.8))
    ax.bar(x - width / 2, auc, width, label="AUC-PR", color="#4c72b0")
    ax.bar(x + width / 2, f1, width, label="F1-minority", color="#c44e52")
    ax.axhline(base_auc, ls="--", lw=1, color="#4c72b0", alpha=0.6)
    ax.axhline(base_f1, ls="--", lw=1, color="#c44e52", alpha=0.6)

    # Подписать дельту к base над барами GNN-адаптаций.
    for i, v in enumerate(variants):
        if v == BASE_LABEL:
            continue
        d_auc, d_f1 = auc[i] - base_auc, f1[i] - base_f1
        ax.annotate(f"{d_auc:+.3f}", (x[i] - width / 2, auc[i]), ha="center", va="bottom", fontsize=7, color="#26456e")
        ax.annotate(f"{d_f1:+.3f}", (x[i] + width / 2, f1[i]), ha="center", va="bottom", fontsize=7, color="#7a2c2f")

    if "XGBoost" in by:
        ax.axhline(by["XGBoost"].get("auc_pr") or 0, ls=":", lw=1.2, color="black", alpha=0.7,
                   label=f"XGBoost AUC-PR ({by['XGBoost'].get('auc_pr') or 0:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylabel("score")
    ax.set_title(f"IBM AML: ablation мультиграфовых адаптаций{regime} (пунктир = base GINe)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(results_dir, f"{out_name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def plot_ladder(rows: list[dict], results_dir: str = "results",
                out_name: str = "transition_ladder") -> None:
    """Главный визуал работы: F1/AUC-PR карабкаются по шагам L0→L6 (слабый→сильный).

    Линии AUC-PR и F1-minority по шагам лестницы; XGBoost(+fan) — горизонтальные
    референс-линии. Показывает, КАКОЙ рычаг закрывает разрыв до табличного бейзлайна.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by = {r["variant"]: r for r in rows}
    steps = [r for r in rows if r["variant"] not in LADDER_REFS]
    if len(steps) < 2:
        print("[ladder] <2 шагов лестницы — график пропущен")
        return

    labels = [r["variant"] for r in steps]
    f1 = [r.get("f1") or 0 for r in steps]
    auc = [r.get("auc_pr") or 0 for r in steps]
    x = np.arange(len(steps))

    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 1.4), 5))
    ax.plot(x, auc, "-o", color="#4c72b0", label="AUC-PR")
    ax.plot(x, f1, "-s", color="#c44e52", label="F1-minority")
    for ref, color in (("XGBoost", "black"), ("XGBoost+fan", "#888888")):
        if ref in by:
            v = by[ref].get("auc_pr") or 0
            ax.axhline(v, ls=":", lw=1.3, color=color, alpha=0.8,
                       label=f"{ref} AUC-PR ({v:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_title("IBM AML: переход слабый→сильный режим GNN (L0→L6)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(results_dir, f"{out_name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def plot_ibm_all_ranking(rows: list[dict], results_dir: str = "results") -> None:
    """Горизонтальный рейтинг всех IBM-вариантов по AUC-PR с F1-маркерами."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    vals = [r for r in rows if isinstance(r.get("auc_pr"), (int, float))]
    if not vals:
        return
    vals = sorted(vals, key=lambda r: r["auc_pr"])
    labels = [f"{r['variant']} [{r['regime']}]" for r in vals]
    auc = [r["auc_pr"] for r in vals]
    f1 = [r.get("f1") if isinstance(r.get("f1"), (int, float)) else np.nan for r in vals]
    colors = {
        "Tabular": "#4c72b0",
        "Hybrid test": "#8172b3",
        "Hybrid": "#8172b3",
        "GNN: GINe": "#55a868",
        "GNN: GINe+EU": "#dd8452",
        "GNN: PNA": "#c44e52",
        "Heuristics": "#888888",
    }
    bar_colors = [colors.get(r["family"], "#999999") for r in vals]

    y = np.arange(len(vals))
    fig_h = max(7, len(vals) * 0.34)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(y, auc, color=bar_colors, alpha=0.88, label="AUC-PR")
    ax.scatter(f1, y, color="black", s=18, label="F1", zorder=3)
    for yi, v in zip(y, auc):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("score")
    ax.set_title("IBM AML: рейтинг всех рассмотренных вариантов (test)")
    ax.set_xlim(0, max(auc) * 1.16)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(results_dir, "ibm_all_variants_ranking.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[saved] {out}")


def plot_ibm_ablation_heatmap(results_dir: str = "results") -> None:
    """Heatmap AUC-PR: GINe/Multi-GNN адаптации × режимы обучения."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = []
    row_labels = []
    for label, names in IBM_ABLATION_HEATMAP:
        row = []
        for name in names:
            path = os.path.join(results_dir, f"{name}_metrics.json")
            if not os.path.exists(path):
                row.append(np.nan)
            else:
                row.append(_metrics_from_json(path).get("auc_pr", np.nan))
        matrix.append(row)
        row_labels.append(label)
    arr = np.asarray(matrix, dtype=float)
    if np.isnan(arr).all():
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(arr, cmap="YlGnBu", vmin=0, vmax=np.nanmax(arr) * 1.15)
    ax.set_xticks(np.arange(len(IBM_ABLATION_HEATMAP_COLS)))
    ax.set_xticklabels(IBM_ABLATION_HEATMAP_COLS)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("IBM AML: AUC-PR адаптаций GINe по режимам")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            txt = "—" if np.isnan(arr[i, j]) else f"{arr[i, j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUC-PR")
    fig.tight_layout()
    out = os.path.join(results_dir, "ibm_ablation_heatmap.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[saved] {out}")


def plot_ibm_family_best(rows: list[dict], results_dir: str = "results") -> None:
    """Лучший результат по семействам: Tabular / Hybrid / GINe / PNA / Heuristics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    families = ["Hybrid", "Tabular", "GNN: PNA", "GNN: GINe", "GNN: GINe+EU", "Heuristics"]
    best_rows = []
    for fam in families:
        r = _best_by(rows, fam)
        if r is not None:
            best_rows.append(r)
    if not best_rows:
        return
    def short_label(r):
        family = r["family"].replace("GNN: ", "")
        variant = (
            r["variant"]
            .replace("XGBoost", "XGB")
            .replace("Multi-GNN emb -> XGB", "Multi-GNN emb->XGB")
            .replace("Multi-GNN + EU big-nbr", "Multi-GNN+EU big")
            .replace("Degree heuristics", "degree heuristics")
        )
        return f"{family}\n{variant}"

    labels = [short_label(r) for r in best_rows]
    auc = [r.get("auc_pr") or 0 for r in best_rows]
    f1 = [r.get("f1") or 0 for r in best_rows]

    x = np.arange(len(best_rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(best_rows) * 1.7), 4.8))
    ax.bar(x - width / 2, auc, width, label="AUC-PR", color="#4c72b0")
    ax.bar(x + width / 2, f1, width, label="F1", color="#c44e52")
    for i, v in enumerate(auc):
        ax.text(i - width / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(f1):
        ax.text(i + width / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("IBM AML: лучшие варианты по семействам")
    ax.set_ylim(0, max(max(auc), max(f1)) * 1.24)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(results_dir, "ibm_family_best.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[saved] {out}")


# ───────────────────── Per-pattern разбивка (RQ3, Фаза E) ─────────────────────
def collect_per_pattern(results_dir: str = "results") -> tuple[list[str], dict]:
    """Собрать per_pattern.f1 по семействам из results/ibm_*_metrics.json.

    Возвращает (labels, data): labels — порядок семейств (только присутствующие),
    data[label][pattern] = f1 (+ data[label]['__npos__'][pattern] = n_pos).
    """
    labels: list[str] = []
    data: dict = {}
    for name, label in PER_PATTERN_FAMILIES:
        path = os.path.join(results_dir, f"{name}_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            pp = json.load(f).get("per_pattern", {})
        if not pp:
            continue
        labels.append(label)
        pats = CANONICAL_PATTERNS + ["unknown"]
        data[label] = {p: (pp.get(p, {}) or {}).get("f1", 0.0) for p in pats}
        data[label]["__npos__"] = {p: (pp.get(p, {}) or {}).get("n_pos", 0) for p in pats}
    return labels, data


def write_per_pattern_table(labels: list[str], data: dict, results_dir: str = "results") -> None:
    """Таблица F1 × 8 паттернов × семейства (Markdown + CSV) с пометкой лучшего."""
    import csv

    md_path = os.path.join(results_dir, "per_pattern.md")
    csv_path = os.path.join(results_dir, "per_pattern.csv")
    npos = data[labels[0]]["__npos__"] if labels else {}

    rows_order = CANONICAL_PATTERNS + ["unknown"]
    cols = ["pattern", "n_pos"] + labels
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in rows_order:
            w.writerow([p, npos.get(p, 0)] + [f"{data[l][p]:.4f}" for l in labels])

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for p in rows_order:
        # «Лучшее» считаем только среди 8 канонических (unknown — справочно).
        vals = {l: data[l][p] for l in labels}
        best = max(vals, key=vals.get) if (vals and p != "unknown") else None
        cells = []
        for l in labels:
            v = vals[l]
            cells.append(f"**{v:.3f}**" if l == best and v > 0 else f"{v:.3f}")
        lines.append("| " + " | ".join([p, str(npos.get(p, 0))] + cells) + " |")

    matched = sum(npos.get(p, 0) for p in CANONICAL_PATTERNS)
    unknown_n = npos.get("unknown", 0)
    md = ("# IBM AML: F1 по паттернам отмывания (test, RQ3)\n\n"
          "_Режим: full-data/no-time — лучшая версия каждого семейства (XGBoost\n"
          "no-time, GINe/Multi-GNN full-data, эвристики)._\n\n"
          f"**Coverage:** из размеченных laundering-рёбер в test {matched} имеют тип\n"
          f"из 8 паттернов `HI-Small_Patterns.txt`, ещё {unknown_n} — `unknown` (нет\n"
          "совпадения в Patterns.txt). Анализ «по 8 паттернам» относится к первой\n"
          "группе; формулировать как «по размеченным паттернам», не «по всем\n"
          "laundering-рёбрам». Строка `unknown` — справочно.\n\n"
          "Жирным — лучшее семейство для паттерна. Фактический итог: XGBoost лидирует\n"
          "на всех паттернах (вкл. структурные cycle/scatter_gather, где ожидался\n"
          "перевес GNN); GNN-семейства следом, reverse тянет Multi-GNN вниз; степенные\n"
          "эвристики не дискриминативны (illicit-счета НИЖЕ по степени, чем легитимные —\n"
          "отмывание через низкостепенных «мулов», не хабы).\n\n"
          + "\n".join(lines) + "\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + "\n".join(lines))
    print(f"\n[saved] {csv_path}\n[saved] {md_path}")


def plot_per_pattern(labels: list[str], data: dict, results_dir: str = "results") -> None:
    """Сгруппированный bar-chart F1 по паттернам (группы = семейства моделей)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#000000", "#4c72b0", "#c44e52", "#55a868", "#8172b3"]
    x = np.arange(len(CANONICAL_PATTERNS))
    n = len(labels)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, l in enumerate(labels):
        vals = [data[l][p] for p in CANONICAL_PATTERNS]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=l, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(CANONICAL_PATTERNS, rotation=20, ha="right")
    ax.set_ylabel("F1 (позитив = laundering)")
    ax.set_title("IBM AML: F1 по 8 паттернам — сравнение семейств (RQ3)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(results_dir, "per_pattern.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def summarize_per_pattern(results_dir: str = "results") -> None:
    """Per-pattern сводка (таблица + график) из готовых results/."""
    labels, data = collect_per_pattern(results_dir)
    if not labels:
        print("Нет per_pattern данных в", results_dir)
        return
    write_per_pattern_table(labels, data, results_dir)
    plot_per_pattern(labels, data, results_dir)


def summarize_ibm(results_dir: str = "results") -> None:
    """Собрать IBM-сводку: ablation (с временем и без) + per-pattern из готовых results/."""
    rows = collect_ibm(results_dir, IBM_VARIANTS)
    if not rows:
        print("Нет IBM-результатов в", results_dir, "(прогони --run-ibm на ПК с CUDA)")
        return
    rows_all = collect_ibm_all(results_dir)
    if rows_all:
        print("\n=== все рассмотренные IBM-варианты ===")
        write_ibm_all_table(rows_all, results_dir)
        plot_ibm_all_ranking(rows_all, results_dir)
        plot_ibm_family_best(rows_all, results_dir)
        plot_ibm_ablation_heatmap(results_dir)

    write_ibm_table(rows, results_dir, name="ibm_comparison", regime=" (с norm_time)")
    plot_ablation(rows, results_dir, out_name="ablation", regime=" (с временем)")

    # Режим без norm_time (P1.6) — если перепрогнан.
    rows_nt = collect_ibm(results_dir, IBM_VARIANTS_NOTIME)
    if len([r for r in rows_nt if r["variant"] != "XGBoost"]) >= 2:
        print("\n=== режим без norm_time ===")
        write_ibm_table(rows_nt, results_dir, name="ibm_comparison_notime", regime=" (без norm_time)")
        plot_ablation(rows_nt, results_dir, out_name="ablation_notime", regime=" (без времени)")

    # Режим full-data обучения (no-time + все train-рёбра) — если перепрогнан.
    rows_fd = collect_ibm(results_dir, IBM_VARIANTS_FULLDATA)
    if len([r for r in rows_fd if r["variant"] != "XGBoost"]) >= 2:
        n_gnn = len([r for r in rows_fd if r["variant"] not in ("XGBoost",)])
        # Полный ablation только если есть промежуточные варианты (rev/port/ego),
        # иначе это сравнение base vs full (честно отражаем в заголовке).
        partial = n_gnn < 4
        tag = " (full-data/no-time — base vs full)" if partial else " (full-data/no-time)"
        print(f"\n=== режим full-data (no-time + все train-рёбра){' — base vs full' if partial else ''} ===")
        write_ibm_table(rows_fd, results_dir, name="ibm_comparison_fulldata", regime=tag)
        plot_ablation(rows_fd, results_dir, out_name="ablation_fulldata", regime=tag)

    # Режим GIN+EU (edge updates поверх full-data) — если перепрогнан.
    rows_eu = collect_ibm(results_dir, IBM_VARIANTS_EU)
    if any(r["variant"].endswith("+EU") for r in rows_eu):
        print("\n=== режим full-data + edge-updates (GIN+EU) ===")
        write_ibm_table(rows_eu, results_dir, name="ibm_comparison_eu",
                        regime=" (full-data + edge-updates)")

    # Лестница перехода слабый→сильный (L0…L6) — собирается по мере появления
    # шагов сильного режима (PNA/большие окрестности). Главный визуал работы.
    rows_ladder = collect_ibm(results_dir, IBM_VARIANTS_LADDER)
    n_steps = len([r for r in rows_ladder if r["variant"] not in LADDER_REFS])
    if n_steps >= 2:
        print("\n=== лестница перехода слабый→сильный (L0→L6) ===")
        write_ibm_table(
            rows_ladder, results_dir, name="ibm_ladder",
            regime=" (L0→L6)",
            title="лестница перехода слабый→сильный режим GNN",
            intro=("Метрики: AUC-PR и F1-minority (позитив = laundering). Каждый шаг "
                   "добавляет ОДИН рычаг при фиксированном протоколе (full-data/no-time,\n"
                   "pos_weight=100, единые окрестности/глубина для L3–L6): L0–L2 — слабый "
                   "режим ([10,10]/2 слоя), L3 — те же адаптации при больших окрестностях,\n"
                   "L4–L6 — смена агрегатора на PNA вплоть до целевого Multi-PNA+EU "
                   "(Egressy 2024). XGBoost(+fan) — табличный референс (горизонтали)."))
        plot_ladder(rows_ladder, results_dir)

    summarize_per_pattern(results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение моделей (Elliptic + IBM AML)")
    parser.add_argument("--run", action="store_true", help="прогнать все Elliptic-конфиги")
    parser.add_argument("--run-ibm", action="store_true",
                        help="прогнать IBM-сетку (XGBoost + ablation edge-GNN) — нужен CUDA (ПК)")
    parser.add_argument("--ibm", action="store_true",
                        help="собрать IBM-сводку (таблица + ablation.png) из готовых results/")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    # IBM-режим самостоятелен (Elliptic-сводка — отдельным вызовом без флагов).
    if args.run_ibm or args.ibm:
        if args.run_ibm:
            run_ibm()
        summarize_ibm(args.results_dir)
        return

    if args.run:
        run_all()
    rows = collect(args.results_dir)
    if not rows:
        print("Нет результатов в", args.results_dir)
        return
    write_table(rows, args.results_dir)
    plot(rows, args.results_dir)


if __name__ == "__main__":
    main()
