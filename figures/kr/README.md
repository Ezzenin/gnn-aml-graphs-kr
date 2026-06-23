# Визуалы для текста КР

Папка содержит плоский набор рисунков для вставки в текст работы по плану
`plan_KR_v3_napisanie_teksta.md`. Префикс `Rxx` соответствует номеру рисунка в
плане.

| Рисунок | Файл | Раздел |
|---|---|---|
| Р1 | `R01_elliptic_model_comparison.png` | 4.2 |
| Р2 | `R02a_elliptic_xgboost_pr_curve.png` | 4.2 / Прил. |
| Р2 | `R02b_elliptic_graphsage_pr_curve.png` | 4.2 / Прил. |
| Р2 | `R02c_elliptic_gcn_pr_curve.png` | 4.2 / Прил. |
| Р2 | `R02d_elliptic_gin_pr_curve.png` | 4.2 / Прил. |
| Р2 | `R02e_elliptic_pna_pr_curve.png` | 4.2 / Прил. |
| Р2 | `R02f_elliptic_gat_pr_curve.png` | 4.2 / Прил. |
| Р2 | `R02g_elliptic_logreg_pr_curve.png` | 4.2 / Прил. |
| Р3 | `R03_feature_ablation_xgboost.png` | 4.3 |
| Р4 | `R04_ibm_all_variants_ranking.png` | 4.3 / 4.6 |
| Р5 | `R05_ablation_heatmap_reverse_port_ego.png` | 4.4 |
| Р6 | `R06_ladder_l0_l6_vs_xgboost.png` | 4.4 |
| Р7 | `R07_hybrid_lift_embedding_structure.png` | 4.5 |
| Р8 | `R08_family_best_comparison.png` | 4.5 |
| Р9 | `R09a_ibm_xgboost_notime_pr_curve.png` | 4.5 / Прил. |
| Р9 | `R09b_ibm_best_gnn_pna_pr_curve.png` | 4.5 / Прил. |
| Р9 | `R09c_ibm_multignn_pr_curve.png` | 4.5 / Прил. |
| Р9 | `R09d_ibm_gine_pr_curve.png` | 4.5 / Прил. |
| Р10 | `R10_per_pattern_f1.png` | 4.6 |
| Р11 | `R11_degree_diagnostics_mules_not_hubs.png` | 4.7 |
| Р12 | `R12_gnn_roc_auc_vs_auc_pr.png` | 4.7 |
| Р13 | `R13_suspicious_chain_highlight.png` | 4.8 |
| Р14 | `R14_graph_representations_schema.png` | 2.2 |
| Р15 | `R15_hybrid_gnn_xgboost_dataflow.png` | 3.5 |

Примечание по Р9: полноценная PR-кривая гибрида не восстановима из сохраненного
`results/ibm_hybrid_gnn_xgb_metrics.json`, потому что в JSON сохранены итоговые
метрики, но не все test scores. Поэтому в папку положены доступные PR-кривые
табличного baseline и standalone GNN; гибрид сравнивается отдельным lift-рисунком
Р7 и таблицей метрик.
