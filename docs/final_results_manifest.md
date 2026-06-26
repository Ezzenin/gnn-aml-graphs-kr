# Final results manifest

Назначение: единая карта артефактов перед написанием текста работы. Здесь
зафиксированы финальные claims, воспроизводимость результатов и ограничения,
которые нужно честно вынести в текст.

## Финальные claims

1. **Standalone GNN не превосходит XGBoost на IBM AML HI-Small.**
   Лучший пересчитанный на Kaggle standalone GNN — `PNA` в full-data режиме:
   test AUC-PR `0.059`, F1 `0.087`. Лучший обычный табличный baseline — `XGBoost no-time`:
   test AUC-PR `0.240`, F1 `0.298`.

2. **Граф полезен как representation.**
   Лучший общий вариант — гибрид `Multi-GNN embedding -> XGBoost`: test AUC-PR
   `0.330`, F1 `0.398`. Он выше raw-edge XGBoost (`0.289`) и выше full XGBoost
   (`0.248`). Это главный позитивный результат проекта.

3. **Multi-GNN адаптации не дают устойчивого standalone-прироста.**
   В режимах with-time / no-time / full-data знак эффекта reverse, port, ego
   меняется. В full-data/no-time `GINe` и `GINe+port` примерно равны, а
   `Multi-GNN` ниже base.

4. **PNA остается лучшим подтвержденным standalone GNN, но порядок GNN нестабилен.**
   В Kaggle verification PNA (`0.059`) выше GINe/Multi-GNN, но существенно ниже
   XGBoost/hybrid; ранее сохраненный локальный PNA был выше, поэтому в тексте
   не стоит делать сильный claim о точном порядке GNN.

5. **Высокий ROC-AUC при низком AUC-PR означает не случайность GNN, а провал
   в высокоточном режиме при prevalence около 0.1%.**
   Это нужно формулировать отдельно: GNN ранжирует, но плохо отделяет top-risk
   при жесткой PR-метрике.

## Главные артефакты

| Артефакт | Что использовать в тексте |
|---|---|
| `results/ibm_all_variants.md` | полный рейтинг всех IBM-вариантов |
| `results/ibm_all_variants_ranking.png` | главный обзорный график всех вариантов |
| `results/ibm_family_best.png` | короткий график для основного текста: Hybrid / Tabular / PNA / GINe |
| `results/ibm_ablation_heatmap.png` | compact RQ2: reverse/port/ego по трем режимам |
| `results/ibm_ladder.md`, `results/transition_ladder.png` | лестница weak -> strong GNN |
| `results/per_pattern.md`, `results/per_pattern.png` | RQ3 по паттернам |
| `docs/kaggle_gpu_results.md` | финальная интерпретация Kaggle GPU verification |
| `notebooks/kaggle_gpu_results_2.ipynb` | выполненный Kaggle notebook с финальным прогоном |
| `docs/findings_summary.md` | развернутый тезис для текста/защиты |

## Команды воспроизведения

Сборка всех таблиц и графиков из уже готовых JSON:

```bash
python3 -m src.compare --ibm
```

Базовые табличные прогоны:

```bash
python3 -m src.train_baseline --config configs/ibm_xgb.yaml
python3 -m src.train_baseline --config configs/ibm_xgb_notime.yaml
python3 -m src.train_baseline --config configs/ibm_xgb_fan.yaml
```

GNN ablation:

```bash
python3 -m src.train_edge --config configs/ibm_gine.yaml
python3 -m src.train_edge --config configs/ibm_gine_rev.yaml
python3 -m src.train_edge --config configs/ibm_gine_port.yaml
python3 -m src.train_edge --config configs/ibm_gine_ego.yaml
python3 -m src.train_edge --config configs/ibm_multignn.yaml
python3 -m src.train_edge --config configs/ibm_gine_fulldata.yaml
python3 -m src.train_edge --config configs/ibm_multignn_fulldata.yaml
```

Гибридный эксперимент:

```bash
python3 scripts/hybrid_gnn_xgb.py - checkpoints/ibm_multignn_fulldata.pt results/ibm_hybrid_gnn_xgb_metrics.json
```

Текущий `results/ibm_hybrid_gnn_xgb_metrics.json` — полный Kaggle-прогон:
AUC-PR/F1/ROC-AUC/precision/recall/threshold сохранены.

## Неполные результаты

`results/ibm_multipna_fulldata_metrics.json` и
`results/ibm_multipna_eu_fulldata_metrics.json` восстановлены из run log. Они
используются только для AUC-PR/F1 в ladder и общем рейтинге. Для них не сохранены:
threshold, precision, recall, ROC-AUC, per-pattern. В тексте нельзя делать
тонкие выводы по precision/recall для L5/L6.

## Multi-seed status

Финальные результаты сейчас single-seed (`seed=42`). Многосидовый прогон не
выполнен из-за стоимости полного IBM HI-Small пайплайна и гибридного extract
embeddings на 5.08M рёбер. В тексте это нужно указать как ограничение. Формулировка:
«Эксперименты выполнены для одного фиксированного seed; устойчивость оценивалась
по нескольким режимам обучения и архитектурным абляциям, но не по независимым
повторам seed».
