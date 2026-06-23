# Kaggle GPU verification: выводы по финальному прогону

Источник: `kaggle_gpu_results_2.zip`, распакован в `results/` и
`checkpoints/kaggle/`. Выполненный Kaggle notebook сохранен как
`notebooks/kaggle_gpu_results_2.ipynb`. Прогон выполнен на Kaggle GPU для
ключевых full-data GNN и полного гибрида `GNN embedding -> XGBoost`; затем
локально пересобраны сводки через `python3 -m src.compare --ibm`.

## Что прошло

| Артефакт | Статус |
|---|---|
| `results/ibm_gine_fulldata_metrics.json` | пересчитан на Kaggle |
| `results/ibm_multignn_fulldata_metrics.json` | пересчитан на Kaggle |
| `results/ibm_pna_fulldata_metrics.json` | пересчитан на Kaggle |
| `results/ibm_hybrid_gnn_xgb_metrics.json` | полный гибрид, full metrics |
| `checkpoints/kaggle/ibm_gine_fulldata.pt` | сохранен |
| `checkpoints/kaggle/ibm_multignn_fulldata.pt` | сохранен |
| `checkpoints/kaggle/ibm_pna_fulldata.pt` | сохранен |
| `notebooks/kaggle_gpu_results_2.ipynb` | выполненный notebook сохранен |
| `results/ibm_all_variants.md` / PNG | пересобраны |

## Основные test-метрики

| вариант | AUC-PR | F1 | ROC-AUC | precision | recall | R@P90 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost no-time | 0.2398 | 0.2976 | 0.9709 | 0.2859 | 0.3103 | 0.0217 |
| XGBoost raw-edge | 0.2893 | 0.4121 | 0.9614 | 0.4481 | 0.3815 | 0.0000 |
| **Hybrid GNN -> XGBoost** | **0.3299** | **0.3981** | 0.9579 | **0.5680** | 0.3065 | 0.0161 |
| GINe full-data | 0.0427 | 0.1199 | 0.9408 | 0.1085 | 0.1340 | 0.0000 |
| Multi-GNN full-data | 0.0517 | 0.1233 | 0.9171 | 0.1492 | 0.1051 | 0.0000 |
| PNA full-data | 0.0591 | 0.0872 | 0.9639 | 0.1369 | 0.0640 | 0.0000 |

## Главный вывод

Финальный Kaggle-прогон подтверждает базовый результат проекта:

1. **Standalone GNN остаются слабыми относительно табличных моделей.**
   Лучший пересчитанный standalone GNN — PNA full-data с AUC-PR `0.0591`;
   это выше GINe/Multi-GNN, но далеко ниже XGBoost no-time `0.2398`.

2. **Граф полезен как representation в гибриде.**
   Полный гибрид `GNN embedding -> XGBoost` дал AUC-PR `0.3299`, что выше:
   raw-edge XGBoost `0.2893` на `+0.0406`, и full XGBoost `0.2484` на `+0.0815`.
   Это самый сильный подтвержденный результат в текущем проекте.

3. **Порядок внутри standalone GNN нестабилен, но это не меняет общий вывод.**
   В Kaggle-прогоне Multi-GNN (`0.0517`) оказался выше GINe (`0.0427`), а PNA
   (`0.0591`) выше обоих. Ранее локальный PNA был заметно выше; различие
   объяснимо stochastic neighbor sampling, другим batch size (`2048` на Kaggle
   вместо исходного strong `4096`) и чувствительностью PNA. В тексте лучше не
   делать сильный claim о точном порядке GNN, а говорить: все standalone GNN
   остаются существенно ниже XGBoost/hybrid.

4. **Высокий ROC-AUC не превращается в хороший PR.**
   PNA имеет ROC-AUC `0.9639`, но AUC-PR только `0.0591` и R@P90 `0.0`.
   Это подтверждает, что GNN умеет грубо ранжировать, но не дает достаточно
   чистый top-risk список при prevalence около `0.1%`.

## Гибрид: детальнее

Полный гибрид теперь сохранен не как summary, а как full metrics:

```text
raw-edge XGBoost:   AUC-PR 0.2893, F1 0.4121
full XGBoost:       AUC-PR 0.2484, F1 0.2874
GNN embedding XGB:  AUC-PR 0.3299, F1 0.3981
delta vs raw:       +0.0406 AUC-PR
```

Интерпретация: GNN-head сам по себе слабый, но его embedding содержит
комплементарный структурный сигнал. XGBoost лучше использует этот сигнал, чем
MLP-head GNN, особенно при сильном дисбалансе классов.

## Per-pattern вывод

В пересобранном `results/per_pattern.md` XGBoost+fan остается лучшим на всех
8 размеченных паттернах. Multi-GNN в Kaggle full-data стал сильнее GINe на части
паттернов (`gather_scatter`, `scatter_gather`, `random`, `stack`), но абсолютный
уровень F1 по паттернам все равно существенно ниже табличных моделей.

Практическая формулировка для текста: графовые модели видят часть структурного
сигнала, но в текущем протоколе этот сигнал лучше использовать как feature
extractor для бустинга, а не как самостоятельный классификатор.

## Что использовать в тексте

Основные графики:

- `results/ibm_all_variants_ranking.png` — общий рейтинг.
- `results/ibm_family_best.png` — компактное сравнение семейств.
- `results/ibm_ablation_heatmap.png` — эффект GINe/Multi-GNN адаптаций по режимам.
- `results/per_pattern.png` — паттерны.

Основные таблицы:

- `results/ibm_all_variants.md`
- `results/ibm_comparison_fulldata.md`
- `results/ibm_ladder.md`
- `results/per_pattern.md`

Ограничения, которые нужно честно написать:

- один seed (`42`);
- Kaggle PNA запускался с batch `2048`, а не `4096`;
- часть L5/L6 Multi-PNA результатов остается summary из run log;
- таблицы смешивают свежие Kaggle-пересчеты ключевых моделей и ранее сохраненную
  сетку для reverse/port/ego/EU.
