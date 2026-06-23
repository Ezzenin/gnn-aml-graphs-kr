# IBM AML HI-Small (test): бейзлайн + ablation Multi-GNN (full-data/no-time)

Главные метрики: AUC-PR и F1 (позитив = laundering). Ablation: вклад
адаптаций reverse / port / ego поверх базовой GINe (RQ2).

| variant | auc_pr | f1 | recall_at_precision_90 | recall |
|---|---|---|---|---|
| XGBoost | 0.2398 | 0.2976 | 0.0217 | 0.3103 |
| XGBoost+fan | 0.2282 | 0.2825 | 0.0156 | 0.2586 |
| Hybrid GNN→XGBoost | 0.3299 | 0.3981 | 0.0161 | 0.3065 |
| GINe (base) | 0.0427 | 0.1199 | 0.0000 | 0.1340 |
| +reverse | 0.0401 | 0.0831 | 0.0000 | 0.1007 |
| +port | 0.0542 | 0.1114 | 0.0000 | 0.1279 |
| +ego | 0.0327 | 0.0810 | 0.0000 | 0.2653 |
| Multi-GNN (full) | 0.0517 | 0.1233 | 0.0000 | 0.1051 |

## Reference results (literature) — F1-minority, %

> Внимание: другой сплит (60/20/20) и режим обучения (все train-рёбра, class
> weights). НЕ сравнивать напрямую с нашими AUC-PR/F1 выше; приведено для
> ориентира масштаба и анализа расхождений (см. docs/lit_benchmarks.md).

| Модель (источник) | F1-minority % | примечание |
|---|---|---|
| XGBoost+GF (Altman 2023) | 63.2 | + подграфовые fan/cycle-фичи (GFP) |
| LightGBM+GF (Altman 2023) | 62.9 |  |
| GIN base (Egressy 2024) | 28.7 | 2 слоя, все train-рёбра, class weights |
| GIN+Ports | 54.9 | port numbering |
| GIN+ReverseMP | 46.8 | reverse MP — у нас не переносится при [10,10] |
| Multi-GIN (rev+port+ego) | 57.1 | ego поверх rev+port почти не добавляет |
| Multi-PNA+EU (SOTA) | 68.2 | единственный обошёл GBT+GF на всех AML |
| XGBoost без GF (Blanuša 2024) | 24.5 | ≈ наш XGBoost 19.0 по порядку |

Наш режим ослаблен (сабсэмпл негативов, окрестности [10,10], XGBoost без
подграфовых GF-фич), поэтому абсолютные числа ниже. Устойчивого переноса
эффекта адаптаций (reverse/port/ego) не наблюдается ни в одном из трёх
режимов (с временем / без времени / full-data): знак и порядок дельт к base
меняются между режимами, full-data не воспроизводит направление Egressy.
См. docs/lit_benchmarks.md §1.2.
