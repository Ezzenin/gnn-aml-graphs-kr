# IBM AML HI-Small: все рассмотренные варианты (test)

Единая таблица для отчета: табличные бейзлайны, все режимы GINe/Multi-GNN,
edge-updates, сильная PNA-лестница, эвристики и гибрид GNN→XGBoost.
Главная сортировка ниже — по AUC-PR на test.

## Короткий вывод

- Лучший общий вариант: **Multi-GNN emb -> XGBoost** (Hybrid, AUC-PR 0.330, F1 0.3981).
- Лучший standalone GNN: **PNA** (big/full-data, AUC-PR 0.059, F1 0.0872).
- Лучший табличный baseline из отдельных прогонов: **XGBoost no-time** (AUC-PR 0.240).
- Лучший гибрид: **Multi-GNN emb -> XGBoost** (AUC-PR 0.330); это показывает, что графовый эмбеддинг полезен как представление, хотя standalone GNN слабее XGBoost.

## Рейтинг

| family | regime | variant | auc_pr | f1 | roc_auc | recall_at_precision_90 | precision | recall |
|---|---|---|---|---|---|---|---|---|
| Hybrid | GNN embedding | Multi-GNN emb -> XGBoost | 0.3299 | 0.3981 | 0.9579 | 0.0161 | 0.5680 | 0.3065 |
| Hybrid test | raw edge attrs | XGBoost raw-edge | 0.2893 | 0.4121 | 0.9614 | 0.0000 | 0.4481 | 0.3815 |
| Tabular | no-time | XGBoost no-time | 0.2398 | 0.2976 | 0.9709 | 0.0217 | 0.2859 | 0.3103 |
| Tabular | fan features | XGBoost+fan | 0.2282 | 0.2825 | 0.9697 | 0.0156 | 0.3112 | 0.2586 |
| Tabular | with norm_time | XGBoost | 0.1289 | 0.1900 | 0.9674 | 0.0000 | 0.1529 | 0.2508 |
| GNN: PNA | big/full-data | PNA | 0.0591 | 0.0872 | 0.9639 | 0.0000 | 0.1369 | 0.0640 |
| GNN: PNA | big/full-data | Multi-PNA | 0.0574 | 0.0960 | — | 0.0000 | — | — |
| GNN: GINe | full-data/no-time | GINe + port | 0.0542 | 0.1114 | 0.9443 | 0.0000 | 0.0986 | 0.1279 |
| GNN: GINe | full-data/no-time | Multi-GNN | 0.0517 | 0.1233 | 0.9171 | 0.0000 | 0.1492 | 0.1051 |
| GNN: GINe | with norm_time | Multi-GNN | 0.0510 | 0.1202 | 0.9472 | 0.0000 | 0.0990 | 0.1529 |
| GNN: GINe | no-time | GINe + ego | 0.0491 | 0.1245 | 0.9264 | 0.0000 | 0.1110 | 0.1418 |
| GNN: GINe | no-time | GINe | 0.0442 | 0.1199 | 0.9432 | 0.0000 | 0.1074 | 0.1357 |
| GNN: GINe | full-data/no-time | GINe | 0.0427 | 0.1199 | 0.9408 | 0.0000 | 0.1085 | 0.1340 |
| GNN: GINe+EU | big/full-data | Multi-GNN + EU big-nbr | 0.0423 | 0.0957 | 0.9600 | 0.0000 | 0.0746 | 0.1335 |
| GNN: GINe | full-data/no-time | GINe + reverse | 0.0401 | 0.0831 | 0.9267 | 0.0000 | 0.0708 | 0.1007 |
| GNN: PNA | big/full-data | Multi-PNA + EU | 0.0401 | 0.0950 | — | 0.0000 | — | — |
| GNN: GINe | with norm_time | GINe + reverse | 0.0396 | 0.0851 | 0.9530 | 0.0000 | 0.0589 | 0.1529 |
| GNN: GINe | with norm_time | GINe + port | 0.0385 | 0.1055 | 0.9223 | 0.0000 | 0.0874 | 0.1329 |
| GNN: GINe | with norm_time | GINe + ego | 0.0366 | 0.1077 | 0.9286 | 0.0000 | 0.0773 | 0.1774 |
| GNN: GINe | no-time | GINe + port | 0.0354 | 0.1002 | 0.9141 | 0.0000 | 0.0958 | 0.1051 |
| GNN: GINe+EU | full-data/no-time | Multi-GNN + EU | 0.0346 | 0.1153 | 0.9472 | 0.0000 | 0.0836 | 0.1858 |
| GNN: GINe | full-data/no-time | GINe + ego | 0.0327 | 0.0810 | 0.9318 | 0.0000 | 0.0478 | 0.2653 |
| GNN: GINe+EU | full-data/no-time | GINe + EU | 0.0260 | 0.0587 | 0.9292 | 0.0000 | 0.0315 | 0.4288 |
| GNN: GINe | no-time | GINe + reverse | 0.0255 | 0.0510 | 0.9455 | 0.0000 | 0.0269 | 0.5028 |
| GNN: GINe | with norm_time | GINe | 0.0190 | 0.0562 | 0.7947 | 0.0000 | 0.0318 | 0.2442 |
| GNN: GINe | no-time | Multi-GNN | 0.0125 | 0.0332 | 0.8553 | 0.0000 | 0.0179 | 0.2297 |
| Heuristics | no-time | Degree heuristics | 0.0011 | 0.0031 | 0.2161 | 0.0000 | 0.0017 | 0.0150 |

Примечание: строки `Multi-PNA` и `Multi-PNA + EU` восстановлены из run log,
поэтому для них отсутствуют `precision`, `recall`, `ROC-AUC` и threshold.
