# Hybrid leakage audit

Цель: зафиксировать, почему результат `GNN embedding -> XGBoost` можно использовать
как честный эксперимент, а не как утечку будущего в признаки.

## Что делает гибрид

Скрипт `scripts/hybrid_gnn_xgb.py` извлекает embedding классифицируемого ребра
из обученной GNN перед финальной head-MLP:

```text
[h_src || h_dst || e_label]
```

Затем этот embedding добавляется к raw-edge признакам и подается в XGBoost.

## Контексты message passing

В гибриде используются те же антиутечечные контексты, что в `src.train_edge`:

| Split seed-рёбер | Контекст для GNN embedding | Будущие test-рёбра видны? |
|---|---|---|
| train | только train-рёбра | нет |
| val | только train-рёбра | нет |
| test | train+val-рёбра | test-рёбра нет |

Это означает, что test embedding не строится на test-транзакциях как соседях.
Он использует только прошлое относительно test-периода.

## Признаки seed-рёбра

Для каждого классифицируемого seed-ребра модель получает `data.edge_attr` этого
же ребра: amount/currency/payment format/time-ablation state. Это не утечка,
потому что в edge-classification признаки самой оцениваемой транзакции доступны
в момент скоринга. Та же информация доступна XGBoost baseline.

## XGBoost stage

XGBoost обучается только на `train_mask`. Threshold выбирается на val и применяется
к test. Test labels не используются ни при обучении GNN, ни при обучении XGBoost,
ни при подборе порога.

## Что остается ограничением

- Текущий сохраненный `results/ibm_hybrid_gnn_xgb_metrics.json` — полный Kaggle
  прогон: сохранены val/test metrics, threshold, precision, recall и R@P90.
- Гибрид использует checkpoint GNN из одного seed.
- Embedding извлекается из слабого режима GINe/Multi-GNN; strong PNA embedding
  как гибридный признак отдельно не проверялся.

## Защищаемая формулировка

Гибрид не доказывает, что standalone GNN лучше табличной модели. Он доказывает
более узкий и важный тезис: GNN learned representation содержит комплементарный
структурный сигнал, который XGBoost может использовать лучше, чем GNN-head при
сильном дисбалансе классов.
