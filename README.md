# Курсовой проект: графовые нейронные сети для выявления цепочек финансовых операций

Курсовая работа. НИУ ВШЭ, ФКН, магистерская программа «Финансовые технологии и анализ
данных» (МФТАД). Автор — Есенин Александр Сергеевич.

Тема: «Исследование методов построения графовых представлений транзакционных данных и
применения графовых нейронных сетей для выявления цепочек финансовых операций».
Английское название: *Research on methods for constructing graph representations of
transactional data and applying graph neural networks for detecting chains of financial
operations.*

## Что делает проект

Проект реализует и анализирует воспроизводимый пайплайн детекции **цепочек отмывания**
(8 типологий AMLSim: fan-out, fan-in, gather-scatter, scatter-gather, cycle, random,
bipartite, stack) и отвечает на вопрос, **где графовое представление реально полезно**
относительно сильного табличного бейзлайна.

Основная экспериментальная линия (IBM AML HI-Small):

* представление: направленный мультиграф «узел = счёт, ребро = транзакция»,
  edge-classification; строгий **temporal split** без утечки (узловые признаки и
  параллельные рёбра считаются только по train);
* табличный бейзлайн: **XGBoost** на признаках ребра (`amount/currency/format` +
  параллельные рёбра, degree/sum концов);
* графовые модели: базовая **GINe**, мультиграфовые адаптации Egressy
  (reverse MP, port numbering, ego-IDs), edge-updates, **PNA** и Multi-PNA;
* **гибрид**: эмбеддинг ребра из обученной GNN подаётся доп-признаками в XGBoost;
* оценка при дисбалансе 0.1%: **AUC-PR**, **F1-minority**, **ROC-AUC**,
  **recall @ precision 0.9**, разбивка по 8 типологиям; порог фиксируется по val.

Доп разрез: node-classification на **Elliptic (Bitcoin)** — линейка
GCN/SAGE/GAT/GIN/PNA против XGBoost/LogReg.

**Главный результат:** граф полезен как *обучаемое представление* — гибрид
`Multi-GNN эмбеддинг → XGBoost` даёт лучший общий результат (test AUC-PR **0.330**),
тогда как standalone edge-GNN уступают XGBoost. То есть вклад графа — комплементарный
структурный сигнал в гибридной схеме, а не превосходство GNN-head при экстремальном
дисбалансе.

## Структура репозитория

```
configs/      YAML-конфигурации экспериментов (Elliptic + IBM AML)
src/          реализация: datasets, models, train/train_edge/train_baseline,
              metrics, heuristics, compare, eval, graph_build
scripts/      воспроизводимые entrypoints (гибрид, визуализации, GPU-прогон)
tests/        регрессионные и smoke-тесты (граф-билдеры, метрики, модели)
notebooks/    reader-facing ноутбуки поверх сохранённых результатов
data/          раскладка данных (raw/cache в .gitignore)
results/      финальные метрики (json), таблицы (md), фигуры (png), сводные csv
figures/      фигуры для текста курсовой (figures/kr/)
docs/         постановка, обзор литературы, сводки наблюдений, аудит, ограничения
app/          Streamlit-демо: скоринг ребра обученной GNN + визуализация цепочки
```

## Гайд по докам

1. `docs/findings_summary.md` —  тезис: главный результат
   (гибрид), почему чистая GNN проигрывает (feature-ablation), RQ2/RQ3, контекст
   литературы, ограничения.
2. `docs/final_results_manifest.md` — карта артефактов и финальных claims для текста.
3. `results/ibm_all_variants.md` + `results/ibm_all_variants_ranking.png` — полный
   рейтинг всех IBM-вариантов; `results/ibm_family_best.png` — короткий обзор
   Hybrid / Tabular / PNA / GINe.
4. `docs/kaggle_gpu_results.md` — интерпретация Kaggle GPU verification;
   `notebooks/kaggle_gpu_results_2.ipynb` — выполненный финальный прогон.
5. `docs/lit_benchmarks.md`, `docs/lit_review.md`, `docs/problem_statement.md` —
   литература, опорные числа и постановка.

Протокол анти-утечки гибрида (эмбеддинги извлекаются на тех же контекстах, что и
обучение GNN; XGBoost учится на train, порог — по val, метрики — на полном test)
описан в `docs/findings_summary.md` §1 и реализован в `scripts/hybrid_gnn_xgb.py`.



## Воспроизводимые эксперименты

Зависимости: `requirements.txt`. Обучение GNN тяжёлое (рекомендуется CUDA; на CPU/MPS
работает, но медленно — любой `device: cuda` сам откатывается на cpu, см.
`resolve_device`). XGBoost, эвристики, сборка сводок и Streamlit — на CPU.

Данные с Kaggle **HI-Small**:
`data/ibm_aml/HI-Small_Trans.csv` (~5.08M транзакций) и `HI-Small_Patterns.txt`.

```bash
pip install -r requirements.txt

# Этап 1 — Elliptic (node-classification): GNN-линейка vs XGBoost
python -m src.train_baseline --config configs/elliptic_xgb.yaml
python -m src.compare --run                 # все модели + сводка results/comparison.md

# Этап 2 — IBM AML (edge-classification)
python -m src.train_baseline --config configs/ibm_xgb_notime.yaml   # табличный бейзлайн (CPU)
python -m src.train_edge --config configs/ibm_gine_fulldata.yaml    # base GINe (CUDA)
python -m src.train_edge --config configs/ibm_multignn_fulldata.yaml  # Multi-GNN ablation
python -m src.heuristics                                            # графовые эвристики (RQ3)
python -m src.compare --ibm                                         # все таблицы и графики IBM

# Гибрид GNN→XGBoost (главный результат); нужен чекпоинт GNN
python scripts/hybrid_gnn_xgb.py - checkpoints/ibm_multignn_fulldata.pt results/ibm_hybrid_gnn_xgb_metrics.json

# Продукт (режим антифрод)
streamlit run app/streamlit_app.py
```

Конфигурации: вся сетка IBM в `configs/ibm_*` (три режима признаков — `with-time` /
`_notime` / `_fulldata`; адаптации `rev/port/ego/eu`; `pna`/`multipna`). Сильный режим
на арендованном GPU — `scripts/run_strong_gnn_vast.sh`, `notebooks/colab_strong_gnn.ipynb`.

Рядом с каждым прогоном сохранены `<name>_metrics.json` (val/test метрики, threshold,
per-pattern) и `<name>_pr_curve.png`, чтобы reported numbers можно было проверить без
повторного inference. Текущий `results/ibm_hybrid_gnn_xgb_metrics.json` — полный
Kaggle-прогон с precision/recall/threshold.


## Рез-ы

Финальные числа (test, single-seed; верифицированы на GPU):

* **Гибрид Multi-GNN→XGBoost — лучший общий результат:** AUC-PR **0.330**, F1 0.398,
  precision 0.568. Выше raw-edge XGBoost (0.289) и полного XGBoost (0.248).
* **Лучший standalone GNN (Kaggle):** PNA, AUC-PR **0.059** — существенно ниже
  XGBoost/гибрида; порядок чистых GNN чувствителен к sampling/batch.
* **Источник сигнала табличный** (feature-ablation): raw-edge XGBoost 0.289 ≫
  «структура-only» (degree/sum) 0.067 ≈ уровень чистого GNN. Узловые графовые фичи
  дереву даже слегка вредят. См. `results/ibm_family_best.png`, `figures/kr/R03_*`.
* **RQ2 (адаптации):** reverse/port/ego и edge-updates не дают устойчивого
  standalone-прироста ни в одном режиме; знак эффекта меняется между
  with-time / no-time / full-data. `norm_time` при temporal split вреден.
  Лестница «слабый → сильный режим» — `results/ibm_ladder.md`,
  `results/transition_ladder.png`; компактная сводка — `results/ibm_ablation_heatmap.png`.
* **RQ3 (типологии):** XGBoost лидирует по recall на всех 8 паттернах
  (`results/per_pattern.md`). Степенные эвристики не работают: незаконные счета
  **ниже** по степени, чем легитимные (отмывание через низкостепенных «мулов», не хабы).
* **ROC-AUC vs AUC-PR:** ROC-AUC всех GNN высокий (до ~0.96) — сигнал и ранжирование
  есть; провал именно в высокоточном режиме при prevalence ~0.1%.
* **Этап 1 (Elliptic):** сильнейший — XGBoost (AUC-PR 0.80); лучший GNN — GraphSAGE (0.66).

Полный рейтинг и сводная таблица всех вариантов — `results/ibm_all_variants.md`,
`results/kaggle_gpu_seed_summary.csv`.

## Ограничения

* Все финальные результаты — **single-seed** (`seed=42`); устойчивость оценивалась по
  нескольким режимам обучения и архитектурным абляциям, но не по независимым повторам seed.
* Faithful-репродукция Multi-PNA+EU (раздельные AGG_in/out) и GFP-честное усиление дерева —
  вне объёма работы; поэтому утверждение делается узкое: граф полезен как представление в
  гибридной схеме, а не «standalone GNN > XGBoost».
* Порядок чистых GNN между прогонами нестабилен — в тексте без сильных claim о точном
  ранжировании GINe/Multi-GNN/PNA.

## Стек

Python · PyTorch · PyTorch Geometric · XGBoost · scikit-learn · NetworkX · Streamlit ·
Weights & Biases (логирование опционально, по умолчанию выключено — `src.utils.init_wandb`).
