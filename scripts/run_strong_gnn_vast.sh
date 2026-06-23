#!/usr/bin/env bash
# Досчёт «сильной лестницы» на арендованном GPU (vast.ai, A100 PCIe).
# L3 (ibm_multignn_big_fulldata) и L4 (ibm_pna_fulldata) уже посчитаны на Colab —
# здесь только L5/L6 (+ опционально Egressy-репродукция).
#
# Предусловие: данные лежат в data/ibm_aml/ (HI-Small_Trans.csv, HI-Small_Patterns.txt).
# Запуск из корня репозитория:  bash scripts/run_strong_gnn_vast.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "[repo] $(pwd)"

# 1) Зависимости + ОБЯЗАТЕЛЬНЫЙ бэкенд семплинга PyG (иначе NeighborLoader падает).
TORCH=$(python -c "import torch;print(torch.__version__.split('+')[0])")
CU=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu')")
echo "[deps] torch ${TORCH} / ${CU}"
pip -q install torch_geometric xgboost networkx pyyaml
pip -q install pyg-lib torch-scatter torch-sparse -f "https://data.pyg.org/whl/torch-${TORCH}+${CU}.html"
python -c "import torch_sparse, torch_geometric; print('PyG', torch_geometric.__version__, '+ sampler OK')"

# 2) Проверка данных и GPU.
[ -f data/ibm_aml/HI-Small_Trans.csv ]    || { echo "!! нет data/ibm_aml/HI-Small_Trans.csv";    exit 1; }
[ -f data/ibm_aml/HI-Small_Patterns.txt ] || { echo "!! нет data/ibm_aml/HI-Small_Patterns.txt"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# 3) Прогон оставшихся шагов (уже готовые — пропускаются).
RUNS="ibm_multipna_fulldata ibm_multipna_eu_fulldata"
# RUNS="$RUNS ibm_multipna_eu_egressy"   # ← раскомментируй для опц. T5 (протокол Egressy 60/20/20)
for c in $RUNS; do
  if [ -f "results/${c}_metrics.json" ]; then echo "[skip] ${c} уже посчитан"; continue; fi
  echo "========== ${c} =========="
  python -m src.train_edge --config "configs/${c}.yaml"
done

echo "[готово] результаты:"
ls -la results/ibm_multipna_*_metrics.json results/ibm_multipna_*_pr_curve.png 2>/dev/null || true
