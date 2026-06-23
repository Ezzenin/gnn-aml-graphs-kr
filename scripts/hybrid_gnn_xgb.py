"""Гибрид GNN->XGBoost + комплементарность. Два честных теста пользы графа:
(1) сырьё+граф-эмбеддинг > сырьё?  (2) ловит ли GNN позитивы, что XGBoost пропускает?
"""
import json
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.datasets import load_ibm_aml, build_edge_features
from src.models import build_edge_model, add_reverse_edges, compute_degree_histogram
from src.baselines import train_xgboost, predict_scores
from src.metrics import evaluate

MAX_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None
CKPT = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/ibm_gine_fulldata.pt"
OUT_JSON = sys.argv[3] if len(sys.argv) > 3 else "results/ibm_hybrid_gnn_xgb_metrics.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}", flush=True)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["config"]; mp = cfg["model"]["params"]
rev = bool(mp.get("reverse_mp", False)); nbr = list(cfg["train"]["num_neighbors"]); bs = 4096
print(f"[ckpt] {CKPT} rev={rev} nbr={nbr}", flush=True)

data, meta = load_ibm_aml(root="data/ibm_aml", variant="HI-Small", include_time=False, max_rows=MAX_ROWS)
y = data.edge_label.numpy()
tr, va, te = data.train_mask.numpy(), data.val_mask.numpy(), data.test_mask.numpy()
print(f"[data] edges={data.num_edges:,} pos tr/va/te={int(y[tr].sum())}/{int(y[va].sum())}/{int(y[te].sum())}", flush=True)

def ctx_of(mask):
    ei = data.edge_index[:, mask]; ea = data.edge_attr[mask].float()
    if rev: ei, ea = add_reverse_edges(ei, ea)
    d = Data(x=data.x.float(), edge_index=ei, edge_attr=ea); d.num_nodes = data.num_nodes
    return d
train_ctx = ctx_of(tr); trainval_ctx = ctx_of(tr | va)

model_type = cfg.get("model", {}).get("type", "gine")
deg = compute_degree_histogram(train_ctx.edge_index, train_ctx.num_nodes) if model_type == "pna" else None
model = build_edge_model(model_type, in_node=ck["in_node"], in_edge=ck["in_edge"], in_edge_label=ck["in_edge_label"],
                         hidden=int(mp.get("hidden", 64)), num_layers=int(mp.get("num_layers", 2)),
                         dropout=float(mp.get("dropout", 0.5)), reverse_mp=rev,
                         ports=bool(mp.get("ports", False)), ego_ids=bool(mp.get("ego_ids", False)),
                         edge_updates=bool(mp.get("edge_updates", False)), deg=deg)
model.load_state_dict(ck["state_dict"]); model.eval().to(device)
store = {}
model.head.register_forward_pre_hook(lambda m, inp: store.__setitem__("e", inp[0].detach()))
EMB = 3 * int(mp.get("hidden", 64))

def extract(ctx, idx):
    seed_attr = data.edge_attr[idx].float()
    loader = LinkNeighborLoader(ctx, num_neighbors=nbr, edge_label_index=data.edge_index[:, idx],
        edge_label=data.edge_label[idx], batch_size=bs, shuffle=False, num_workers=0)
    emb = np.zeros((len(idx), EMB), dtype=np.float32); sc = np.zeros(len(idx), dtype=np.float32)
    with torch.no_grad():
        for j, b in enumerate(loader):
            b = b.to(device)
            ela = seed_attr[b.input_id.cpu()].to(device)
            out = model(b.x, b.edge_index, b.edge_attr, b.edge_label_index, ela)
            ii = b.input_id.cpu().numpy()
            emb[ii] = store["e"].cpu().numpy()
            sc[ii] = torch.softmax(out, 1)[:, 1].cpu().numpy()
            if j % 200 == 0: print(f"   batch {j}", flush=True)
    return emb, sc

print("[emb] извлекаю эмбеддинги/скоры GNN (анти-утечка контексты)...", flush=True)
idx_tr, idx_va, idx_te = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)
emb = np.zeros((data.num_edges, EMB), dtype=np.float32); gsc = np.zeros(data.num_edges, dtype=np.float32)
for nm, ctx, idx in [("train", train_ctx, idx_tr), ("val", train_ctx, idx_va), ("test", trainval_ctx, idx_te)]:
    e, s = extract(ctx, idx); emb[idx] = e; gsc[idx] = s; print(f"[emb] {nm} готов", flush=True)

X = build_edge_features(data, fan_features=False).astype(np.float32)
ea = data.edge_attr.shape[1]; nd = data.x.shape[1]
RAW = list(range(0, ea)) + [ea + 2 * nd]
params = dict(n_estimators=400, max_depth=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, n_jobs=0, random_state=42)

def run(Xmat, tag):
    m = train_xgboost(Xmat[tr], y[tr], Xmat[va], y[va], params)
    sv = predict_scores(m, Xmat[va])
    vm = evaluate(y[va], sv, None)
    thr = vm["threshold"]
    sx = predict_scores(m, Xmat[te])
    tm = evaluate(y[te], sx, thr)
    print(f"  {tag:40s} AUC-PR {tm['auc_pr']:.4f} | F1 {tm['f1']:.4f} | ROC {tm['roc_auc']:.3f}", flush=True)
    return {"val_metrics": vm, "test_metrics": tm, "threshold": thr}, sx

print("\n=== ТЕСТ 1: ГИБРИД GNN->XGBoost ===", flush=True)
raw_result, sx_raw = run(X[:, RAW], "сырьё ребра (raw)")
full_result, _ = run(X, "полный XGBoost (raw+node+parallel)")
hybrid_result, _ = run(np.hstack([X[:, RAW], emb]), "сырьё + GNN-эмбеддинг (ГИБРИД)")
delta = hybrid_result["test_metrics"]["auc_pr"] - raw_result["test_metrics"]["auc_pr"]
print(f"  Δ гибрид-сырьё = {delta:+.4f}  ({'>>> ГРАФ ПОМОГ' if delta > 0 else 'граф не помог'})", flush=True)

print("\n=== ТЕСТ 2: КОМПЛЕМЕНТАРНОСТЬ (ловит ли GNN то, что XGBoost пропускает) ===", flush=True)
yt = y[te]; sg = gsc[idx_te]
rx = (-sx_raw).argsort(); rg = (-sg).argsort()
P = int(yt.sum())
for N in [500, 1000, 2000, 5000]:
    top_x = set(rx[:N].tolist()); top_g = set(rg[:N].tolist())
    pos = set(np.flatnonzero(yt == 1).tolist())
    cx = len(top_x & pos); cg = len(top_g & pos); cu = len((top_x | top_g) & pos)
    only_g = len((top_g & pos) - top_x)
    # ансамбль по среднему рангу при ТОМ ЖЕ бюджете N
    rank_x = np.empty(len(yt)); rank_x[rx] = np.arange(len(yt))
    rank_g = np.empty(len(yt)); rank_g[rg] = np.arange(len(yt))
    ens = (rank_x + rank_g).argsort()
    ce = len(set(ens[:N].tolist()) & pos)
    print(f"  budget N={N:5d}: XGB ловит {cx:4d}/{P} | ансамбль(XGB+GNN) {ce:4d} | только-GNN сверх XGB +{only_g}", flush=True)

result = {
    "_note": "Гибрид GNN->XGBoost: эмбеддинг ребра GNN ([h_u||h_v||e_label], 192d) доп-фичами в XGBoost.",
    "model_type": "hybrid_gnn_xgboost",
    "result_status": "full_metrics_from_script",
    "config": {
        "dataset": {"name": "ibm_aml", "variant": "HI-Small", "include_time": False, "max_rows": MAX_ROWS},
        "checkpoint": CKPT,
        "output_name": "ibm_hybrid_gnn_xgb",
        "xgboost_params": params,
    },
    "anti_leakage_audit": {
        "train_embeddings_context": "train edges only",
        "val_embeddings_context": "train edges only",
        "test_embeddings_context": "train+val edges only",
        "seed_edge_attr_source": "raw data.edge_attr for classified seed edge",
        "xgboost_fit": "train mask only",
        "threshold": "selected on val, applied to test",
    },
    "dataset_meta": meta,
    "comparison": {
        "xgboost_raw_edge": raw_result["test_metrics"],
        "xgboost_full": full_result["test_metrics"],
        "hybrid_embedding_xgb": hybrid_result["test_metrics"] | {"delta_vs_raw": delta},
    },
    "fixed_threshold_from_val": hybrid_result["threshold"],
    "val_metrics": hybrid_result["val_metrics"],
    "test_metrics": hybrid_result["test_metrics"],
}
os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"[saved] {OUT_JSON}", flush=True)
print("\n[готово]", flush=True)
