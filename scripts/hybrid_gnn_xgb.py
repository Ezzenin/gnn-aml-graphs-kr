"""Гибрид GNN->XGBoost + комплементарность. Два честных теста пользы графа:
(1) сырьё+граф-эмбеддинг > сырьё?  (2) ловит ли GNN позитивы, что XGBoost пропускает?
"""
import sys
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.datasets import load_ibm_aml, build_edge_features
from src.models import build_edge_model, add_reverse_edges
from src.baselines import train_xgboost, predict_scores
from src.metrics import evaluate

MAX_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None
CKPT = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/ibm_gine_fulldata.pt"
device = "cpu"

ck = torch.load(CKPT, map_location=device, weights_only=False)
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

model = build_edge_model("gine", in_node=ck["in_node"], in_edge=ck["in_edge"], in_edge_label=ck["in_edge_label"],
                         hidden=int(mp.get("hidden", 64)), num_layers=int(mp.get("num_layers", 2)),
                         dropout=float(mp.get("dropout", 0.5)), reverse_mp=rev,
                         ports=bool(mp.get("ports", False)), ego_ids=bool(mp.get("ego_ids", False)))
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
    thr = evaluate(y[va], predict_scores(m, Xmat[va]), None)["threshold"]
    sx = predict_scores(m, Xmat[te]); tm = evaluate(y[te], sx, thr)
    print(f"  {tag:40s} AUC-PR {tm['auc_pr']:.4f} | F1 {tm['f1']:.4f} | ROC {tm['roc_auc']:.3f}", flush=True)
    return tm["auc_pr"], sx

print("\n=== ТЕСТ 1: ГИБРИД GNN->XGBoost ===", flush=True)
a, sx_raw = run(X[:, RAW], "сырьё ребра (raw)")
b, _ = run(np.hstack([X[:, RAW], emb]), "сырьё + GNN-эмбеддинг (ГИБРИД)")
print(f"  Δ гибрид-сырьё = {b - a:+.4f}  ({'>>> ГРАФ ПОМОГ' if b > a else 'граф не помог'})", flush=True)

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
print("\n[готово]", flush=True)
