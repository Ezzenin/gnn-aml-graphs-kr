"""Модели GNN для node-classification на Elliptic.

Линейка базовых архитектур: GCN, GraphSAGE, GAT, GIN, PNA. Все — 2-слойные
message-passing сети с бинарным выходом (out_channels=2: licit/illicit), обучаются
только на размеченных узлах через маску в функции потерь. Сборка через build_model().

Следующий этап: Multi-GNN адаптации (ego-IDs, port numbering, reverse MP) поверх
этих архитектур.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, PNAConv, SAGEConv


class GCN(torch.nn.Module):
    """2-слойный GCN."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(torch.nn.Module):
    """2-слойный GraphSAGE (агрегация по соседям)."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GAT(torch.nn.Module):
    """2-слойный GAT (multi-head attention в первом слое)."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GIN(torch.nn.Module):
    """2-слойный GIN (MLP-агрегатор + BatchNorm)."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.conv1 = GINConv(_mlp(in_channels, hidden_channels), train_eps=True)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = GINConv(_mlp(hidden_channels, hidden_channels), train_eps=True)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class PNA(torch.nn.Module):
    """2-слойный PNA (несколько агрегаторов и скейлеров; требует deg-гистограмму)."""

    AGGREGATORS = ["mean", "min", "max", "std"]
    SCALERS = ["identity", "amplification", "attenuation"]

    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5, deg=None):
        super().__init__()
        if deg is None:
            raise ValueError("PNA требует deg (гистограмму степеней обучающего графа)")
        common = dict(aggregators=self.AGGREGATORS, scalers=self.SCALERS, deg=deg, towers=1)
        self.conv1 = PNAConv(in_channels, hidden_channels, **common)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = PNAConv(hidden_channels, hidden_channels, **common)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


def _mlp(in_dim: int, out_dim: int) -> Sequential:
    return Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))


# ─────────────────────── Edge-classification (IBM AML) ───────────────────────
class EdgeGNN(torch.nn.Module):
    """Message-passing с учётом edge_attr + голова edge-классификации (GINe-база).

    Кодировщики узлов/рёбер → стек GINEConv (учитывает edge_attr) → голова
    MLP([h_u || h_v || e_label]) -> 2 логита для каждого классифицируемого ребра.
    Голова получает И контекст двух счетов (h_u, h_v), И признаки САМОЙ
    классифицируемой транзакции (amount/currency/format/time) через отдельный
    label_edge_enc — иначе модель судит о ребре только по контексту аккаунтов
    (как было до фикса P0.2), что нечестно против табличного XGBoost.
    Адаптации Фазы D:
      - ego_ids / ports — батч-уровневые (в forward), т.к. зависят от подграфа
        мини-батча (сид-ребро и кратность рёбер в выборке);
      - reverse_mp — на уровне ДАННЫХ (add_reverse_edges применяется к графу
        ДО семплинга loader'ом, см. train_edge), поэтому здесь reverse_mp лишь
        информативный флаг: направленческий столбец уже приходит внутри in_edge.
    in_edge_label — размерность СЫРЫХ признаков сид-ребра (без reverse/port
    флагов); по умолчанию = in_edge.
    """

    def __init__(self, in_node, in_edge, hidden=64, num_layers=2, dropout=0.5,
                 reverse_mp=False, ports=False, ego_ids=False, in_edge_label=None,
                 edge_updates=False, conv_type="gine", deg=None):
        super().__init__()
        self.reverse_mp = reverse_mp  # обрабатывается на уровне данных (train_edge)
        self.ports = ports
        self.ego_ids = ego_ids
        self.edge_updates = edge_updates
        self.conv_type = conv_type
        self.dropout = dropout

        node_in = in_node + (1 if ego_ids else 0)
        # reverse-флаг УЖЕ учтён в in_edge (add_reverse_edges на уровне данных).
        edge_in = in_edge + (1 if ports else 0)
        self.node_enc = Linear(node_in, hidden)
        self.edge_enc = Linear(edge_in, hidden)
        # Кодировщик признаков самого классифицируемого ребра (для головы).
        self.label_edge_enc = Linear(in_edge_label if in_edge_label is not None else in_edge, hidden)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        # GIN+EU (Egressy 2024): на каждом слое обновляем и эмбеддинг ребра из
        # [e, h_src, h_dst] — самый дешёвый сильный буст (+19пп в статье), не
        # мультиграфовая адаптация. Резидуально, отдельный MLP на слой.
        self.edge_mlps = torch.nn.ModuleList() if edge_updates else None
        for _ in range(num_layers):
            self.convs.append(self._build_conv(conv_type, hidden, deg))
            self.bns.append(BatchNorm1d(hidden))
            if edge_updates:
                self.edge_mlps.append(_mlp(3 * hidden, hidden))

        self.head = Sequential(
            Linear(3 * hidden, hidden), ReLU(),
            torch.nn.Dropout(dropout), Linear(hidden, 2),
        )

    @staticmethod
    def _build_conv(conv_type, hidden, deg):
        """Один edge-aware слой свёртки: GINEConv (база) или PNAConv (сильный режим).

        Обе свёртки имеют одинаковый интерфейс forward(h, edge_index, e), где e —
        эмбеддинг ребра в размерности hidden (edge_dim=hidden), поэтому остальной
        стек EdgeGNN (адаптации, edge-updates, голова) от conv_type не зависит.
        PNA — главный архитектурный рычаг «сильного режима» (Egressy 2024):
        несколько агрегаторов (mean/min/max/std) + степенные скейлеры; требует
        deg-гистограмму обучающего графа.
        """
        if conv_type == "pna":
            if deg is None:
                raise ValueError("conv_type='pna' требует deg-гистограмму train-графа")
            return PNAConv(hidden, hidden, aggregators=PNA.AGGREGATORS,
                           scalers=PNA.SCALERS, deg=deg, edge_dim=hidden, towers=1)
        if conv_type == "gine":
            return GINEConv(_mlp(hidden, hidden), edge_dim=hidden, train_eps=True)
        raise ValueError(f"Неизвестный conv_type: {conv_type!r} (gine|pna)")

    def forward(self, x, edge_index, edge_attr, edge_label_index, edge_label_attr):
        # ── Батч-уровневые адаптации (Фаза D): ego → port ──
        # (reverse MP применяется к графу на уровне данных, до loader'а.)
        if self.ego_ids:
            # Пометить два конца классифицируемого сид-ребра (локальные индексы
            # в подграфе мини-батча известны из edge_label_index).
            ego = x.new_zeros((x.size(0), 1))
            ego[edge_label_index.reshape(-1)] = 1.0
            x = torch.cat([x, ego], dim=1)
        if self.ports:
            # Порядковый номер ребра среди входящих в узел-получатель
            # (различение кратных рёбер мультиграфа); log1p ограничивает диапазон.
            p = compute_ports(edge_index, x.size(0)).to(edge_attr.dtype)
            edge_attr = torch.cat([edge_attr, torch.log1p(p).unsqueeze(1)], dim=1)

        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)
        src, dst = edge_index[0], edge_index[1]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if self.edge_updates:
                # Обновление ребра из текущих состояний концов (резидуально).
                e = e + self.edge_mlps[i](torch.cat([e, h[src], h[dst]], dim=-1))
            h = F.relu(bn(conv(h, edge_index, e)))
            h = F.dropout(h, p=self.dropout, training=self.training)
        h_u = h[edge_label_index[0]]
        h_v = h[edge_label_index[1]]
        e_label = self.label_edge_enc(edge_label_attr)  # признаки самой транзакции
        return self.head(torch.cat([h_u, h_v, e_label], dim=-1))


def build_edge_model(name: str, in_node: int, in_edge: int, hidden: int = 64,
                     num_layers: int = 2, dropout: float = 0.5,
                     reverse_mp: bool = False, ports: bool = False,
                     ego_ids: bool = False, in_edge_label: Optional[int] = None,
                     edge_updates: bool = False, deg: Optional["torch.Tensor"] = None
                     ) -> torch.nn.Module:
    """Фабрика edge-моделей. name='gine' (C2, база) | 'pna' (сильный режим, Egressy);
    мультиграфовые адаптации и edge-updates — флагами (Фаза D). PNA требует deg."""
    name = name.lower()
    if name in ("gine", "gin", "edgegnn", "pna"):
        conv_type = "pna" if name == "pna" else "gine"
        return EdgeGNN(in_node, in_edge, hidden, num_layers, dropout,
                       reverse_mp=reverse_mp, ports=ports, ego_ids=ego_ids,
                       in_edge_label=in_edge_label, edge_updates=edge_updates,
                       conv_type=conv_type, deg=deg)
    raise ValueError(f"Неизвестная edge-архитектура: {name!r} (gine|pna)")


def build_model(
    name: str,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int = 2,
    dropout: float = 0.5,
    heads: int = 8,
    deg: Optional["torch.Tensor"] = None,
) -> torch.nn.Module:
    """Фабрика моделей по имени архитектуры (gcn|sage|gat|gin|pna)."""
    name = name.lower()
    if name == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, dropout)
    if name in ("sage", "graphsage"):
        return GraphSAGE(in_channels, hidden_channels, out_channels, dropout)
    if name == "gat":
        return GAT(in_channels, hidden_channels, out_channels, dropout, heads=heads)
    if name == "gin":
        return GIN(in_channels, hidden_channels, out_channels, dropout)
    if name == "pna":
        return PNA(in_channels, hidden_channels, out_channels, dropout, deg=deg)
    raise ValueError(f"Неизвестная архитектура: {name!r} (gcn|sage|gat|gin|pna)")


def add_reverse_edges(edge_index, edge_attr):
    """reverse MP: добавить обратные рёбра + бинарный флаг направления.

    Применяется к message-passing графу ДО семплинга LinkNeighborLoader'ом —
    тогда окрестность каждого сид-ребра честно двунаправленная (loader
    семплирует соседей по направлению рёбер; если добавлять reverse уже после
    семплинга, настоящих обратных соседей не появится). Флаг (0=прямое,
    1=обратное) — новый последний столбец edge_attr, сохраняет различение
    направления для message passing.

    Возвращает (edge_index_bi [2, 2E], edge_attr_bi [2E, F+1]).
    """
    n = edge_attr.size(0)
    fwd = torch.cat([edge_attr, edge_attr.new_zeros((n, 1))], dim=1)
    rev = torch.cat([edge_attr, edge_attr.new_ones((n, 1))], dim=1)
    edge_attr_bi = torch.cat([fwd, rev], dim=0)
    edge_index_bi = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index_bi, edge_attr_bi


def compute_ports(edge_index, num_nodes: int) -> "torch.Tensor":
    """Порядковый номер ребра среди входящих в его узел-получатель.

    Различает кратные рёбра мультиграфа: параллельные рёбра в один узел
    получают разные порты 0, 1, 2… Векторно через argsort по dst (стабильный,
    детерминированный). Возвращает LongTensor [E] на устройстве edge_index.
    num_nodes не используется (порты локальны для dst), оставлен для единого
    интерфейса с остальными граф-хелперами.
    """
    dst = edge_index[1]
    if dst.numel() == 0:
        return dst.new_zeros(0)
    order = torch.argsort(dst, stable=True)
    sorted_dst = dst[order]
    _, counts = torch.unique_consecutive(sorted_dst, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(dst.numel(), device=dst.device) - torch.repeat_interleave(starts, counts)
    ports = torch.empty_like(dst)
    ports[order] = within
    return ports


def compute_degree_histogram(edge_index, num_nodes: int) -> "torch.Tensor":
    """Гистограмма входных степеней — нужна PNA-агрегаторам."""
    from torch_geometric.utils import degree

    d = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
    return torch.bincount(d)
