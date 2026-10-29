"""M5 core: hot-cold spatial equilibrium via a gravity allocation and max-flow."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow


def gravity_allocation(A, resistance, gamma, x_total):
    """Allocate x_total by attractiveness and traffic resistance.

    factor_i = (1+resistance_i)^(-gamma); x_i proportional to A_i * factor_i.
    """
    A = np.asarray(A, float)
    factor = (1.0 + np.asarray(resistance, float)) ** (-gamma)
    raw = A * factor
    shares = raw / raw.sum()
    return shares * x_total, shares, factor


def apply_investment(cfg):
    """Apply the cold-point marketing + transit investment; return post states."""
    inv = cfg["investment"]
    n = len(cfg["nodes"])
    A = np.asarray(cfg["attractiveness_baseline"], float)
    A_pot = np.asarray(cfg["attractiveness_potential"], float)
    res = np.asarray(cfg["resistance_baseline"], float)
    cap = np.asarray(cfg["capacity_baseline"], float)
    id_index = {node["id"]: i for i, node in enumerate(cfg["nodes"])}

    # Baseline marketing factor for cold nodes: effective/potential.
    mkt_factor = np.where(A_pot > 0, A / A_pot, 1.0)
    edge_caps = {f"{e['from']}->{e['to']}": e["capacity_baseline"] for e in cfg["edges"]}
    details = []

    for node_id, spend in inv["allocation"].items():
        i = id_index[node_id]
        mkt = spend * inv["marketing_fraction"]
        transit = spend * (1.0 - inv["marketing_fraction"])
        # Marketing raises the awareness factor toward 1.
        mkt_factor[i] = min(1.0, mkt_factor[i] + inv["beta_marketing"] * mkt)
        A[i] = A_pot[i] * mkt_factor[i]
        # Transit lowers resistance and raises annual capacity.
        res[i] = max(0.0, res[i] - inv["beta_resistance"] * transit)
        cap[i] = cap[i] + inv["beta_capacity"] * transit
        # Transit raises the daily road/boat edge capacity.
        for e in cfg["edges"]:
            if e["to"] == node_id:
                key = f"{e['from']}->{e['to']}"
                edge_caps[key] = edge_caps[key] + inv["beta_edge"] * transit
        details.append(dict(node=node_id, spend=spend, marketing=mkt, transit=transit,
                            marketing_factor=float(mkt_factor[i]),
                            attractiveness=float(A[i]), resistance=float(res[i]),
                            capacity=float(cap[i])))
    return A, res, cap, edge_caps, details


def network_max_flow(edge_caps, daily_demand, node_ids):
    """Max flow from the port node through road edges to a super-sink.

    Returns flow value, outgoing demand, and whether it is feasible.
    """
    port = node_ids[0]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    sink = len(node_ids)
    size = sink + 1
    scale = 10000  # scipy maximum_flow requires integer capacities
    row, col, data = [], [], []
    for key, cap in edge_caps.items():
        a, b = key.split("->")
        row.append(idx[a]); col.append(idx[b]); data.append(int(round(cap * scale)))
    outgoing = 0.0
    for nid in node_ids[1:]:
        d = float(daily_demand[idx[nid]])
        row.append(idx[nid]); col.append(sink); data.append(int(round(d * scale)))
        outgoing += d
    graph = sp.csr_matrix((data, (row, col)), shape=(size, size))
    result = maximum_flow(graph, idx[port], sink)
    return float(result.flow_value) / scale, outgoing, abs(float(result.flow_value) / scale - outgoing) < 1e-3


def build_transfer_matrix(x_before, x_after, node_ids):
    """Construct a hot->cold transfer matrix from the net reallocation.

    The gravity model gives net changes only; each hot loss is distributed to the
    cold gains in proportion to their gain. The diagonal is retained visitors.
    """
    delta = np.asarray(x_after) - np.asarray(x_before)
    n = len(node_ids)
    T = np.zeros((n, n))
    gains = np.where(delta > 0, delta, 0.0)
    total_gain = gains.sum()
    for i in range(n):
        if delta[i] < 0 and total_gain > 0:
            for j in range(n):
                if delta[j] > 0:
                    T[i, j] = (-delta[i]) * gains[j] / total_gain
    for i in range(n):
        T[i, i] = min(x_before[i], x_after[i])
    return T
