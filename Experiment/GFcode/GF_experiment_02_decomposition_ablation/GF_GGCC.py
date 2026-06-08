import os
import time
import json
import math
import random
import threading
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

import pulp
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================
# 0) Paths
# =========================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
_DATA_DIR = "/work/home/acn1eb8nfq/yuezhang46/GFdata"
EDGE_DIR = "/work/home/acn1eb8nfq/yuezhang46/GFdata/Edge"
NODE_PATH = "/work/home/acn1eb8nfq/yuezhang46/GFdata/Node/candidate_stations.csv"
SCENARIO_FILES = {
    "spring": os.path.join(EDGE_DIR, "spring__weighted_edges.csv"),
    "summer": os.path.join(EDGE_DIR, "summer__weighted_edges.csv"),
    "autumn": os.path.join(EDGE_DIR, "autumn__weighted_edges.csv"),
    "winter": os.path.join(EDGE_DIR, "winter__weighted_edges.csv"),
}

SCENARIO_DAYS = {"spring": 31, "summer": 30, "autumn": 31, "winter": 30}

OUT_PLAN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcpcc_station_plan_CN.csv")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "gcpcc_logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _resolve_seed(run_id: int, seed: Optional[int]) -> int:
    return int(seed) if seed is not None else 1000 + int(run_id)


def _build_run_output_dir(output_root: str, run_id: int, seed: int) -> str:
    run_output_dir = os.path.join(output_root, f"run_{run_id:02d}_seed_{seed}")
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir


# =========================
# 1) Binary segment encoding (paper one-hot segment, L=3)
# =========================
SEG_LEN = 3  # L=3 for S/M/L

# scale_id: 0=NONE, 1=S, 2=M, 3=L
SCALE_LABEL = {0: "NONE", 1: "S", 2: "M", 3: "L"}

# binary segments for each scale_id (NONE is all-zero)
SCALE_ID_TO_SEG: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),  # NONE
    1: (1, 0, 0),  # S
    2: (0, 1, 0),  # M
    3: (0, 0, 1),  # L
}
SEG_TO_SCALE_ID: Dict[Tuple[int, int, int], int] = {v: k for k, v in SCALE_ID_TO_SEG.items()}


def seg_to_id(seg: Tuple[int, int, int]) -> int:
    return SEG_TO_SCALE_ID.get(tuple(seg), 0)


def id_to_seg(scale_id: int) -> Tuple[int, int, int]:
    return SCALE_ID_TO_SEG[int(scale_id)]


def seg_to_str(seg: Tuple[int, int, int]) -> str:
    return "".join(str(int(b)) for b in seg)


L_MAX = 3  # number of real station scales (S/M/L)

# sessions/day (capacity)
C_sess_day = {0: 0, 1: 256, 2: 512, 3: 768}
# build cost (USD)
build_cost = {0: 0, 1: 89992, 2: 179984, 3: 269976}
# op cost (USD/yr)
op_cost_yr = {0: 0, 1: 6299, 2: 12599, 3: 18898}


# =========================
# 2) Knobs
# =========================
MAX_K_SPECTRAL = 10

# Unified hard cap by generation.
MAX_OUTER_GEN = 50

GA_POP = 16
GA_GEN = 16

TOURNAMENT_K = 4
P_CROSS = 0.5

P_MUT_MIN = 0.12
P_MUT_MAX = 0.22

BETA_PENALTY = 1e3

SAVE_EVERY_FE = 1000
SAVE_EVERY_OUTER_GEN = 1

PARALLEL_WORKERS = min(
    3,
    max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
)

# ---- Preprocess threshold ----
EDGE_WEIGHT_MIN = 1.0

# ---- Spatial fragment reassignment (post Spatial-Cut) ----
REASSIGN_TOPK = 20

# ---- Consensus Cut (Version A) ----
CONSENSUS_MIN_AGREE = 3  # 3/4. 4 is strict.

# ---- Optional: merge tiny consensus components as whole comps ----
CONSENSUS_MERGE_SMALL = True
CONSENSUS_SMALL_ABS = 30
CONSENSUS_SMALL_FRAC = 0.15
CONSENSUS_REASSIGN_TOPK = 50

# ---- Classical Differential Grouping (static) ----
DG_EPS_ABS = 1e-6
DG_EPS_REL = 1e-6

# ---- Ablation decomposition modes ----
DECOMP_MODE_ORIG_CONSENSUS_RR = "orig_consensus_rr"
DECOMP_MODE_SPATIAL_ONLY_RR = "spatial_only_rr"
DECOMP_MODE_TOPO_CC_RR = "topo_cc_rr"
DECOMP_MODE_SINGLE_COMPONENT = "single_component"
DECOMP_MODE_RANDOM_GROUPING_RR = "random_grouping_rr"
DECOMP_MODE_GEO_GROUPING_RR = "geo_grouping_rr"
DECOMP_MODE_DG_STATIC_RR = "dg_static_rr"
DECOMP_MODES = {
    DECOMP_MODE_ORIG_CONSENSUS_RR,
    DECOMP_MODE_SPATIAL_ONLY_RR,
    DECOMP_MODE_TOPO_CC_RR,
    DECOMP_MODE_SINGLE_COMPONENT,
    DECOMP_MODE_RANDOM_GROUPING_RR,
    DECOMP_MODE_GEO_GROUPING_RR,
    DECOMP_MODE_DG_STATIC_RR,
}
DEFAULT_DECOMP_MODE = DECOMP_MODE_GEO_GROUPING_RR


# =========================
# 3) Data structure
# =========================
@dataclass
class ScenarioGraph:
    name: str
    nodes: np.ndarray
    id2idx: Dict[int, int]
    edges: pd.DataFrame
    W: csr_matrix


# =========================
# 4) RunLogger
# =========================
class RunLogger:
    def __init__(self, run_id: int, save_every_fe: int = 1000, save_every_gen: int = 1):
        self.run_id = run_id
        self.save_every_fe = save_every_fe
        self.save_every_gen = save_every_gen
        self.fe = 0

        self.best_local_obj = float("inf")
        self.best_local_cost = float("inf")
        self.best_local_viol = float("inf")

        self.best_global_obj = float("inf")
        self.best_global_cost = float("inf")
        self.best_global_viol = float("inf")
        self.latest_eval_obj = float("inf")
        self.latest_eval_cost = float("inf")
        self.latest_eval_viol = float("inf")

        self.stop_reason = "not_set"
        self.stop_outer_gen = 0

        self.cur_outer_gen = 0
        self.curve_gen = []

        self._t0 = None
        self._t0_cpu = None
        self.wall_time_sec = None
        self.cpu_time_sec = None

        self._lock = threading.Lock()

    def start(self):
        self._t0 = time.perf_counter()
        self._t0_cpu = time.process_time()

    def end(self):
        self.wall_time_sec = time.perf_counter() - self._t0
        self.cpu_time_sec = time.process_time() - self._t0_cpu

    def set_outer_gen(self, g: int):
        self.cur_outer_gen = g

    def update_local_fe(self, obj: float, cost: float, viol: float):
        with self._lock:
            self.latest_eval_obj = float(obj)
            self.latest_eval_cost = float(cost)
            self.latest_eval_viol = float(viol)
            if obj < self.best_local_obj:
                self.best_local_obj = obj
                self.best_local_cost = cost
                self.best_local_viol = viol

            if obj < self.best_global_obj:
                self.best_global_obj = obj
                self.best_global_cost = cost
                self.best_global_viol = viol

    def add_fe(self, delta_fe: int):
        with self._lock:
            self.fe += int(max(0, delta_fe))

    def record_global_gen_point(self, global_obj: float, global_cost: float, global_viol: float):
        with self._lock:
            if global_obj < self.best_global_obj:
                self.best_global_obj = global_obj
                self.best_global_cost = global_cost
                self.best_global_viol = global_viol

            self.curve_gen.append({
                "run": self.run_id,
                "outer_gen": self.cur_outer_gen,
                "fe": self.fe,
                "global_obj": global_obj,
                "global_cost": global_cost,
                "global_viol": global_viol,
                "best_global_obj": self.best_global_obj,
                "best_global_cost": self.best_global_cost,
                "best_global_viol": self.best_global_viol,
            })

    def set_stop(self, reason: str, outer_gen: int):
        with self._lock:
            self.stop_reason = str(reason)
            self.stop_outer_gen = int(outer_gen)

    def dump_run_logs(self, log_dir: str):
        legacy_fe_path = os.path.join(log_dir, f"run_{self.run_id:02d}_curve_fe.jsonl")
        if os.path.exists(legacy_fe_path):
            os.remove(legacy_fe_path)
        gen_path = os.path.join(log_dir, f"run_{self.run_id:02d}_curve_gen.jsonl")
        with open(gen_path, "w", encoding="utf-8") as f:
            for row in self.curve_gen:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return gen_path


# =========================
# 5) IO helpers
# =========================
def read_nodes(node_path: str) -> pd.DataFrame:
    df = pd.read_csv(node_path)
    required = {"id", "longitude", "latitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"candidate_stations.csv missing columns: {missing}")
    df = df.sort_values("id").reset_index(drop=True)
    df["id"] = df["id"].astype(int)
    df["longitude"] = df["longitude"].astype(float)
    df["latitude"] = df["latitude"].astype(float)
    return df


def read_edges(edge_path: str) -> pd.DataFrame:
    df = pd.read_csv(edge_path)
    required = {"u", "v", "distance_km", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{edge_path} missing columns: {missing}")
    df["u"] = df["u"].astype(int)
    df["v"] = df["v"].astype(int)
    df["weight"] = df["weight"].astype(float)
    df["distance_km"] = df["distance_km"].astype(float)

    uv_min = np.minimum(df["u"].values, df["v"].values)
    uv_max = np.maximum(df["u"].values, df["v"].values)
    df["_a"], df["_b"] = uv_min, uv_max
    df = df.drop_duplicates(subset=["_a", "_b"]).drop(columns=["_a", "_b"]).reset_index(drop=True)
    return df


def build_symmetric_W(nodes: np.ndarray, edges: pd.DataFrame) -> csr_matrix:
    n = len(nodes)
    id2idx = {int(nid): i for i, nid in enumerate(nodes.tolist())}
    rows, cols, vals = [], [], []
    for _, r in edges.iterrows():
        u, v, w = int(r["u"]), int(r["v"]), float(r["weight"])
        if u not in id2idx or v not in id2idx:
            continue
        iu, iv = id2idx[u], id2idx[v]
        if iu == iv:
            continue
        rows.extend([iu, iv])
        cols.extend([iv, iu])
        vals.extend([w, w])
    W = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    W.sum_duplicates()
    return W


def filter_edges_and_get_active_nodes(
    all_nodes: List[int],
    edges_by_scenario: Dict[str, pd.DataFrame],
    w_min: float = 1.0
) -> Tuple[Dict[str, pd.DataFrame], List[int], List[int]]:
    """
    1) Filter each scenario edges: keep weight >= w_min
    2) Active nodes: nodes with at least one incident kept edge across all scenarios
    3) Inactive nodes: otherwise (fixed NONE, excluded from solve)
    """
    all_set = set(int(x) for x in all_nodes)

    filtered: Dict[str, pd.DataFrame] = {}
    incident = {int(x): 0 for x in all_nodes}

    for sname, edf in edges_by_scenario.items():
        keep = edf[(edf["weight"] >= float(w_min)) &
                   (edf["u"].isin(all_set)) &
                   (edf["v"].isin(all_set))].copy()
        keep.reset_index(drop=True, inplace=True)
        filtered[sname] = keep

        for u, v in zip(keep["u"].astype(int).values, keep["v"].astype(int).values):
            incident[int(u)] += 1
            incident[int(v)] += 1

    active_nodes = sorted([nid for nid, deg in incident.items() if deg > 0])
    inactive_nodes = sorted([nid for nid, deg in incident.items() if deg == 0])

    return filtered, active_nodes, inactive_nodes


# =========================
# 6) Spatial cut (spectral clustering)
# =========================
def choose_k_by_eigengap(evals_L: np.ndarray, k_min: int = 2, k_max: int = 20) -> int:
    r = int(len(evals_L))
    if r < 3:
        return max(2, min(k_max, r - 1))
    upper = min(int(k_max), r - 1)
    if upper < k_min:
        return max(2, min(k_max, r - 1))

    best_k, best_gap = k_min, -1.0
    for ell in range(k_min, upper + 1):
        gap = float(evals_L[ell] - evals_L[ell - 1])
        if gap > best_gap:
            best_gap = gap
            best_k = ell
    return int(best_k)


def spatial_cut(W: csr_matrix, node_ids: np.ndarray, max_k: int = 20) -> List[List[int]]:
    n = W.shape[0]
    if n <= 2:
        return [[int(x)] for x in node_ids.tolist()]

    d = np.array(W.sum(axis=1)).reshape(-1)
    d = np.maximum(d, 1e-9)
    D_inv_sqrt = 1.0 / np.sqrt(d)

    A = W.copy().astype(np.float64)
    A = A.multiply(D_inv_sqrt[:, None])
    A = A.multiply(D_inv_sqrt[None, :])

    k_eigs = min(max_k + 1, n - 1)
    if k_eigs < 3:
        return [[int(x)] for x in node_ids.tolist()]

    try:
        evals_A, evecs_A = eigsh(A, k=k_eigs, which="LA")
        order = np.argsort(-evals_A)
        evals_A = evals_A[order]
        evecs_A = evecs_A[:, order]
    except Exception:
        A_dense = A.toarray()
        evals_A, evecs_A = np.linalg.eigh(A_dense)
        order = np.argsort(-evals_A)
        evals_A = evals_A[order]
        evecs_A = evecs_A[:, order]

    evals_L = np.sort(1.0 - evals_A)
    k_star = choose_k_by_eigengap(evals_L[:k_eigs], k_min=2, k_max=min(max_k, k_eigs - 1))

    U = evecs_A[:, 1:k_star + 1]  # skip trivial eigenvector
    U_hat = U / (np.linalg.norm(U, axis=1)[:, None] + 1e-12)

    km = KMeans(n_clusters=k_star, n_init=10, random_state=42)
    labels = km.fit_predict(U_hat)

    clusters = []
    for c in range(k_star):
        idxs = np.where(labels == c)[0]
        clusters.append([int(node_ids[i]) for i in idxs])
    return clusters


# =========================
# 6.5) Spatial fragment repair (KDTree reassignment)
# =========================
def repair_spatial_fragments(
    sg: ScenarioGraph,
    clusters: List[List[int]],
    kd_tree: KDTree,
    coords: np.ndarray,
    id_list: np.ndarray,
    id2posidx: Dict[int, int],
    topk: int = 20,
) -> Tuple[List[List[int]], Dict[int, int], set]:
    labels: Dict[int, int] = {}
    for ci, nodes in enumerate(clusters):
        for nid in nodes:
            labels[int(nid)] = int(ci)

    Gs = nx.Graph()
    Gs.add_nodes_from([int(x) for x in sg.nodes.tolist()])
    if len(sg.edges) > 0:
        Gs.add_edges_from(list(zip(sg.edges["u"].astype(int).values, sg.edges["v"].astype(int).values)))

    fragment_nodes: set = set()

    for ci, nodes in enumerate(clusters):
        if not nodes or len(nodes) <= 1:
            continue
        subG = Gs.subgraph(nodes)
        comps = list(nx.connected_components(subG))
        if not comps:
            continue
        comps.sort(key=len, reverse=True)
        for comp in comps[1:]:
            fragment_nodes.update(int(x) for x in comp)

    k_eff = int(min(max(1, topk), len(id_list))) if len(id_list) > 0 else 1
    for frag in fragment_nodes:
        if frag not in id2posidx:
            continue
        if frag not in labels:
            continue

        orig_label = labels[frag]
        frag_coord = coords[id2posidx[frag]]

        _dists, neighbors = kd_tree.query(frag_coord, k=k_eff)
        neighbors = np.atleast_1d(neighbors)

        new_label = None
        for nb_idx in neighbors:
            nb_id = int(id_list[int(nb_idx)])
            if nb_id == frag:
                continue
            if nb_id in fragment_nodes:
                continue
            if nb_id not in labels:
                continue
            if labels[nb_id] == orig_label:
                continue
            new_label = labels[nb_id]
            break

        if new_label is not None:
            labels[frag] = int(new_label)

    uniq = sorted(set(labels.values())) if labels else [0]
    old2new = {old: i for i, old in enumerate(uniq)}
    labels2 = {int(nid): int(old2new[lab]) for nid, lab in labels.items()}

    clusters2 = [[] for _ in range(len(uniq))]
    for nid, lab in labels2.items():
        clusters2[int(lab)].append(int(nid))
    for c in clusters2:
        c.sort()

    return clusters2, labels2, fragment_nodes


# =========================
# 7) Consensus cut (Version A)
# =========================
def consensus_cut_vA(
    scenario_node_labels: Dict[str, Dict[int, int]],
    union_edges: pd.DataFrame,
    active_nodes: List[int],
    min_agree: int = 3
) -> List[List[int]]:
    seasons = list(scenario_node_labels.keys())
    m = int(max(1, min(int(min_agree), len(seasons))))

    Gc = nx.Graph()
    Gc.add_nodes_from([int(x) for x in active_nodes])

    if len(union_edges) == 0:
        comps = [sorted([int(x)]) for x in active_nodes]
        comps.sort(key=lambda c: (-len(c), c[0]))
        return comps

    for _, r in union_edges.iterrows():
        u, v = int(r["u"]), int(r["v"])
        agree = 0
        for s in seasons:
            lab = scenario_node_labels[s]
            if lab.get(u, None) is None or lab.get(v, None) is None:
                continue
            if lab[u] == lab[v]:
                agree += 1
        if agree >= m:
            Gc.add_edge(u, v)

    comps = [sorted(list(comp)) for comp in nx.connected_components(Gc)]
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def merge_small_components_to_nearest_component(
    components: List[List[int]],
    node_df_active: pd.DataFrame,
    small_abs: int = 30,
    small_frac: float = 0.15,
    topk: int = 50
) -> List[List[int]]:
    if not components:
        return components

    id2coord = {
        int(r["id"]): (float(r["longitude"]), float(r["latitude"]))
        for _, r in node_df_active.iterrows()
    }

    sizes = [len(c) for c in components]
    med = float(np.median(sizes)) if sizes else 0.0
    thr = int(max(int(small_abs), int(med * float(small_frac))))
    thr = max(1, thr)

    labels: Dict[int, int] = {}
    for ci, comp in enumerate(components):
        for nid in comp:
            labels[int(nid)] = int(ci)

    small_idx = [ci for ci, comp in enumerate(components) if len(comp) < thr]
    big_idx = [ci for ci, comp in enumerate(components) if len(comp) >= thr]
    if not small_idx or not big_idx:
        return components

    big_nodes, big_coords = [], []
    for ci in big_idx:
        for nid in components[ci]:
            if int(nid) in id2coord:
                big_nodes.append(int(nid))
                big_coords.append(id2coord[int(nid)])

    if not big_nodes:
        return components

    big_nodes = np.array(big_nodes, dtype=int)
    big_coords = np.array(big_coords, dtype=float)
    kd = KDTree(big_coords)
    k_eff = int(min(max(1, topk), len(big_nodes)))

    merged_cnt = 0
    for ci_small in small_idx:
        comp_nodes = components[ci_small]
        coords = [id2coord[n] for n in comp_nodes if n in id2coord]
        if not coords:
            continue

        cx = float(np.mean([p[0] for p in coords]))
        cy = float(np.mean([p[1] for p in coords]))

        _d, neigh = kd.query((cx, cy), k=k_eff)
        neigh = np.atleast_1d(neigh)

        target_label = None
        for nb_i in neigh:
            nb_id = int(big_nodes[int(nb_i)])
            tl = labels.get(nb_id, None)
            if tl is not None and tl in big_idx:
                target_label = tl
                break
        if target_label is None:
            continue

        for nid in comp_nodes:
            labels[int(nid)] = int(target_label)
        merged_cnt += 1

    uniq = sorted(set(labels.values()))
    old2new = {old: i for i, old in enumerate(uniq)}
    labels2 = {nid: old2new[lab] for nid, lab in labels.items()}

    new_comps = [[] for _ in range(len(uniq))]
    for nid, ci in labels2.items():
        new_comps[int(ci)].append(int(nid))

    new_comps = [sorted(c) for c in new_comps if len(c) > 0]
    new_comps.sort(key=lambda c: (-len(c), c[0]))

    print(f"[ConsensusMerge] med={int(med)} thr={thr} merged={merged_cnt}", flush=True)
    return new_comps


# =========================
# 8) Dependency graph + coloring
# =========================
def build_dependency_graph(components: List[List[int]], all_edges_union: pd.DataFrame) -> nx.Graph:
    node2comp = {}
    for ci, nodes in enumerate(components):
        for nid in nodes:
            node2comp[int(nid)] = ci

    H = nx.Graph()
    H.add_nodes_from(range(len(components)))

    for _, r in all_edges_union.iterrows():
        u, v = int(r["u"]), int(r["v"])
        if u not in node2comp or v not in node2comp:
            continue
        cu, cv = node2comp[u], node2comp[v]
        if cu != cv:
            H.add_edge(cu, cv)
    return H


def greedy_coloring_groups(H: nx.Graph) -> Dict[int, List[int]]:
    color_map = nx.coloring.greedy_color(H, strategy="largest_first")
    groups = {}
    for comp_idx, col in color_map.items():
        groups.setdefault(col, []).append(comp_idx)
    for col in groups:
        groups[col] = sorted(groups[col])
    return groups


# =========================
# 9) Neighborhood nodes (for local LP)
# =========================
def build_component_neighborhood_nodes(comp_nodes: List[int], edges_by_scenario: Dict[str, pd.DataFrame]) -> List[int]:
    comp_set = set(int(x) for x in comp_nodes)
    neigh = set(comp_set)
    for edf in edges_by_scenario.values():
        sub = edf[(edf["u"].isin(comp_set)) | (edf["v"].isin(comp_set))]
        for _, r in sub.iterrows():
            neigh.add(int(r["u"]))
            neigh.add(int(r["v"]))
    return sorted(neigh)


# =========================
# 10) Context cache (residual capacity)
# =========================
def build_context_allocation_cache(
    all_nodes: List[int],
    edges_by_scenario: Dict[str, pd.DataFrame],
    x_plan: Dict[int, Tuple[int, int, int]],
) -> dict:
    node_set = set(int(i) for i in all_nodes)
    y = {int(i): (1 if seg_to_id(x_plan[int(i)]) != 0 else 0) for i in all_nodes}

    cache = {
        "node_total_load": {},  # {scenario -> {node -> total load}}
        "edge_contrib": {},     # {scenario -> {edge_idx -> (u_load, v_load)}}
    }

    for tname, edf in edges_by_scenario.items():
        days_t = SCENARIO_DAYS[tname]
        cap = {int(i): C_sess_day[seg_to_id(x_plan[int(i)])] * days_t for i in all_nodes}

        sub_edges = edf[edf["u"].isin(node_set) & edf["v"].isin(node_set)]
        prob = pulp.LpProblem(f"TDPCS_context_{tname}", pulp.LpMinimize)

        o_vars = {idx: pulp.LpVariable(f"o_{idx}", lowBound=0.0, upBound=1.0, cat="Continuous")
                  for idx in sub_edges.index}
        s_vars = {nid: pulp.LpVariable(f"s_{nid}", lowBound=0.0, cat="Continuous")
                  for nid in all_nodes}

        prob += pulp.lpSum([s_vars[nid] for nid in all_nodes])

        out_edges = {int(nid): [] for nid in all_nodes}
        in_edges = {int(nid): [] for nid in all_nodes}

        for idx, r in sub_edges.iterrows():
            u, v, w = int(r["u"]), int(r["v"]), float(r["weight"])
            if w <= 0:
                continue
            out_edges[u].append((idx, w))
            in_edges[v].append((idx, w))

        for nid in all_nodes:
            nid = int(nid)
            lhs_terms = []
            for idx, w in out_edges[nid]:
                lhs_terms.append(o_vars[idx] * w)
            for idx, w in in_edges[nid]:
                lhs_terms.append((1.0 - o_vars[idx]) * w)
            prob += pulp.lpSum(lhs_terms) <= cap[nid] + s_vars[nid]

        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception:
            prob.solve()

        node_total = {int(nid): 0.0 for nid in all_nodes}
        edge_contrib = {}

        for idx, r in sub_edges.iterrows():
            u, v, w = int(r["u"]), int(r["v"]), float(r["weight"])
            if w <= 0:
                continue
            o_val = pulp.value(o_vars[idx])
            if o_val is None:
                o_val = 0.5
            o_val = float(o_val)
            u_load = o_val * w
            v_load = (1.0 - o_val) * w
            edge_contrib[int(idx)] = (float(u_load), float(v_load))
            if u in node_total:
                node_total[u] += float(u_load)
            if v in node_total:
                node_total[v] += float(v_load)

        cache["node_total_load"][tname] = node_total
        cache["edge_contrib"][tname] = edge_contrib

    return cache


# =========================
# 11) Local LP feasibility (slack + memo)
#     residual is not truncated
# =========================
def lp_violation_local_with_memo(
    neighborhood_nodes: List[int],
    core_nodes: List[int],
    edges_by_scenario: Dict[str, pd.DataFrame],
    x_plan: Dict[int, Tuple[int, int, int]],
    memo: dict,
    context_cache: dict
) -> float:
    key = (
        tuple(int(n) for n in core_nodes),
        tuple((int(nid), x_plan[int(nid)]) for nid in neighborhood_nodes)
    )
    if key in memo:
        return memo[key]

    node_ids = np.array(neighborhood_nodes, dtype=int)
    node_set = set(int(x) for x in neighborhood_nodes)
    core_set = set(int(x) for x in core_nodes)

    y = {}
    for i in node_ids.tolist():
        sid = seg_to_id(x_plan[int(i)])
        y[int(i)] = 1 if sid != 0 else 0

    total_slack = 0.0

    for tname, edf in edges_by_scenario.items():
        days_t = SCENARIO_DAYS[tname]

        cap_full = {}
        for i in node_ids.tolist():
            sid = seg_to_id(x_plan[int(i)])
            cap_full[int(i)] = C_sess_day[sid] * days_t

        mask = (
            (edf["u"].isin(core_set) & edf["v"].isin(node_set)) |
            (edf["v"].isin(core_set) & edf["u"].isin(node_set))
        )
        sub_edges = edf.loc[mask]

        node_total_load_ctx = context_cache["node_total_load"].get(tname, {})
        edge_contrib_ctx = context_cache["edge_contrib"].get(tname, {})

        sub_contrib = {int(nid): 0.0 for nid in node_ids.tolist()}
        for idx, r in sub_edges.iterrows():
            idx = int(idx)
            u, v, w = int(r["u"]), int(r["v"]), float(r["weight"])
            if w <= 0:
                continue
            if idx not in edge_contrib_ctx:
                continue
            u_load, v_load = edge_contrib_ctx[idx]
            if u in sub_contrib:
                sub_contrib[u] += float(u_load)
            if v in sub_contrib:
                sub_contrib[v] += float(v_load)

        cap_eff = {}
        for nid in node_ids.tolist():
            nid = int(nid)
            total_load = float(node_total_load_ctx.get(nid, 0.0))
            fixed_outside = total_load - float(sub_contrib.get(nid, 0.0))
            if fixed_outside < 0.0:
                fixed_outside = 0.0
            res = float(cap_full[nid]) - fixed_outside
            cap_eff[nid] = res

        prob = pulp.LpProblem(f"TDPCS_local_{tname}", pulp.LpMinimize)

        o_vars = {idx: pulp.LpVariable(f"o_{idx}", lowBound=0.0, upBound=1.0, cat="Continuous")
                  for idx in sub_edges.index}
        s_vars = {nid: pulp.LpVariable(f"s_{nid}", lowBound=0.0, cat="Continuous")
                  for nid in node_ids.tolist()}

        prob += pulp.lpSum([s_vars[nid] for nid in node_ids.tolist()])

        out_edges = {int(nid): [] for nid in node_ids.tolist()}
        in_edges = {int(nid): [] for nid in node_ids.tolist()}

        for idx, r in sub_edges.iterrows():
            u, v, w = int(r["u"]), int(r["v"]), float(r["weight"])
            if w <= 0:
                continue
            out_edges[u].append((idx, w))
            in_edges[v].append((idx, w))

        for nid in node_ids.tolist():
            nid = int(nid)
            lhs_terms = []
            for idx, w in out_edges[nid]:
                lhs_terms.append(o_vars[idx] * w)
            for idx, w in in_edges[nid]:
                lhs_terms.append((1.0 - o_vars[idx]) * w)
            prob += pulp.lpSum(lhs_terms) <= cap_eff[nid] + s_vars[nid]

        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception:
            prob.solve()

        scen_slack = 0.0
        for nid in node_ids.tolist():
            val = pulp.value(s_vars[int(nid)])
            if val is None:
                val = 1e6
            scen_slack += float(val)
        total_slack += scen_slack

    memo[key] = total_slack
    return total_slack


# =========================
# 12) Global evaluation
# =========================
def solution_cost_global(all_nodes: List[int], x_plan: Dict[int, Tuple[int, int, int]]) -> float:
    build = 0.0
    op = 0.0
    for n in all_nodes:
        sid = seg_to_id(x_plan[int(n)])
        build += build_cost[sid]
    for tname, days in SCENARIO_DAYS.items():
        frac = days / 365.0
        for n in all_nodes:
            sid = seg_to_id(x_plan[int(n)])
            op += op_cost_yr[sid] * frac
    return build + op


def evaluate_global_plan(
    all_nodes: List[int],
    edges_by_scenario: Dict[str, pd.DataFrame],
    x_plan: Dict[int, Tuple[int, int, int]],
    beta_penalty: float,
    logger: Optional[RunLogger] = None
) -> Tuple[float, float, float]:
    global_cost = solution_cost_global(all_nodes, x_plan)

    y = {int(i): (1 if seg_to_id(x_plan[int(i)]) != 0 else 0) for i in all_nodes}
    node_set = set(all_nodes)

    total_slack = 0.0
    for tname, edf in edges_by_scenario.items():
        days_t = SCENARIO_DAYS[tname]
        cap = {int(i): C_sess_day[seg_to_id(x_plan[int(i)])] * days_t for i in all_nodes}

        sub_edges = edf[edf["u"].isin(node_set) & edf["v"].isin(node_set)]
        prob = pulp.LpProblem(f"TDPCS_global_{tname}", pulp.LpMinimize)

        o_vars = {idx: pulp.LpVariable(f"o_{idx}", lowBound=0.0, upBound=1.0, cat="Continuous")
                  for idx in sub_edges.index}
        s_vars = {nid: pulp.LpVariable(f"s_{nid}", lowBound=0.0, cat="Continuous")
                  for nid in all_nodes}

        prob += pulp.lpSum([s_vars[nid] for nid in all_nodes])

        out_edges = {int(nid): [] for nid in all_nodes}
        in_edges = {int(nid): [] for nid in all_nodes}

        for idx, r in sub_edges.iterrows():
            u, v, w = int(r["u"]), int(r["v"]), float(r["weight"])
            if w <= 0:
                continue
            out_edges[u].append((idx, w))
            in_edges[v].append((idx, w))

        for nid in all_nodes:
            nid = int(nid)
            lhs_terms = []
            for idx, w in out_edges[nid]:
                lhs_terms.append(o_vars[idx] * w)
            for idx, w in in_edges[nid]:
                lhs_terms.append((1.0 - o_vars[idx]) * w)
            prob += pulp.lpSum(lhs_terms) <= cap[nid] + s_vars[nid]

        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception:
            prob.solve()

        scen_slack = 0.0
        for nid in all_nodes:
            val = pulp.value(s_vars[int(nid)])
            if val is None:
                val = 1e6
            scen_slack += float(val)
        total_slack += scen_slack

    global_obj = global_cost + beta_penalty * total_slack
    if logger is not None:
        logger.update_local_fe(obj=global_obj, cost=global_cost, viol=total_slack)
    return global_obj, global_cost, total_slack


# =========================
# 13) d_norm + roulette init
# =========================
def compute_d_norm(all_nodes_solve: List[int], edges_by_scenario: Dict[str, pd.DataFrame]) -> Dict[int, float]:
    """Build d_norm from incident edge-weight sums, then min-max normalize to [0,1]."""
    d_raw = {int(nid): 0.0 for nid in all_nodes_solve}
    solve_set = set(int(x) for x in all_nodes_solve)

    for edf in edges_by_scenario.values():
        sub = edf[edf["u"].isin(solve_set) & edf["v"].isin(solve_set)]
        for _, r in sub.iterrows():
            u, v, w = int(r["u"]), int(r["v"]), float(r["weight"])
            if w <= 0:
                continue
            d_raw[u] += w
            d_raw[v] += w

    vals = np.array(list(d_raw.values()), dtype=np.float64)
    dmin, dmax = float(vals.min()), float(vals.max())
    if dmax <= dmin + 1e-12:
        return {nid: 0.0 for nid in d_raw.keys()}
    return {nid: (float(v) - dmin) / (dmax - dmin + 1e-12) for nid, v in d_raw.items()}


def init_plan_normal_roulette_scaleid(all_nodes_solve: List[int], d_norm: Dict[int, float], rng: random.Random, sigma: float = 1.0) -> Dict[int, int]:
    """Sample one scale_id per node from a Gaussian-like roulette over [0,1,2,3]."""
    scale_candidates = [0, 1, 2, 3]
    plan: Dict[int, int] = {}
    for nid in all_nodes_solve:
        nid = int(nid)
        di = float(d_norm[int(nid)])
        if di <= 1e-12:
            plan[nid] = 0
            continue
        L_i = 3.0 * di
        weights = [math.exp(-((l - L_i) ** 2) / (2.0 * sigma * sigma)) for l in scale_candidates]
        plan[nid] = int(rng.choices(scale_candidates, weights=weights, k=1)[0])
    return plan


def init_plan_normal_roulette_segments_sa_like(all_nodes_solve: List[int], d_norm: Dict[int, float], rng: random.Random, sigma: float = 1.0) -> Dict[int, Tuple[int, int, int]]:
    """Convert SA-like roulette scale initialization to segment encoding."""
    plan_scale = init_plan_normal_roulette_scaleid(all_nodes_solve, d_norm, rng, sigma=sigma)
    return {int(nid): id_to_seg(int(sid)) for nid, sid in plan_scale.items()}


# =========================
# 14) GA operators on segments
# =========================
def compute_component_dbar(comp_nodes: List[int], d_raw: Dict[int, float]) -> Dict[int, float]:
    eps = 1e-12
    vals = [float(d_raw[int(n)]) for n in comp_nodes]
    dmin, dmax = min(vals), max(vals)
    out = {}
    for n in comp_nodes:
        n = int(n)
        if dmax <= dmin + eps:
            out[n] = 0.0
        else:
            out[n] = (float(d_raw[n]) - dmin) / (dmax - dmin + eps)
    return out


# =========================
# Mutation v2
# =========================
_MAX_BUILD = float(max(build_cost.values())) if len(build_cost) > 0 else 1.0
_MAX_BUILD = max(_MAX_BUILD, 1.0)


def _scale_logits_cost_aware(dbar: float, a: float = 2.2, b: float = 1.6) -> Tuple[List[int], List[float]]:
    cands = [0, 1, 2, 3]
    logits = [0.0] * 4
    logits[0] = 1.0 * (1.0 - float(dbar))
    for l in [1, 2, 3]:
        cost_norm = float(build_cost[l]) / _MAX_BUILD
        logits[l] = float(a) * float(dbar) * float(l) - float(b) * cost_norm
    return cands, logits


def _softmax_sample(cands: List[int], logits: List[float], rng: random.Random) -> int:
    arr = np.array(logits, dtype=np.float64)
    arr = arr - float(arr.max())
    exps = np.exp(arr)
    probs = exps / (float(exps.sum()) + 1e-12)
    return int(rng.choices(cands, weights=probs.tolist(), k=1)[0])


def mutate_segment_v2(
    current_seg: Tuple[int, int, int],
    dbar: float,
    rng: random.Random,
    pmin: float,
    pmax: float,
) -> Tuple[int, int, int]:
    p_mut = pmin + (pmax - pmin) * float(dbar)
    if rng.random() > p_mut:
        return current_seg

    sid = seg_to_id(current_seg)

    p_on = 0.15 + 0.70 * float(dbar)
    p_off = 0.18 * (1.0 - float(dbar))

    if sid == 0:
        if rng.random() > p_on:
            return current_seg
        cands, logits = _scale_logits_cost_aware(dbar)
        for _ in range(3):
            sid2 = _softmax_sample(cands, logits, rng)
            if sid2 != 0:
                return id_to_seg(sid2)
        return id_to_seg(1)

    if rng.random() < p_off:
        return (0, 0, 0)

    if rng.random() < 0.65:
        up_bias = 0.35 + 0.50 * float(dbar)
        if rng.random() < up_bias:
            sid2 = min(3, sid + 1)
        else:
            sid2 = max(1, sid - 1)
        return id_to_seg(sid2)

    cands, logits = _scale_logits_cost_aware(dbar)
    cands_nz = [1, 2, 3]
    logits_nz = [logits[1], logits[2], logits[3]]
    sid2 = _softmax_sample(cands_nz, logits_nz, rng)
    return id_to_seg(sid2)


def crossover_segments_paper(
    parent1: Dict[int, Tuple[int, int, int]],
    parent2: Dict[int, Tuple[int, int, int]],
    comp_nodes: List[int],
    rng: random.Random,
    p_c: float
) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int]]]:
    child1 = dict(parent1)
    child2 = dict(parent2)
    for nid in comp_nodes:
        nid = int(nid)
        if rng.random() < p_c:
            child1[nid], child2[nid] = child2[nid], child1[nid]
    return child1, child2


def tournament_select(scored: List[Tuple[float, Dict[int, Tuple[int, int, int]]]], rng: random.Random, k: int = 2):
    cand = rng.sample(scored, k)
    cand.sort(key=lambda x: x[0])
    return cand[0][1]


def ga_optimize_component(
    comp_nodes: List[int],
    edges_by_scenario: Dict[str, pd.DataFrame],
    shared_plan: Dict[int, Tuple[int, int, int]],
    d_raw: Dict[int, float],
    logger: Optional[RunLogger],
    seed: int,
    pop_size: int,
    generations: int,
    beta_penalty: float,
    context_cache: dict,
) -> Tuple[Dict[int, Tuple[int, int, int]], int]:
    rng = random.Random(seed)
    comp_nodes = [int(x) for x in comp_nodes]
    dbar = compute_component_dbar(comp_nodes, d_raw)

    neighborhood = build_component_neighborhood_nodes(comp_nodes, edges_by_scenario)
    memo = {}

    def local_cost(ind_plan: Dict[int, Tuple[int, int, int]]) -> float:
        build = 0.0
        op = 0.0
        for n in comp_nodes:
            sid = seg_to_id(ind_plan[n])
            build += build_cost[sid]
        for tname, days in SCENARIO_DAYS.items():
            frac = days / 365.0
            for n in comp_nodes:
                sid = seg_to_id(ind_plan[n])
                op += op_cost_yr[sid] * frac
        return build + op

    def make_individual() -> Dict[int, Tuple[int, int, int]]:
        ind = dict(shared_plan)
        for nid in comp_nodes:
            ind[nid] = mutate_segment_v2(
                current_seg=ind[nid],
                dbar=dbar[nid],
                rng=rng,
                pmin=P_MUT_MIN,
                pmax=P_MUT_MAX
            )
        return ind

    base_plan = dict(shared_plan)
    base_viol = lp_violation_local_with_memo(neighborhood, comp_nodes, edges_by_scenario, base_plan, memo, context_cache)
    base_cost = local_cost(base_plan)
    base_fit = base_cost + beta_penalty * base_viol

    if pop_size <= 1:
        population = [dict(shared_plan)]
    else:
        population = [dict(shared_plan)] + [make_individual() for _ in range(pop_size - 1)]

    best_ind = dict(shared_plan)
    best_fit = base_fit
    local_fe_count = 0

    for _gen in range(generations):
        scored: List[Tuple[float, Dict[int, Tuple[int, int, int]]]] = []
        for ind in population:
            viol = lp_violation_local_with_memo(
                neighborhood, comp_nodes, edges_by_scenario, ind, memo, context_cache
            )
            cost = local_cost(ind)
            fit = cost + beta_penalty * viol
            scored.append((fit, ind))
            local_fe_count += 1

        scored.sort(key=lambda x: x[0])
        if scored[0][0] < best_fit:
            best_fit = scored[0][0]
            best_ind = dict(scored[0][1])

        new_pop = [dict(best_ind)]
        while len(new_pop) < pop_size:
            p1 = tournament_select(scored, rng, k=min(TOURNAMENT_K, len(scored)))
            p2 = tournament_select(scored, rng, k=min(TOURNAMENT_K, len(scored)))
            c1, c2 = crossover_segments_paper(p1, p2, comp_nodes, rng, p_c=P_CROSS)

            for nid in comp_nodes:
                c1[nid] = mutate_segment_v2(c1[nid], dbar[nid], rng, P_MUT_MIN, P_MUT_MAX)
                c2[nid] = mutate_segment_v2(c2[nid], dbar[nid], rng, P_MUT_MIN, P_MUT_MAX)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    if best_fit > base_fit + 1e-12:
        return {nid: shared_plan[nid] for nid in comp_nodes}, int(local_fe_count)
    return {nid: best_ind[nid] for nid in comp_nodes}, int(local_fe_count)


# =========================
# 15) One run of GCPCC
# =========================
def plan_change_ratio(all_nodes: List[int], X_old: Dict[int, Tuple[int, int, int]], X_new: Dict[int, Tuple[int, int, int]]) -> float:
    changed = 0
    for nid in all_nodes:
        if X_old[int(nid)] != X_new[int(nid)]:
            changed += 1
    return changed / max(1, len(all_nodes))


def _sort_components(components: List[List[int]]) -> List[List[int]]:
    comps = [sorted([int(x) for x in c]) for c in components if len(c) > 0]
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def _union_connected_components(active_nodes: List[int], all_edges_union: pd.DataFrame) -> List[List[int]]:
    G = nx.Graph()
    G.add_nodes_from([int(x) for x in active_nodes])
    if len(all_edges_union) > 0:
        G.add_edges_from(
            list(
                zip(
                    all_edges_union["u"].astype(int).values.tolist(),
                    all_edges_union["v"].astype(int).values.tolist(),
                )
            )
        )
    comps = [list(comp) for comp in nx.connected_components(G)]
    return _sort_components(comps)


def _aggregate_edges_for_spatial_cut(edges_by_scenario: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for edf in edges_by_scenario.values():
        if len(edf) == 0:
            continue
        tmp = edf[["u", "v", "distance_km", "weight"]].copy()
        uv_min = np.minimum(tmp["u"].values, tmp["v"].values)
        uv_max = np.maximum(tmp["u"].values, tmp["v"].values)
        tmp["_a"], tmp["_b"] = uv_min, uv_max
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["u", "v", "distance_km", "weight"])

    all_df = pd.concat(rows, axis=0, ignore_index=True)
    agg = (
        all_df.groupby(["_a", "_b"], as_index=False)
        .agg({"distance_km": "mean", "weight": "mean"})
        .rename(columns={"_a": "u", "_b": "v"})
    )
    agg["u"] = agg["u"].astype(int)
    agg["v"] = agg["v"].astype(int)
    agg["distance_km"] = agg["distance_km"].astype(float)
    agg["weight"] = agg["weight"].astype(float)
    return agg[["u", "v", "distance_km", "weight"]]


def _divisors(n: int) -> List[int]:
    ds = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            ds.append(i)
            if i != (n // i):
                ds.append(n // i)
    return sorted(ds)


def _pick_divisor_near(n: int, target: int, min_value: int = 1) -> int:
    dlist = [d for d in _divisors(n) if d >= int(min_value)]
    if not dlist:
        return 1
    target = int(max(min_value, target))
    return int(min(dlist, key=lambda d: (abs(d - target), d)))


def make_random_groups_equal_size(
    all_nodes_solve: List[int],
    rng: random.Random,
    group_size: Optional[int] = None,
    num_groups: Optional[int] = None,
) -> Tuple[List[List[int]], int, int]:
    """
    Randomly permute all variables, then evenly split into m groups of size s (n = m*s).
    """
    n = int(len(all_nodes_solve))
    if n <= 0:
        raise ValueError("No active variables for random grouping.")

    s = int(group_size) if group_size is not None else None
    m = int(num_groups) if num_groups is not None else None

    if s is not None and s <= 0:
        raise ValueError("group_size must be positive.")
    if m is not None and m <= 0:
        raise ValueError("num_groups must be positive.")

    if s is not None and m is not None:
        if s * m != n:
            raise ValueError(f"Random grouping requires n=m*s; got n={n}, m={m}, s={s}.")
    elif s is not None:
        if n % s != 0:
            raise ValueError(f"Random grouping requires n%group_size==0; got n={n}, group_size={s}.")
        m = n // s
    elif m is not None:
        if n % m != 0:
            raise ValueError(f"Random grouping requires n%num_groups==0; got n={n}, num_groups={m}.")
        s = n // m
    else:
        # Default: choose m near sqrt(n), then enforce exact divisibility.
        m0 = max(2, int(round(math.sqrt(n))))
        m = _pick_divisor_near(n=n, target=m0, min_value=2)
        s = n // m

    if m * s != n:
        raise ValueError(f"Internal error: n=m*s violated after resolve, n={n}, m={m}, s={s}.")

    perm = [int(x) for x in all_nodes_solve]
    rng.shuffle(perm)
    groups = [sorted(perm[i * s:(i + 1) * s]) for i in range(m)]
    return groups, m, s


def make_geographical_groups_gz_fs(
    node_df_active: pd.DataFrame,
    all_nodes_solve: List[int],
) -> Tuple[List[List[int]], str]:
    """
    Build two fixed geographical components [Guangzhou, Foshan].
    Priority:
    1) city-like column parsing (if available)
    2) KMeans(2) on (longitude, latitude)
    3) deterministic longitude split fallback
    """
    active_set = set(int(x) for x in all_nodes_solve)
    df = node_df_active[node_df_active["id"].astype(int).isin(active_set)].copy()
    if len(df) == 0:
        raise RuntimeError("No active nodes for geographical grouping.")

    df["id"] = df["id"].astype(int)
    df["longitude"] = df["longitude"].astype(float)
    df["latitude"] = df["latitude"].astype(float)

    n = int(len(df))
    method = ""

    city_col_candidates = [
        "city", "city_name", "city_cn", "city_en",
        "所属城市", "城市", "地市", "行政区",
    ]
    gz_kw = ["guangzhou", "骞垮窞", "gz"]
    fs_kw = ["foshan", "浣涘北", "fs"]

    city_col = next((c for c in city_col_candidates if c in df.columns), None)
    if city_col is not None:
        gz_ids, fs_ids, unknown_ids = [], [], []
        for _, r in df.iterrows():
            nid = int(r["id"])
            tok = str(r[city_col]).strip().lower()
            if any(k in tok for k in gz_kw):
                gz_ids.append(nid)
            elif any(k in tok for k in fs_kw):
                fs_ids.append(nid)
            else:
                unknown_ids.append(nid)

        if len(gz_ids) > 0 and len(fs_ids) > 0:
            if len(unknown_ids) > 0:
                id2xy = {
                    int(r["id"]): (float(r["longitude"]), float(r["latitude"]))
                    for _, r in df.iterrows()
                }
                gz_cx = np.mean([id2xy[i][0] for i in gz_ids])
                gz_cy = np.mean([id2xy[i][1] for i in gz_ids])
                fs_cx = np.mean([id2xy[i][0] for i in fs_ids])
                fs_cy = np.mean([id2xy[i][1] for i in fs_ids])
                for nid in unknown_ids:
                    x, y = id2xy[nid]
                    dg = (x - gz_cx) ** 2 + (y - gz_cy) ** 2
                    df2 = (x - fs_cx) ** 2 + (y - fs_cy) ** 2
                    if dg <= df2:
                        gz_ids.append(nid)
                    else:
                        fs_ids.append(nid)
            method = f"city_column:{city_col}"
            gz_nodes = sorted(gz_ids)
            fs_nodes = sorted(fs_ids)
        else:
            gz_nodes, fs_nodes = [], []
    else:
        gz_nodes, fs_nodes = [], []

    if len(gz_nodes) == 0 or len(fs_nodes) == 0:
        try:
            xy = df[["longitude", "latitude"]].values
            if len(df) >= 2:
                km = KMeans(n_clusters=2, n_init=20, random_state=42)
                labels = km.fit_predict(xy)
                c0 = float(df.loc[labels == 0, "longitude"].mean())
                c1 = float(df.loc[labels == 1, "longitude"].mean())
                gz_lab = 0 if c0 >= c1 else 1
                fs_lab = 1 - gz_lab
                gz_nodes = sorted(df.loc[labels == gz_lab, "id"].astype(int).tolist())
                fs_nodes = sorted(df.loc[labels == fs_lab, "id"].astype(int).tolist())
                method = "kmeans_lonlat_2clusters"
        except Exception:
            gz_nodes, fs_nodes = [], []

    if len(gz_nodes) == 0 or len(fs_nodes) == 0:
        lon_sorted = df.sort_values(["longitude", "latitude", "id"]).reset_index(drop=True)
        ids = lon_sorted["id"].astype(int).tolist()
        if n == 1:
            gz_nodes = ids
            fs_nodes = []
        else:
            half = max(1, n // 2)
            fs_nodes = sorted(ids[:half])
            gz_nodes = sorted(ids[half:])
            if len(gz_nodes) == 0 and len(fs_nodes) > 1:
                gz_nodes = [fs_nodes.pop()]
        method = "longitude_deterministic_split"

    gz_set, fs_set = set(gz_nodes), set(fs_nodes)
    all_set = set(int(x) for x in all_nodes_solve)
    if len(gz_set & fs_set) > 0:
        raise RuntimeError("Geographical grouping overlap detected.")
    if (gz_set | fs_set) != all_set:
        missing = sorted(list(all_set - (gz_set | fs_set)))
        for nid in missing:
            fs_nodes.append(int(nid))
        fs_nodes = sorted(fs_nodes)

    components = [sorted(gz_nodes)]
    if len(fs_nodes) > 0:
        components.append(sorted(fs_nodes))
    return components, method


def make_dg_static_components(
    all_nodes_solve: List[int],
    edges_by_scenario: Dict[str, pd.DataFrame],
    base_plan: Dict[int, Tuple[int, int, int]],
    beta_penalty: float,
    eps_abs: float = DG_EPS_ABS,
    eps_rel: float = DG_EPS_REL,
) -> Tuple[List[List[int]], Dict[str, float]]:
    """
    Classical DG-style static grouping:
    1) Fix a baseline context vector.
    2) Pairwise interaction test on (xi, xj) using finite differences on objective.
    3) Build interaction graph and take connected components as fixed groups.
    """
    nodes = [int(x) for x in sorted(all_nodes_solve)]
    n = len(nodes)
    if n == 0:
        raise RuntimeError("No active variables for DG grouping.")
    if n == 1:
        return [[nodes[0]]], {
            "n_nodes": 1,
            "pairs_tested": 0,
            "interaction_edges": 0,
            "cache_evals": 0,
            "eps_abs": float(eps_abs),
            "eps_rel": float(eps_rel),
        }

    sid_base = {int(nid): int(seg_to_id(base_plan[int(nid)])) for nid in nodes}

    def alt_sid(sid: int) -> int:
        # Two-point perturbation in discrete scale space.
        return 3 if int(sid) != 3 else 0

    eval_cache: Dict[Tuple[Tuple[int, int], ...], float] = {}

    def eval_obj_with_overrides(overrides: List[Tuple[int, int]]) -> float:
        key = tuple(sorted((int(nid), int(sid)) for nid, sid in overrides))
        if key in eval_cache:
            return float(eval_cache[key])
        x = dict(base_plan)
        for nid, sid in key:
            x[int(nid)] = id_to_seg(int(sid))
        obj, _cost, _viol = evaluate_global_plan(
            all_nodes=nodes,
            edges_by_scenario=edges_by_scenario,
            x_plan=x,
            beta_penalty=beta_penalty,
            logger=None,  # DG preprocessing should not alter FE logs.
        )
        eval_cache[key] = float(obj)
        return float(obj)

    f_base = eval_obj_with_overrides([])
    single_alt_cache: Dict[int, float] = {}
    for nid in nodes:
        single_alt_cache[int(nid)] = eval_obj_with_overrides([(int(nid), alt_sid(sid_base[int(nid)]))])

    G = nx.Graph()
    G.add_nodes_from(nodes)

    n_pairs = n * (n - 1) // 2
    tested = 0
    for pi in range(n - 1):
        i = int(nodes[pi])
        si_alt = alt_sid(sid_base[i])
        fi_alt = float(single_alt_cache[i])
        for pj in range(pi + 1, n):
            j = int(nodes[pj])
            sj_alt = alt_sid(sid_base[j])
            fj_alt = float(single_alt_cache[j])
            fij_alt = eval_obj_with_overrides([(i, si_alt), (j, sj_alt)])

            # DG interaction criterion:
            # delta1 = f(xi=alt, xj=base) - f(base)
            # delta2 = f(xi=alt, xj=alt)  - f(xj=alt)
            delta1 = float(fi_alt - f_base)
            delta2 = float(fij_alt - fj_alt)
            drift = abs(delta1 - delta2)
            tol = max(float(eps_abs), float(eps_rel) * (abs(delta1) + abs(delta2) + 1.0))
            if drift > tol:
                G.add_edge(i, j)

            tested += 1

    components = _sort_components([list(comp) for comp in nx.connected_components(G)])
    stats = {
        "n_nodes": int(n),
        "pairs_tested": int(tested),
        "interaction_edges": int(G.number_of_edges()),
        "cache_evals": int(len(eval_cache)),
        "eps_abs": float(eps_abs),
        "eps_rel": float(eps_rel),
    }
    return components, stats


def build_components_for_mode(
    decomp_mode: str,
    scenarios: Dict[str, ScenarioGraph],
    edges_by_scenario: Dict[str, pd.DataFrame],
    all_edges_union: pd.DataFrame,
    all_nodes_solve: List[int],
    node_df_active: pd.DataFrame,
    kd_tree: KDTree,
    coords: np.ndarray,
    id_list: np.ndarray,
    id2posidx: Dict[int, int],
) -> List[List[int]]:
    mode = str(decomp_mode).strip().lower()
    if mode not in DECOMP_MODES:
        raise ValueError(f"Unknown decomp_mode={decomp_mode}, allowed={sorted(DECOMP_MODES)}")

    if mode == DECOMP_MODE_SINGLE_COMPONENT:
        print("[2/7] Decomposition: single component", flush=True)
        return [_sort_components([all_nodes_solve])[0]]

    if mode == DECOMP_MODE_RANDOM_GROUPING_RR:
        print("[2/7] Decomposition: random grouping (cycle-wise regrouping)", flush=True)
        # Real grouping is re-sampled at each cycle inside run_gcpcc.
        return [_sort_components([all_nodes_solve])[0]]

    if mode == DECOMP_MODE_GEO_GROUPING_RR:
        print("[2/7] Decomposition: geographical grouping (Guangzhou/Foshan)", flush=True)
        components, method = make_geographical_groups_gz_fs(
            node_df_active=node_df_active,
            all_nodes_solve=all_nodes_solve,
        )
        if len(components) == 0:
            raise RuntimeError("Geographical grouping returned empty components.")
        sizes = [len(c) for c in components]
        print(f"  geo_method={method}", flush=True)
        print(f"  geo_components={len(components)} sizes={sizes}", flush=True)
        return components

    if mode == DECOMP_MODE_DG_STATIC_RR:
        print("[2/7] Decomposition: differential grouping (static, preprocessing)", flush=True)
        # Real DG detection runs in run_gcpcc because it needs a baseline context vector.
        return [_sort_components([all_nodes_solve])[0]]

    if mode == DECOMP_MODE_TOPO_CC_RR:
        print("[2/7] Decomposition: topology connected components", flush=True)
        comps = _union_connected_components(all_nodes_solve, all_edges_union)
        if not comps:
            raise RuntimeError("Topology connected components is empty.")
        return comps

    if mode == DECOMP_MODE_SPATIAL_ONLY_RR:
        print("[2/7] Decomposition: spatial cut on aggregated 4-season graph", flush=True)
        agg_edges = _aggregate_edges_for_spatial_cut(edges_by_scenario)
        W_agg = build_symmetric_W(np.array(all_nodes_solve, dtype=int), agg_edges)
        sg_agg = ScenarioGraph(
            name="aggregate_4s",
            nodes=np.array(all_nodes_solve, dtype=int),
            id2idx={int(nid): i for i, nid in enumerate(all_nodes_solve)},
            edges=agg_edges,
            W=W_agg,
        )
        clusters = spatial_cut(sg_agg.W, sg_agg.nodes, max_k=MAX_K_SPECTRAL)
        clusters_repaired, _labels_repaired, fragment_nodes = repair_spatial_fragments(
            sg=sg_agg,
            clusters=clusters,
            kd_tree=kd_tree,
            coords=coords,
            id_list=id_list,
            id2posidx=id2posidx,
            topk=REASSIGN_TOPK,
        )
        sizes = sorted([len(c) for c in clusters_repaired], reverse=True)
        print(f"  aggregate_4s: K={len(clusters_repaired)} top_sizes={sizes[:10]} fragments={len(fragment_nodes)}", flush=True)
        comps = _sort_components(clusters_repaired)
        if not comps:
            raise RuntimeError("Spatial-only decomposition returned empty components.")
        return comps

    # mode == DECOMP_MODE_ORIG_CONSENSUS_RR
    print("[2/7] Spatial cut + fragment repair", flush=True)
    scenario_node_labels: Dict[str, Dict[int, int]] = {}

    for sname, sg in scenarios.items():
        clusters = spatial_cut(sg.W, sg.nodes, max_k=MAX_K_SPECTRAL)
        clusters_repaired, labels_repaired, fragment_nodes = repair_spatial_fragments(
            sg=sg,
            clusters=clusters,
            kd_tree=kd_tree,
            coords=coords,
            id_list=id_list,
            id2posidx=id2posidx,
            topk=REASSIGN_TOPK,
        )
        scenario_node_labels[sname] = labels_repaired
        sizes = sorted([len(c) for c in clusters_repaired], reverse=True)
        print(f"  {sname}: K={len(clusters_repaired)} top_sizes={sizes[:10]} fragments={len(fragment_nodes)}", flush=True)

    print("[3/7] Consensus cut", flush=True)
    components = consensus_cut_vA(
        scenario_node_labels=scenario_node_labels,
        union_edges=all_edges_union,
        active_nodes=all_nodes_solve,
        min_agree=CONSENSUS_MIN_AGREE,
    )

    if CONSENSUS_MERGE_SMALL:
        components = merge_small_components_to_nearest_component(
            components=components,
            node_df_active=node_df_active,
            small_abs=CONSENSUS_SMALL_ABS,
            small_frac=CONSENSUS_SMALL_FRAC,
            topk=CONSENSUS_REASSIGN_TOPK,
        )
    components = _sort_components(components)
    if not components:
        raise RuntimeError("Consensus decomposition returned empty components. Check labels/edges.")
    return components


def run_gcpcc(
    logger: Optional[RunLogger],
    seed: int,
    run_id: int = 1,
    decomp_mode: str = DEFAULT_DECOMP_MODE,
    log_dir: Optional[str] = None,
    rg_group_size: Optional[int] = None,
    rg_num_groups: Optional[int] = None,
    dg_eps_abs: float = DG_EPS_ABS,
    dg_eps_rel: float = DG_EPS_REL,
    max_outer_gen: Optional[int] = MAX_OUTER_GEN,
):
    np.random.seed(seed)
    random.seed(seed)

    decomp_mode = str(decomp_mode).strip().lower()
    if decomp_mode not in DECOMP_MODES:
        raise ValueError(f"Unknown decomp_mode={decomp_mode}, allowed={sorted(DECOMP_MODES)}")
    if log_dir is None:
        log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    node_df = read_nodes(NODE_PATH)
    all_nodes = node_df["id"].astype(int).values.tolist()
    print(f"[Data] nodes={len(all_nodes)}", flush=True)

    scenarios: Dict[str, ScenarioGraph] = {}
    edges_by_scenario_raw: Dict[str, pd.DataFrame] = {}
    union_edges = []

    print("[1/7] Load scenarios", flush=True)
    for sname, spath in SCENARIO_FILES.items():
        edf = read_edges(spath)
        edges_by_scenario_raw[sname] = edf
        print(f"  {sname}: raw_edges={len(edf)} days={SCENARIO_DAYS[sname]}", flush=True)

    edges_by_scenario, active_nodes, inactive_nodes = filter_edges_and_get_active_nodes(
        all_nodes=all_nodes,
        edges_by_scenario=edges_by_scenario_raw,
        w_min=EDGE_WEIGHT_MIN
    )
    print(f"[Filter] w_min={EDGE_WEIGHT_MIN}", flush=True)
    print(f"  active={len(active_nodes)} inactive={len(inactive_nodes)}(fixed NONE)", flush=True)
    for sname in SCENARIO_FILES.keys():
        raw_n = len(edges_by_scenario_raw[sname])
        fil_n = len(edges_by_scenario[sname])
        print(f"  {sname}: removed={raw_n - fil_n} kept={fil_n}", flush=True)

    if len(active_nodes) == 0:
        raise RuntimeError("After filtering edges, active_nodes is empty. Please lower EDGE_WEIGHT_MIN or check data.")

    active_set = set(active_nodes)

    node_df_active = node_df[node_df["id"].astype(int).isin(active_set)].copy()
    coords = np.array(list(zip(
        node_df_active["longitude"].astype(float).values,
        node_df_active["latitude"].astype(float).values
    )), dtype=float)
    id_list = node_df_active["id"].astype(int).values
    if len(coords) == 0:
        raise RuntimeError("node_df_active is empty; cannot build KDTree.")
    kd_tree = KDTree(coords)
    id2posidx = {int(nid): i for i, nid in enumerate(id_list.tolist())}

    for sname, edf in edges_by_scenario.items():
        edf = edf[edf["u"].isin(active_set) & edf["v"].isin(active_set)].copy()
        edf.reset_index(drop=True, inplace=True)
        edges_by_scenario[sname] = edf

        union_edges.append(edf[["u", "v", "distance_km"]].copy())

        W = build_symmetric_W(np.array(active_nodes, dtype=int), edf)
        scenarios[sname] = ScenarioGraph(
            name=sname,
            nodes=np.array(active_nodes, dtype=int),
            id2idx={int(nid): i for i, nid in enumerate(active_nodes)},
            edges=edf,
            W=W
        )
        print(f"  {sname}: filtered_edges={len(edf)}", flush=True)

    all_nodes_solve = active_nodes

    all_edges_union = pd.concat(union_edges, axis=0, ignore_index=True)
    if len(all_edges_union) > 0:
        uv_min = np.minimum(all_edges_union["u"].values, all_edges_union["v"].values)
        uv_max = np.maximum(all_edges_union["u"].values, all_edges_union["v"].values)
        all_edges_union["_a"], all_edges_union["_b"] = uv_min, uv_max
        all_edges_union = all_edges_union.drop_duplicates(subset=["_a", "_b"]).drop(columns=["_a", "_b"]).reset_index(drop=True)

    # Node demand proxy for GA mutation guidance.
    d_raw = {int(nid): 0.0 for nid in all_nodes_solve}
    for edf in edges_by_scenario.values():
        for _, r in edf.iterrows():
            u, v, w = int(r["u"]), int(r["v"]), float(r["weight"])
            if w <= 0:
                continue
            if u in d_raw:
                d_raw[u] += w
            if v in d_raw:
                d_raw[v] += w

    components: List[List[int]] = []
    if decomp_mode == DECOMP_MODE_DG_STATIC_RR:
        print("[2/7] Decomposition: classical differential grouping (static)", flush=True)
        rng_init = random.Random(seed)
        d_norm_init = compute_d_norm(all_nodes_solve, edges_by_scenario)
        X_init_base = init_plan_normal_roulette_segments_sa_like(all_nodes_solve, d_norm_init, rng_init, sigma=1.0)
        print("[3/7] Pairwise interaction detection", flush=True)
        components, dg_stats = make_dg_static_components(
            all_nodes_solve=all_nodes_solve,
            edges_by_scenario=edges_by_scenario,
            base_plan=X_init_base,
            beta_penalty=BETA_PENALTY,
            eps_abs=dg_eps_abs,
            eps_rel=dg_eps_rel,
        )
        print(
            f"  pairs={dg_stats['pairs_tested']} edges={dg_stats['interaction_edges']} eval_cache={dg_stats['cache_evals']}",
            flush=True,
        )
    elif decomp_mode == DECOMP_MODE_RANDOM_GROUPING_RR:
        print("[2/7] Decomposition: random grouping (fixed preprocessing)", flush=True)
        components, rg_m, rg_s = make_random_groups_equal_size(
            all_nodes_solve=all_nodes_solve,
            rng=random.Random(seed + 7919),
            group_size=rg_group_size,
            num_groups=rg_num_groups,
        )
        print(f"  random_groups={rg_m} group_size={rg_s}", flush=True)
    else:
        components = build_components_for_mode(
            decomp_mode=decomp_mode,
            scenarios=scenarios,
            edges_by_scenario=edges_by_scenario,
            all_edges_union=all_edges_union,
            all_nodes_solve=all_nodes_solve,
            node_df_active=node_df_active,
            kd_tree=kd_tree,
            coords=coords,
            id_list=id_list,
            id2posidx=id2posidx,
        )

    components = _sort_components(components)
    sizes = [len(c) for c in components]
    if sizes:
        print(f"  components={len(components)} min/med/max={min(sizes)}/{int(np.median(sizes))}/{max(sizes)}", flush=True)
        singles = sum(1 for s in sizes if s == 1)
        print(f"  singleton_components={singles}", flush=True)
    else:
        raise RuntimeError("Decomposition returned empty components.")

    print("[4/7] Dependency graph + coloring", flush=True)
    H = build_dependency_graph(components, all_edges_union)
    color_groups = greedy_coloring_groups(H)
    group_ids = sorted(color_groups.keys())
    print(f"  color_groups={len(group_ids)} sizes={[len(color_groups[g]) for g in group_ids]}", flush=True)

    # SA-like roulette initialization.
    print("[5/7] Initialize plan", flush=True)
    rng = random.Random(seed)
    d_norm = compute_d_norm(all_nodes_solve, edges_by_scenario)
    X = init_plan_normal_roulette_segments_sa_like(all_nodes_solve, d_norm, rng, sigma=1.0)

    init_obj, init_cost, init_viol = evaluate_global_plan(
        all_nodes_solve, edges_by_scenario, X, BETA_PENALTY, logger=logger
    )
    print(f"  init: cost={init_cost:.2f} viol={init_viol:.4f} obj={init_obj:.2f}", flush=True)

    best_X = dict(X)
    best_obj, best_cost, best_viol = init_obj, init_cost, init_viol

    print("[6/7] Outer iterations", flush=True)
    outer_gen = 0

    while True:
        outer_gen += 1
        if logger is not None:
            logger.set_outer_gen(outer_gen)

        group = group_ids[(outer_gen - 1) % len(group_ids)]
        comps_to_opt = color_groups[group]
        print(f"\n[Run {run_id:02d} Gen {outer_gen}] group={group} comps={len(comps_to_opt)}", flush=True)

        shared_plan = dict(X)
        context_cache = build_context_allocation_cache(all_nodes_solve, edges_by_scenario, shared_plan)

        updates: Dict[int, Tuple[int, int, int]] = {}
        fe_this_gen_single_comp = 0

        if PARALLEL_WORKERS > 1 and len(comps_to_opt) > 1:
            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
                futures = {}
                for ci in comps_to_opt:
                    comp_nodes = components[ci]
                    comp_seed = (seed * 1000003 + outer_gen * 1009 + ci * 9176) % (2**31 - 1)
                    fut = ex.submit(
                        ga_optimize_component,
                        comp_nodes,
                        edges_by_scenario,
                        shared_plan,
                        d_raw,
                        logger,
                        comp_seed,
                        GA_POP,
                        GA_GEN,
                        BETA_PENALTY,
                        context_cache,
                    )
                    futures[fut] = ci

                for fut in as_completed(futures):
                    upd, comp_fe = fut.result()
                    updates.update(upd)
                    if fe_this_gen_single_comp <= 0:
                        fe_this_gen_single_comp = int(comp_fe)
        else:
            for ci in comps_to_opt:
                comp_nodes = components[ci]
                comp_seed = (seed * 1000003 + outer_gen * 1009 + ci * 9176) % (2**31 - 1)
                upd, comp_fe = ga_optimize_component(
                    comp_nodes=comp_nodes,
                    edges_by_scenario=edges_by_scenario,
                    shared_plan=shared_plan,
                    d_raw=d_raw,
                    logger=logger,
                    seed=comp_seed,
                    pop_size=GA_POP,
                    generations=GA_GEN,
                    beta_penalty=BETA_PENALTY,
                    context_cache=context_cache,
                )
                updates.update(upd)
                if fe_this_gen_single_comp <= 0:
                    fe_this_gen_single_comp = int(comp_fe)

        for nid, seg in updates.items():
            X[int(nid)] = tuple(seg)

        if logger is not None:
            logger.add_fe(fe_this_gen_single_comp)

        global_obj, global_cost, global_viol = evaluate_global_plan(
            all_nodes_solve, edges_by_scenario, X, BETA_PENALTY, logger=logger
        )
        fe_now = (logger.fe if logger is not None else 0)
        print(
            f"[Run {run_id:02d} Gen {outer_gen}] "
            f"fe+={int(fe_this_gen_single_comp)} fe={int(fe_now)} "
            f"cost={global_cost:.2f} viol={global_viol:.4f} obj={global_obj:.2f}",
            flush=True
        )

        if logger is not None:
            logger.record_global_gen_point(global_obj, global_cost, global_viol)

        if global_obj < best_obj - 1e-9:
            best_obj, best_cost, best_viol = global_obj, global_cost, global_viol
            best_X = dict(X)

        if max_outer_gen is not None and outer_gen >= int(max_outer_gen):
            if logger is not None:
                logger.set_stop(
                    reason="max_outer_gen_reached",
                    outer_gen=outer_gen,
                )
            print(f"[Stop] reached max_outer_gen={int(max_outer_gen)}", flush=True)
            break
    if logger is not None and logger.stop_reason == "not_set":
        logger.set_stop(
            reason="finished_without_explicit_stop",
            outer_gen=outer_gen,
        )
    print("[7/7] Write outputs", flush=True)

    out = node_df.copy()

    def get_seg_for_output(nid: int) -> Tuple[int, int, int]:
        nid = int(nid)
        if nid in best_X:
            return best_X[nid]
        return (0, 0, 0)

    out["seg_bits"] = out["id"].astype(int).map(lambda nid: seg_to_str(get_seg_for_output(nid)))
    out["scale_id"] = out["id"].astype(int).map(lambda nid: seg_to_id(get_seg_for_output(nid)))
    out["scale"] = out["scale_id"].map(lambda sid: SCALE_LABEL[int(sid)])

    out["cap_spring"] = out["scale_id"].map(lambda sid: C_sess_day[int(sid)] * SCENARIO_DAYS["spring"])
    out["cap_summer"] = out["scale_id"].map(lambda sid: C_sess_day[int(sid)] * SCENARIO_DAYS["summer"])
    out["cap_autumn"] = out["scale_id"].map(lambda sid: C_sess_day[int(sid)] * SCENARIO_DAYS["autumn"])
    out["cap_winter"] = out["scale_id"].map(lambda sid: C_sess_day[int(sid)] * SCENARIO_DAYS["winter"])

    out["build_cost_usd"] = out["scale_id"].map(lambda sid: build_cost[int(sid)])
    total_days = sum(SCENARIO_DAYS.values())
    out["op_cost_usd_for_4seasons"] = out["scale_id"].map(lambda sid: op_cost_yr[int(sid)] * (total_days / 365.0))

    active_set_out = set(all_nodes_solve)
    out["is_active_solve"] = out["id"].astype(int).map(lambda nid: int(int(nid) in active_set_out))

    run_plan_path = os.path.join(log_dir, f"run_{run_id:02d}_station_plan.csv")
    out.to_csv(run_plan_path, index=False, encoding="utf-8-sig")

    return best_X, best_obj, best_cost, best_viol, run_plan_path


# =========================
# 16) Single-run entry (external scheduling friendly)
# =========================
def run_one(
    run_id: int = 1,
    seed: Optional[int] = None,
    output_root: str = os.path.join(SCRIPT_DIR, "gcpcc_logs_GF_GG"),
    max_outer_gen: Optional[int] = MAX_OUTER_GEN,
):
    run_id = int(run_id)
    seed = _resolve_seed(run_id, seed)
    output_root = output_root or os.path.join(SCRIPT_DIR, "gcpcc_logs_GF_GG")
    os.makedirs(output_root, exist_ok=True)
    run_output_dir = _build_run_output_dir(output_root, run_id, seed)

    print(f"[RunConfig] run_id={run_id} seed={seed}", flush=True)
    print(f"[OutputDir] {run_output_dir}", flush=True)

    logger = RunLogger(run_id=run_id, save_every_fe=SAVE_EVERY_FE, save_every_gen=SAVE_EVERY_OUTER_GEN)
    logger.start()

    best_X, best_obj, best_cost, best_viol, plan_path = run_gcpcc(
        logger=logger,
        seed=seed,
        run_id=run_id,
        decomp_mode=DECOMP_MODE_GEO_GROUPING_RR,
        log_dir=run_output_dir,
        rg_group_size=None,
        rg_num_groups=None,
        dg_eps_abs=DG_EPS_ABS,
        dg_eps_rel=DG_EPS_REL,
        max_outer_gen=max_outer_gen,
    )

    logger.end()
    gen_path = logger.dump_run_logs(run_output_dir)

    summary_rows = [{
        "run": run_id,
        "seed": seed,
        "best_global_obj": best_obj,
        "best_global_cost": best_cost,
        "best_global_viol": best_viol,
        "final_best_fitness": best_obj,
        "fe_count": logger.fe,
        "best_local_obj": logger.best_local_obj,
        "best_local_cost": logger.best_local_cost,
        "best_local_viol": logger.best_local_viol,
        "stop_reason": logger.stop_reason,
        "stop_outer_gen": logger.stop_outer_gen,
        "cpu_time_sec": logger.cpu_time_sec,
        "wall_time_sec": logger.wall_time_sec,
        "fe_curve_path": "",
        "gen_curve_path": gen_path,
        "global_best_curve_fe_path": "",
        "global_best_curve_gen_path": gen_path,
        "plan_path": plan_path,
        "feasible_flag": int(best_viol <= 1e-9),
    }]

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(run_output_dir, "runs_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[Done] summary={summary_path}", flush=True)
    return summary_path


def run_many(
    n_runs: int = 1,
    base_seed: Optional[int] = None,
    output_root: str = os.path.join(SCRIPT_DIR, "gcpcc_logs_GF_GG"),
    max_outer_gen: Optional[int] = MAX_OUTER_GEN,
):
    last_summary_path = ""
    for r in range(1, int(n_runs) + 1):
        run_seed = (int(base_seed) + r) if base_seed is not None else (1000 + int(r))
        last_summary_path = run_one(
            run_id=r,
            seed=run_seed,
            output_root=output_root,
            max_outer_gen=max_outer_gen,
        )
    return last_summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GF_GG ablation: fixed geographical grouping (Guangzhou/Foshan) + GCPCC-aligned downstream optimization."
    )
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=os.path.join(SCRIPT_DIR, "gcpcc_logs_GF_GG"))
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--max_outer_gen", type=int, default=MAX_OUTER_GEN)
    args = parser.parse_args()

    if int(args.n_runs) > 1:
        run_many(
            n_runs=args.n_runs,
            base_seed=args.base_seed,
            output_root=args.output_root,
            max_outer_gen=args.max_outer_gen,
        )
    else:
        run_one(
            run_id=args.run_id,
            seed=args.seed,
            output_root=args.output_root,
            max_outer_gen=args.max_outer_gen,
        )




