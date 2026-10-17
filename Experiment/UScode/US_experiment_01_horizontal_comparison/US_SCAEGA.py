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
import networkx as nx  # (not strictly needed, kept for compatibility)
from tqdm import tqdm

import pulp


# =========================
# 0) Paths (KEEP SAME STYLE)
# =========================
EDGE_DIR = r"/work/home/acn1eb8nfq/yuezhang46/USdata/Edge"
NODE_PATH = r"/work/home/acn1eb8nfq/yuezhang46/USdata/Node/stationCoordinates.csv"

SCENARIO_FILES = {
    "2020": os.path.join(EDGE_DIR, "edges_2020.csv"),
    "2025": os.path.join(EDGE_DIR, "edges_2025.csv"),
    "2030": os.path.join(EDGE_DIR, "edges_2030.csv"),
    "2035": os.path.join(EDGE_DIR, "edges_2035.csv"),
    "2040": os.path.join(EDGE_DIR, "edges_2040.csv"),
}
SCENARIO_DAYS = {"2020": 366, "2025": 365, "2030": 365, "2035": 365, "2040": 366}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "scaega_global_logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _resolve_seed(run_id: int, seed: Optional[int]) -> int:
    return int(seed) if seed is not None else 1000 + int(run_id)


def _build_run_output_dir(output_root: str, run_id: int, seed: int) -> str:
    run_output_dir = os.path.join(output_root, f"run_{run_id:02d}_seed_{seed}")
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir


# =========================
# 1) Scale / Cost / Capacity (MUST MATCH YOUR CONTRACT)
# =========================
SCALE_LABEL = {0: "NONE", 1: "S", 2: "M", 3: "L"}

# For output compatibility: 3-bit one-hot segments
SCALE_ID_TO_SEG: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),  # NONE
    1: (1, 0, 0),  # S
    2: (0, 1, 0),  # M
    3: (0, 0, 1),  # L
}
SEG_TO_SCALE_ID: Dict[Tuple[int, int, int], int] = {v: k for k, v in SCALE_ID_TO_SEG.items()}

def id_to_seg(scale_id: int) -> Tuple[int, int, int]:
    return SCALE_ID_TO_SEG[int(scale_id)]

def seg_to_id(seg: Tuple[int, int, int]) -> int:
    return SEG_TO_SCALE_ID.get(tuple(seg), 0)

def seg_to_str(seg: Tuple[int, int, int]) -> str:
    return "".join(str(int(b)) for b in seg)

L_MAX = 3  # S/M/L

# sessions/day (capacity)
C_sess_day = {0: 0, 1: 359, 2: 945, 3: 1904}
# build cost (USD)
build_cost = {0: 0, 1: 850000, 2: 1750000, 3: 3200000}
# op cost (USD/yr)
op_cost_yr = {0: 0, 1: 25000, 2: 50000, 3: 90000}


# =========================
# 2) Knobs (SCAEGA baseline)
# =========================
EDGE_WEIGHT_MIN = 1.0     # MUST MATCH your preprocessing rule
BETA_PENALTY = 1e3        # MUST MATCH your objective

GA_POP = 256               # you can tune
TOURNAMENT_K = 2
P_CROSS = 0.7

P_MUT_MIN = 0.05
P_MUT_MAX = 0.25

ELITE_K = 2               # elitism

SCAEGA_POP = GA_POP
SCAEGA_INIT_POOL_FACTOR = 2
SCAEGA_INIT_POOL = SCAEGA_POP * SCAEGA_INIT_POOL_FACTOR
SCAEGA_T0 = 1.0
SCAEGA_Q = 0.95
SCAEGA_T_MIN = 0.05

# Unified FE budget for fair comparison across methods.
# 50 * 256 = 12800 FE.
MAX_TOTAL_FE = 50 * 256

SAVE_EVERY_FE = 256
SAVE_EVERY_GEN = 1

# ---- NEW: terminal progress knobs ----
PRINT_EVERY_GEN = 1          # 1=print every generation (suggest 1/2/5/10)
PRINT_EVERY_FE_HINT = SAVE_EVERY_FE   # 0=disable; otherwise print every N FE (local best hint)


# =========================
# 3) RunLogger (KEEP SAME FIELDS/FILES STYLE)
# =========================
class RunLogger:
    def __init__(self, run_id: int, save_every_fe: int = 256, save_every_gen: int = 1):
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
        self.curve_fe = []
        self.curve_gen = []
        self._last_saved_fe = 0

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
        # One candidate evaluation (one evaluate_global_plan call) == 1 FE.
        with self._lock:
            self.fe += 1
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

            if (self.fe - self._last_saved_fe) >= self.save_every_fe:
                self.curve_fe.append({
                    "run": self.run_id,
                    "fe": self.fe,
                    "outer_gen": self.cur_outer_gen,
                    "eval_obj": self.latest_eval_obj,
                    "eval_cost": self.latest_eval_cost,
                    "eval_viol": self.latest_eval_viol,
                    "best_local_obj": self.best_local_obj,
                    "best_local_cost": self.best_local_cost,
                    "best_local_viol": self.best_local_viol,
                    "best_global_obj": self.best_global_obj,
                    "best_global_cost": self.best_global_cost,
                    "best_global_viol": self.best_global_viol,
                })
                self._last_saved_fe = self.fe

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
# 4) IO helpers (KEEP SAME BEHAVIOR)
# =========================
def read_nodes(node_path: str) -> pd.DataFrame:
    df = pd.read_csv(node_path)
    required = {"id", "longitude", "latitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stationCoordinates.csv missing columns: {missing}")
    return df.sort_values("id").reset_index(drop=True)

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

    # undirected dedup by (min, max)
    uv_min = np.minimum(df["u"].values, df["v"].values)
    uv_max = np.maximum(df["u"].values, df["v"].values)
    df["_a"], df["_b"] = uv_min, uv_max
    df = df.drop_duplicates(subset=["_a", "_b"]).drop(columns=["_a", "_b"]).reset_index(drop=True)
    return df

def filter_edges_and_get_active_nodes(
    all_nodes: List[int],
    edges_by_scenario: Dict[str, pd.DataFrame],
    w_min: float = 1.0
) -> Tuple[Dict[str, pd.DataFrame], List[int], List[int]]:
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
# 5) Global evaluation (MUST MATCH YOUR MODEL)
# =========================
def solution_cost_global(all_nodes_solve: List[int], scale_plan: Dict[int, int]) -> float:
    build = 0.0
    op = 0.0
    for n in all_nodes_solve:
        sid = int(scale_plan[int(n)])
        build += build_cost[sid]
    for tname, days in SCENARIO_DAYS.items():
        frac = days / 365.0
        for n in all_nodes_solve:
            sid = int(scale_plan[int(n)])
            op += op_cost_yr[sid] * frac
    return build + op

def evaluate_global_plan(
    all_nodes_solve: List[int],
    edges_by_scenario: Dict[str, pd.DataFrame],
    scale_plan: Dict[int, int],
    beta_penalty: float,
    logger: Optional[RunLogger] = None
) -> Tuple[float, float, float]:
    """
    Returns: (global_obj, global_cost, total_slack)
    global_obj = global_cost + beta * total_slack
    """
    global_cost = solution_cost_global(all_nodes_solve, scale_plan)

    y = {int(i): (1 if int(scale_plan[int(i)]) != 0 else 0) for i in all_nodes_solve}
    node_set = set(all_nodes_solve)

    total_slack = 0.0
    for tname, edf in edges_by_scenario.items():
        days_t = SCENARIO_DAYS[tname]
        cap = {int(i): C_sess_day[int(scale_plan[int(i)])] * days_t for i in all_nodes_solve}

        sub_edges = edf[edf["u"].isin(node_set) & edf["v"].isin(node_set)]

        prob = pulp.LpProblem(f"TDPCS_global_{tname}", pulp.LpMinimize)

        o_vars = {idx: pulp.LpVariable(f"o_{idx}", lowBound=0.0, upBound=1.0, cat="Continuous")
                  for idx in sub_edges.index}
        s_vars = {nid: pulp.LpVariable(f"s_{nid}", lowBound=0.0, cat="Continuous")
                  for nid in all_nodes_solve}

        prob += pulp.lpSum([s_vars[nid] for nid in all_nodes_solve])

        out_edges = {int(nid): [] for nid in all_nodes_solve}
        in_edges = {int(nid): [] for nid in all_nodes_solve}

        for idx, r in sub_edges.iterrows():
            u, v, w = int(r["u"]), int(r["v"]), float(r["weight"])
            if w <= 0:
                continue
            out_edges[u].append((idx, w))
            in_edges[v].append((idx, w))

        for nid in all_nodes_solve:
            nid = int(nid)
            lhs_terms = []
            for idx, w in out_edges[nid]:
                lhs_terms.append(o_vars[idx] * w)
            for idx, w in in_edges[nid]:
                lhs_terms.append((1.0 - o_vars[idx]) * w)
            prob += pulp.lpSum(lhs_terms) <= cap[nid] + s_vars[nid]

        for idx, r in sub_edges.iterrows():
            u, v = int(r["u"]), int(r["v"])
            prob += o_vars[idx] <= y[u]
            prob += 1.0 - o_vars[idx] <= y[v]

        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception:
            prob.solve()

        scen_slack = 0.0
        for nid in all_nodes_solve:
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
# 6) Precompute d_norm (for smarter init/mutation)
# =========================
def compute_d_norm(all_nodes_solve: List[int], edges_by_scenario: Dict[str, pd.DataFrame]) -> Dict[int, float]:
    d_raw = {int(nid): 0.0 for nid in all_nodes_solve}
    solve_set = set(all_nodes_solve)
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


# =========================
# 7) SCAEGA operators (global)
# =========================
def init_individual(all_nodes_solve: List[int], d_norm: Dict[int, float], rng: random.Random, sigma: float = 1.0) -> Dict[int, int]:
    """
    Initialization consistent with your "normal+roulette" spirit:
    L_i = L_MAX * d_norm[i], then sample scale in {0,1,2,3} by Gaussian weights.
    """
    scale_candidates = [0, 1, 2, 3]
    ind: Dict[int, int] = {}
    for nid in all_nodes_solve:
        di = float(d_norm[int(nid)])
        if di <= 1e-12:
            ind[int(nid)] = 0
            continue
        L_i = L_MAX * di
        weights = [math.exp(-((l - L_i) ** 2) / (2.0 * sigma * sigma)) for l in scale_candidates]
        ind[int(nid)] = int(rng.choices(scale_candidates, weights=weights, k=1)[0])
    return ind

def sample_scale_by_dnorm(nid, d_norm, rng):
    di = float(d_norm[int(nid)])
    if di <= 1e-12:
        return 0
    logits = np.array([di * l for l in [1, 2, 3]], dtype=np.float64)
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()
    return int(rng.choices([1, 2, 3], weights=probs.tolist(), k=1)[0])

def individual_signature(ind, all_nodes_solve):
    return tuple(int(ind[int(nid)]) for nid in all_nodes_solve)

def simulated_annealing_select(
    scored,
    rng: random.Random,
    target_size: int,
    temperature: float,
    minimize: bool = True
) -> List[Dict[int, int]]:
    if len(scored) == 0 or target_size <= 0:
        return []
    if len(scored) == 1:
        return [dict(scored[0][1]) for _ in range(target_size)]

    eff_objs = [float(x[0]) if minimize else -float(x[0]) for x in scored]
    f_min = min(eff_objs)
    f_max = max(eff_objs)
    temperature = max(0.0, min(1.0, float(temperature)))

    selected: List[Dict[int, int]] = []
    max_attempts = 100 * target_size
    attempts = 0
    while len(selected) < target_size and attempts < max_attempts:
        attempts += 1
        A1, A2 = rng.sample(scored, 2)
        f1 = float(A1[0]) if minimize else -float(A1[0])
        f2 = float(A2[0]) if minimize else -float(A2[0])

        if f1 <= f2:
            selected.append(dict(A1[1]))
            continue

        Pk = (1.0 - (f1 - f_min) / (f_max - f_min + 1e-12)) * temperature
        Pk = max(0.0, min(1.0, Pk))
        if rng.random() <= Pk:
            selected.append(dict(A1[1]))

    if len(selected) < target_size:
        best = min(scored, key=lambda x: x[0] if minimize else -x[0])
        while len(selected) < target_size:
            selected.append(dict(best[1]))

    return selected

def _light_mutate_one_gene_scaega(
    ind: Dict[int, int],
    all_nodes_solve: List[int],
    d_norm: Dict[int, float],
    rng: random.Random
):
    if len(all_nodes_solve) == 0:
        return
    nid = int(rng.choice(all_nodes_solve))
    cur = int(ind[nid])
    if cur == 0:
        new_sid = sample_scale_by_dnorm(nid, d_norm, rng)
        if new_sid == 0:
            new_sid = int(rng.choice([1, 2, 3]))
        ind[nid] = int(new_sid)
        return

    if rng.random() < (1.0 - float(d_norm[nid])):
        ind[nid] = 0
    else:
        choices = [sid for sid in [1, 2, 3] if sid != cur]
        ind[nid] = int(rng.choice(choices))

def crossover_intersection_scaega(
    p1: Dict[int, int],
    p2: Dict[int, int],
    p1_obj: float,
    p2_obj: float,
    all_nodes_solve: List[int],
    d_norm: Dict[int, float],
    rng: random.Random,
    p_cross: float
) -> Tuple[Dict[int, int], Dict[int, int]]:
    if rng.random() > p_cross:
        return dict(p1), dict(p2)

    items1 = {int(nid): int(sid) for nid, sid in p1.items() if int(sid) != 0}
    items2 = {int(nid): int(sid) for nid, sid in p2.items() if int(sid) != 0}
    exact = {nid: sid for nid, sid in items1.items() if items2.get(nid, 0) == sid}
    same_node_diff = [nid for nid in items1.keys() & items2.keys() if items1[nid] != items2[nid]]

    obj_scale = max(1.0, abs(float(p1_obj)), abs(float(p2_obj)))
    objs_close = abs(float(p1_obj) - float(p2_obj)) <= 1e-9 * obj_scale

    def choose_shared_scale(nid: int) -> int:
        if objs_close:
            return int(rng.choice([items1[nid], items2[nid]]))
        return int(items1[nid] if p1_obj <= p2_obj else items2[nid])

    shared_diff = {nid: choose_shared_scale(nid) for nid in same_node_diff}

    def build_child() -> Dict[int, int]:
        target_count = int(rng.choice([len(items1), len(items2)]))
        selected_items: Dict[int, int] = dict(exact)
        selected_items.update(shared_diff)
        target_count = max(target_count, len(selected_items))

        remaining_candidates: List[Tuple[int, int]] = []
        used_nodes = set(selected_items.keys())
        for nid, sid in items1.items():
            if nid not in used_nodes:
                remaining_candidates.append((nid, sid))
        for nid, sid in items2.items():
            if nid not in used_nodes and nid not in items1:
                remaining_candidates.append((nid, sid))

        rng.shuffle(remaining_candidates)
        for nid, sid in remaining_candidates:
            if len(selected_items) >= target_count:
                break
            selected_items[int(nid)] = int(sid)

        unused_nodes = [int(nid) for nid in all_nodes_solve if int(nid) not in selected_items]
        rng.shuffle(unused_nodes)
        for nid in unused_nodes:
            if len(selected_items) >= target_count:
                break
            sid = sample_scale_by_dnorm(nid, d_norm, rng)
            if sid != 0:
                selected_items[int(nid)] = int(sid)

        child = {int(nid): 0 for nid in all_nodes_solve}
        for nid, sid in selected_items.items():
            child[int(nid)] = int(sid)
        return child

    c1 = build_child()
    c2 = build_child()

    p1_sig = individual_signature(p1, all_nodes_solve)
    p2_sig = individual_signature(p2, all_nodes_solve)
    if individual_signature(c1, all_nodes_solve) in (p1_sig, p2_sig):
        _light_mutate_one_gene_scaega(c1, all_nodes_solve, d_norm, rng)
    if individual_signature(c2, all_nodes_solve) in (p1_sig, p2_sig):
        _light_mutate_one_gene_scaega(c2, all_nodes_solve, d_norm, rng)

    return c1, c2

def get_adaptive_mutation_bounds(scored):
    if len(scored) == 0:
        return P_MUT_MIN, P_MUT_MAX
    objs = np.array([x[0] for x in scored], dtype=np.float64)
    mean_obj = np.mean(objs)
    std_obj = np.std(objs)
    cv = std_obj / (abs(mean_obj) + 1e-12)

    if cv < 0.05:
        pmin = P_MUT_MIN * 1.5
        pmax = P_MUT_MAX * 1.5
    elif cv > 0.20:
        pmin = P_MUT_MIN * 0.7
        pmax = P_MUT_MAX * 0.7
    else:
        pmin = P_MUT_MIN
        pmax = P_MUT_MAX

    pmin = max(0.01, min(0.30, pmin))
    pmax = max(pmin, min(0.50, pmax))
    return pmin, pmax

def mutate_individual(ind: Dict[int, int], all_nodes_solve: List[int], d_norm: Dict[int, float], rng: random.Random,
                      pmin: float, pmax: float):
    """
    Adaptive mutation: p_mut = pmin + (pmax-pmin)*d_norm[i]
    Mutation move: with prob, either flip to NONE or sample among {1,2,3} biased by d_norm.
    """
    for nid in all_nodes_solve:
        nid = int(nid)
        di = float(d_norm[nid])
        p_mut = pmin + (pmax - pmin) * di
        if rng.random() > p_mut:
            continue

        cur = int(ind[nid])
        if cur == 0:
            logits = np.array([di * l for l in [1, 2, 3]], dtype=np.float64)
            exps = np.exp(logits - logits.max())
            probs = exps / exps.sum()
            new_sid = rng.choices([1, 2, 3], weights=probs.tolist(), k=1)[0]
            ind[nid] = int(new_sid)
        else:
            if rng.random() < (1.0 - di):
                ind[nid] = 0
            else:
                logits = np.array([di * l for l in [1, 2, 3]], dtype=np.float64)
                exps = np.exp(logits - logits.max())
                probs = exps / exps.sum()
                new_sid = rng.choices([1, 2, 3], weights=probs.tolist(), k=1)[0]
                ind[nid] = int(new_sid)

def plan_change_ratio(all_nodes_solve: List[int], a: Dict[int, int], b: Dict[int, int]) -> float:
    changed = 0
    for nid in all_nodes_solve:
        if int(a[int(nid)]) != int(b[int(nid)]):
            changed += 1
    return changed / max(1, len(all_nodes_solve))


# =========================
# 8) One run of Global SCAEGA
# =========================
def run_scaega_global(logger: Optional[RunLogger], seed: int, run_id: int = 1):
    if logger is None:
        raise ValueError("logger is required to enforce FE-budget stopping.")

    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)

    node_df = read_nodes(NODE_PATH)
    all_nodes = node_df["id"].astype(int).values.tolist()
    print(f"[Info] #nodes = {len(all_nodes)}", flush=True)

    edges_by_scenario_raw: Dict[str, pd.DataFrame] = {}
    print("[Load] Reading scenarios ...", flush=True)
    for sname, spath in SCENARIO_FILES.items():
        edf = read_edges(spath)
        edges_by_scenario_raw[sname] = edf
        print(f"  - {sname}: raw edges={len(edf)}, days={SCENARIO_DAYS[sname]}", flush=True)

    # Filter edges + active nodes (MUST MATCH YOUR RULE)
    edges_by_scenario, active_nodes, inactive_nodes = filter_edges_and_get_active_nodes(
        all_nodes=all_nodes,
        edges_by_scenario=edges_by_scenario_raw,
        w_min=EDGE_WEIGHT_MIN
    )
    print(f"[Filter] EDGE_WEIGHT_MIN={EDGE_WEIGHT_MIN}", flush=True)
    print(f"  - active_nodes={len(active_nodes)}  inactive_nodes(fixed NONE)={len(inactive_nodes)}", flush=True)
    for sname in SCENARIO_FILES.keys():
        raw_n = len(edges_by_scenario_raw[sname])
        fil_n = len(edges_by_scenario[sname])
        print(f"  - {sname}: removed={raw_n - fil_n}, kept={fil_n}", flush=True)

    if len(active_nodes) == 0:
        raise RuntimeError("After filtering edges, active_nodes is empty. Please lower EDGE_WEIGHT_MIN or check data.")

    # Only solve active subgraph edges
    active_set = set(active_nodes)
    for sname in list(edges_by_scenario.keys()):
        edf = edges_by_scenario[sname]
        edf = edf[edf["u"].isin(active_set) & edf["v"].isin(active_set)].copy()
        edf.reset_index(drop=True, inplace=True)
        edges_by_scenario[sname] = edf
        print(f"  - {sname}: filtered edges (active subgraph)={len(edf)}", flush=True)

    all_nodes_solve = active_nodes
    d_norm = compute_d_norm(all_nodes_solve, edges_by_scenario)

    print("[SCAEGA] Initialize population pool ...", flush=True)
    if logger is not None:
        logger.set_outer_gen(0)

    init_pool = [init_individual(all_nodes_solve, d_norm, rng=rng, sigma=1.0) for _ in range(SCAEGA_INIT_POOL)]
    init_scored: List[Tuple[float, Dict[int, int], float, float]] = []
    for ind in init_pool:
        if (logger is not None) and (logger.fe >= MAX_TOTAL_FE):
            break
        obj, cost, viol = evaluate_global_plan(all_nodes_solve, edges_by_scenario, ind, BETA_PENALTY, logger=logger)
        init_scored.append((obj, ind, cost, viol))

        if PRINT_EVERY_FE_HINT and (logger is not None) and (logger.fe % PRINT_EVERY_FE_HINT == 0):
            print(
                f"[Run {run_id:02d} | FE {logger.fe}] "
                f"best_local_obj={logger.best_local_obj:.2f}  "
                f"best_local_cost={logger.best_local_cost:.2f}  "
                f"best_local_viol={logger.best_local_viol:.4f}",
                flush=True
            )

    if len(init_scored) == 0:
        raise RuntimeError("SCAEGA initialization could not evaluate any candidate under the FE budget.")

    init_scored.sort(key=lambda x: x[0])
    best_obj, best_ind, best_cost, best_viol = (
        init_scored[0][0],
        dict(init_scored[0][1]),
        init_scored[0][2],
        init_scored[0][3],
    )

    init_temperature = max(SCAEGA_T_MIN, min(1.0, SCAEGA_T0))
    population = simulated_annealing_select(
        scored=init_scored,
        rng=rng,
        target_size=SCAEGA_POP,
        temperature=init_temperature,
        minimize=True
    )

    best_sig = individual_signature(best_ind, all_nodes_solve)
    if not any(individual_signature(ind, all_nodes_solve) == best_sig for ind in population):
        if len(population) == 0:
            population.append(dict(best_ind))
        else:
            population[-1] = dict(best_ind)
    while len(population) < SCAEGA_POP:
        population.append(dict(best_ind))
    population = population[:SCAEGA_POP]

    initial_eval_cache: Optional[Dict[Tuple[int, ...], Tuple[float, float, float]]] = {}
    for obj, ind, cost, viol in init_scored:
        sig = individual_signature(ind, all_nodes_solve)
        if sig not in initial_eval_cache or obj < initial_eval_cache[sig][0]:
            initial_eval_cache[sig] = (obj, cost, viol)

    print("[SCAEGA] Init population selected.", flush=True)

    print("[SCAEGA] Evolve under FE budget ...", flush=True)
    pbar = tqdm(desc=f"SCAEGA Generations (run {run_id:02d})", dynamic_ncols=True)

    gen = 0
    while True:
        if (logger is not None) and (logger.fe >= MAX_TOTAL_FE):
            logger.set_stop(
                reason="max_total_fe_reached",
                outer_gen=gen,
            )
            print(f"\n[Stop] reached max_total_fe={MAX_TOTAL_FE}.", flush=True)
            break

        gen += 1
        pbar.update(1)
        if logger is not None:
            logger.set_outer_gen(gen)

        scored: List[Tuple[float, Dict[int, int], float, float]] = []
        for ind in population:
            sig = individual_signature(ind, all_nodes_solve)
            if initial_eval_cache is not None and sig in initial_eval_cache:
                obj, cost, viol = initial_eval_cache[sig]
            else:
                if (logger is not None) and (logger.fe >= MAX_TOTAL_FE):
                    break
                obj, cost, viol = evaluate_global_plan(all_nodes_solve, edges_by_scenario, ind, BETA_PENALTY, logger=logger)
            scored.append((obj, ind, cost, viol))

            # ---- NEW: FE hint print (optional) ----
            if PRINT_EVERY_FE_HINT and (logger is not None) and (logger.fe % PRINT_EVERY_FE_HINT == 0):
                print(
                    f"[Run {run_id:02d} | FE {logger.fe}] "
                    f"best_local_obj={logger.best_local_obj:.2f}  "
                    f"best_local_cost={logger.best_local_cost:.2f}  "
                    f"best_local_viol={logger.best_local_viol:.4f}",
                    flush=True
                )
        initial_eval_cache = None

        if len(scored) == 0:
            if logger is not None:
                logger.set_stop(
                    reason="max_total_fe_reached",
                    outer_gen=gen,
                )
            print(f"\n[Stop] reached max_total_fe={MAX_TOTAL_FE}.", flush=True)
            break

        scored.sort(key=lambda x: x[0])

        cur_best_obj, cur_best_ind, cur_best_cost, cur_best_viol = scored[0]
        cur_avg_obj = float(np.mean([x[0] for x in scored])) if len(scored) > 0 else float("nan")

        if cur_best_obj < best_obj - 1e-9:
            best_obj, best_ind, best_cost, best_viol = cur_best_obj, dict(cur_best_ind), cur_best_cost, cur_best_viol

        if logger is not None:
            logger.record_global_gen_point(cur_best_obj, cur_best_cost, cur_best_viol)

        # ---- NEW: tqdm postfix (real-time) ----
        pbar.set_postfix({
            "cur_best_obj": f"{cur_best_obj:.2e}",
            "cur_cost": f"{cur_best_cost:.2e}",
            "cur_viol": f"{cur_best_viol:.2e}",
            "best_obj": f"{best_obj:.2e}",
            "avg_obj": f"{cur_avg_obj:.2e}",
            "fe": (logger.fe if logger is not None else 0),
        })

        # ---- NEW: periodic explicit print (like your GCPCC logs) ----
        if (PRINT_EVERY_GEN is not None) and (PRINT_EVERY_GEN > 0) and (gen % PRINT_EVERY_GEN == 0):
            print(
                f"[Run {run_id:02d} | Gen {gen}] "
                f"cur_best: cost={cur_best_cost:.2f}  viol={cur_best_viol:.4f}  obj={cur_best_obj:.2f} | "
                f"best_so_far: cost={best_cost:.2f}  viol={best_viol:.4f}  obj={best_obj:.2f}",
                flush=True
            )

        fe_now = (logger.fe if logger is not None else 0)
        if (logger is not None) and (fe_now >= MAX_TOTAL_FE):
            logger.set_stop(
                reason="max_total_fe_reached",
                outer_gen=gen,
            )
            print(f"\n[Stop] reached max_total_fe={MAX_TOTAL_FE}.", flush=True)
            break

        temperature = SCAEGA_T0 * (SCAEGA_Q ** gen)
        temperature = max(SCAEGA_T_MIN, min(1.0, temperature))

        mating_pool = simulated_annealing_select(
            scored=scored,
            rng=rng,
            target_size=GA_POP,
            temperature=temperature,
            minimize=True
        )

        cur_p_mut_min, cur_p_mut_max = get_adaptive_mutation_bounds(scored)
        obj_cache = {
            individual_signature(ind, all_nodes_solve): float(obj)
            for obj, ind, _, _ in scored
        }

        # Next generation with elite retention
        new_pop: List[Dict[int, int]] = []

        for k in range(min(ELITE_K, len(scored))):
            new_pop.append(dict(scored[k][1]))

        while len(new_pop) < GA_POP:
            parent1 = rng.choice(mating_pool)
            parent2 = rng.choice(mating_pool)
            p1_obj = obj_cache[individual_signature(parent1, all_nodes_solve)]
            p2_obj = obj_cache[individual_signature(parent2, all_nodes_solve)]

            c1, c2 = crossover_intersection_scaega(
                parent1, parent2,
                p1_obj, p2_obj,
                all_nodes_solve,
                d_norm,
                rng,
                P_CROSS
            )

            mutate_individual(c1, all_nodes_solve, d_norm, rng, cur_p_mut_min, cur_p_mut_max)
            mutate_individual(c2, all_nodes_solve, d_norm, rng, cur_p_mut_min, cur_p_mut_max)
            new_pop.append(c1)
            if len(new_pop) < GA_POP:
                new_pop.append(c2)

        if best_ind is not None:
            best_sig = individual_signature(best_ind, all_nodes_solve)
            if not any(individual_signature(ind, all_nodes_solve) == best_sig for ind in new_pop):
                new_pop[-1] = dict(best_ind)

        population = new_pop

    pbar.close()
    if logger is not None and logger.stop_reason == "not_set":
        logger.set_stop(
            reason="finished_without_explicit_stop",
            outer_gen=gen,
        )

    if best_ind is None:
        best_ind = population[0]
        best_obj, best_cost, best_viol = evaluate_global_plan(all_nodes_solve, edges_by_scenario, best_ind, BETA_PENALTY, logger=None)

    print(f"\n[Run {run_id:02d}] best_global_cost={best_cost:.4f}, best_global_viol={best_viol:.6f}, best_global_obj={best_obj:.4f}", flush=True)

    # Write outputs (MUST MATCH YOUR OUTPUT FIELDS)
    out = node_df.copy()
    active_set_out = set(all_nodes_solve)

    def get_scale_id_for_output(nid: int) -> int:
        nid = int(nid)
        if nid in active_set_out:
            return int(best_ind[nid])
        return 0  # inactive fixed NONE

    out["seg_bits"] = out["id"].astype(int).map(lambda nid: seg_to_str(id_to_seg(get_scale_id_for_output(nid))))
    out["scale_id"] = out["id"].astype(int).map(lambda nid: get_scale_id_for_output(nid))
    out["scale"] = out["scale_id"].map(lambda sid: SCALE_LABEL[int(sid)])

    out["cap_2020"] = out["scale_id"].map(lambda sid: C_sess_day[int(sid)] * SCENARIO_DAYS["2020"])
    out["cap_2025"] = out["scale_id"].map(lambda sid: C_sess_day[int(sid)] * SCENARIO_DAYS["2025"])
    out["cap_2030"] = out["scale_id"].map(lambda sid: C_sess_day[int(sid)] * SCENARIO_DAYS["2030"])
    out["cap_2035"] = out["scale_id"].map(lambda sid: C_sess_day[int(sid)] * SCENARIO_DAYS["2035"])
    out["cap_2040"] = out["scale_id"].map(lambda sid: C_sess_day[int(sid)] * SCENARIO_DAYS["2040"])

    out["build_cost_usd"] = out["scale_id"].map(lambda sid: build_cost[int(sid)])
    total_days = sum(SCENARIO_DAYS.values())
    out["op_cost_usd_for_total_duration"] = out["scale_id"].map(
        lambda sid: op_cost_yr[int(sid)] * (total_days / 365.0)
    )

    out["is_active_solve"] = out["id"].astype(int).map(lambda nid: int(int(nid) in active_set_out))

    run_plan_path = os.path.join(LOG_DIR, f"run_{run_id:02d}_station_plan.csv")
    out.to_csv(run_plan_path, index=False, encoding="utf-8-sig")

    return best_ind, best_obj, best_cost, best_viol, run_plan_path


# =========================
# 9) Single-run / multi-run entry
# =========================
def run_one(
    run_id: int = 1,
    seed: Optional[int] = None,
    output_root: str = LOG_DIR,
):
    global LOG_DIR

    run_id = int(run_id)
    seed = _resolve_seed(run_id, seed)
    output_root = output_root or LOG_DIR
    os.makedirs(output_root, exist_ok=True)
    run_output_dir = _build_run_output_dir(output_root, run_id, seed)

    print(f"[RunConfig] run_id={run_id} seed={seed}", flush=True)
    print(f"[OutputDir] {run_output_dir}", flush=True)

    prev_log_dir = LOG_DIR
    LOG_DIR = run_output_dir
    try:
        logger = RunLogger(run_id=run_id, save_every_fe=SAVE_EVERY_FE, save_every_gen=SAVE_EVERY_GEN)
        logger.start()

        best_plan, best_obj, best_cost, best_viol, plan_path = run_scaega_global(
            logger=logger, seed=seed, run_id=run_id
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
    finally:
        LOG_DIR = prev_log_dir


def run_many(n_runs: int = 1, base_seed: Optional[int] = None, output_root: str = LOG_DIR):
    last_summary_path = ""
    for r in range(1, int(n_runs) + 1):
        run_seed = (int(base_seed) + r) if base_seed is not None else (1000 + int(r))
        last_summary_path = run_one(run_id=r, seed=run_seed, output_root=output_root)
    return last_summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SCAEGA global baseline.")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=LOG_DIR)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=None)
    args = parser.parse_args()

    if int(args.n_runs) > 1:
        run_many(n_runs=args.n_runs, base_seed=args.base_seed, output_root=args.output_root)
    else:
        run_one(run_id=args.run_id, seed=args.seed, output_root=args.output_root)

