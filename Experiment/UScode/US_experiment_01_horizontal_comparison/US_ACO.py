import os
import time
import json
import math
import random
import threading
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import pulp


# =========================
# 0) Paths
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
LOG_DIR = os.path.join(SCRIPT_DIR, "aco_global_logs")
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

SCALE_ID_TO_SEG: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 1, 0),
    3: (0, 0, 1),
}

def id_to_seg(scale_id: int) -> Tuple[int, int, int]:
    return SCALE_ID_TO_SEG[int(scale_id)]

def seg_to_str(seg: Tuple[int, int, int]) -> str:
    return "".join(str(int(b)) for b in seg)

L_MAX = 3

C_sess_day = {0: 0, 1: 359, 2: 945, 3: 1904}
build_cost = {0: 0, 1: 850000, 2: 1750000, 3: 3200000}
op_cost_yr = {0: 0, 1: 25000, 2: 50000, 3: 90000}


# =========================
# 2) Knobs (ACO baseline)
# =========================
EDGE_WEIGHT_MIN = 1.0
BETA_PENALTY = 1e3

# ACO parameters
ACO_ANTS = 30

ALPHA_PHER = 1.0   # pheromone exponent
BETA_HEU = 2.0     # heuristic exponent
RHO = 0.2          # evaporation rate
Q_PHER = 1.0       # deposit constant (scaled by 1/obj)

# pheromone bounds (avoid collapse)
TAU_MIN = 1e-3
TAU_MAX = 10.0

# heuristic knobs
EPS_D = 1e-3
LAMBDA_COST = 1e-5    # cost penalty weight in heuristic

SAVE_EVERY_FE = 256
SAVE_EVERY_GEN = 1

# Unified FE budget for fair comparison across methods.
# 50 * 256 = 12800 FE.
MAX_TOTAL_FE = 50 * 256

# ---- NEW: terminal progress knobs ----
PRINT_EVERY_IT = 1          # 1=print every iteration (suggest 1/2/5/10)
PRINT_EVERY_FE_HINT = SAVE_EVERY_FE  # 0=disable; otherwise print every N FE (local best hint)


# =========================
# 3) RunLogger (same fields)
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
# 4) IO helpers (same behavior)
# =========================
def read_nodes(node_path: str) -> pd.DataFrame:
    df = pd.read_csv(node_path)
    required = {"id", "longitude", "latitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{node_path} missing columns: {missing}")
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
# 5) Evaluation (same LP model)
# =========================
def solution_cost_global(all_nodes_solve: List[int], plan: Dict[int, int]) -> float:
    build = 0.0
    op = 0.0
    for n in all_nodes_solve:
        sid = int(plan[int(n)])
        build += build_cost[sid]
    for tname, days in SCENARIO_DAYS.items():
        frac = days / 365.0
        for n in all_nodes_solve:
            sid = int(plan[int(n)])
            op += op_cost_yr[sid] * frac
    return build + op

def evaluate_global_plan(
    all_nodes_solve: List[int],
    edges_by_scenario: Dict[str, pd.DataFrame],
    plan: Dict[int, int],
    beta_penalty: float,
    logger: Optional[RunLogger] = None
) -> Tuple[float, float, float]:
    global_cost = solution_cost_global(all_nodes_solve, plan)

    node_set = set(all_nodes_solve)

    total_slack = 0.0
    for tname, edf in edges_by_scenario.items():
        days_t = SCENARIO_DAYS[tname]
        cap = {int(i): C_sess_day[int(plan[int(i)])] * days_t for i in all_nodes_solve}

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

    obj = global_cost + beta_penalty * total_slack

    if logger is not None:
        logger.update_local_fe(obj=obj, cost=global_cost, viol=total_slack)

    return obj, global_cost, total_slack


# =========================
# 6) d_norm and heuristic
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

def build_eta_for_node(di: float) -> np.ndarray:
    eta = np.zeros((4,), dtype=np.float64)
    eta[0] = (EPS_D + (1.0 - di))
    for l in [1, 2, 3]:
        eta[l] = (EPS_D + di) ** l / (1.0 + LAMBDA_COST * float(build_cost[l]))
    eta = np.maximum(eta, 1e-12)
    return eta


# =========================
# 7) ACO core
# =========================
def sample_scale(tau_row: np.ndarray, eta_row: np.ndarray, rng: random.Random) -> int:
    w = (tau_row ** ALPHA_PHER) * (eta_row ** BETA_HEU)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return int(rng.choice([0, 1, 2, 3]))
    p = (w / s).tolist()
    return int(rng.choices([0, 1, 2, 3], weights=p, k=1)[0])

def construct_ant_solution(active_nodes: List[int], tau: np.ndarray, eta: np.ndarray, rng: random.Random) -> Dict[int, int]:
    plan: Dict[int, int] = {}
    for i, nid in enumerate(active_nodes):
        l = sample_scale(tau[i], eta[i], rng)
        plan[int(nid)] = int(l)
    return plan

def evaporate(tau: np.ndarray) -> np.ndarray:
    tau = (1.0 - RHO) * tau
    tau = np.clip(tau, TAU_MIN, TAU_MAX)
    return tau

def deposit(tau: np.ndarray, best_plan: Dict[int, int], active_nodes: List[int], best_obj: float):
    delta = Q_PHER / max(float(best_obj), 1e-12)
    for i, nid in enumerate(active_nodes):
        l = int(best_plan[int(nid)])
        tau[i, l] += delta
    np.clip(tau, TAU_MIN, TAU_MAX, out=tau)


def plan_change_ratio(all_nodes_solve: List[int], a: Dict[int, int], b: Dict[int, int]) -> float:
    changed = 0
    for nid in all_nodes_solve:
        if int(a[int(nid)]) != int(b[int(nid)]):
            changed += 1
    return changed / max(1, len(all_nodes_solve))


# =========================
# 8) One run of ACO
# =========================
def run_aco_global(logger: Optional[RunLogger], seed: int, run_id: int = 1):
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

    active_set = set(active_nodes)
    for sname in list(edges_by_scenario.keys()):
        edf = edges_by_scenario[sname]
        edf = edf[edf["u"].isin(active_set) & edf["v"].isin(active_set)].copy()
        edf.reset_index(drop=True, inplace=True)
        edges_by_scenario[sname] = edf
        print(f"  - {sname}: filtered edges (active subgraph)={len(edf)}", flush=True)

    all_nodes_solve = active_nodes

    # heuristic
    d_norm = compute_d_norm(all_nodes_solve, edges_by_scenario)
    eta = np.zeros((len(all_nodes_solve), 4), dtype=np.float64)
    for i, nid in enumerate(all_nodes_solve):
        eta[i] = build_eta_for_node(float(d_norm[int(nid)]))

    # pheromone init
    tau = np.full((len(all_nodes_solve), 4), 1.0, dtype=np.float64)

    best_plan = None
    best_obj = float("inf")
    best_cost = float("inf")
    best_viol = float("inf")

    print("[ACO] Pheromone and heuristic initialized.", flush=True)

    print("[ACO] Iterate under FE budget ...", flush=True)
    pbar = tqdm(desc=f"ACO Iters (run {run_id:02d})", dynamic_ncols=True)

    it = 0

    while True:
        if (logger is not None) and (logger.fe >= MAX_TOTAL_FE):
            logger.set_stop(
                reason="max_total_fe_reached",
                outer_gen=it,
            )
            print(f"\n[Stop] reached max_total_fe={MAX_TOTAL_FE}.", flush=True)
            break

        it += 1
        pbar.update(1)
        if logger is not None:
            logger.set_outer_gen(it)

        iter_best_plan = None
        iter_best_obj = float("inf")
        iter_best_cost = float("inf")
        iter_best_viol = float("inf")

        # construct & evaluate ants
        for k in range(ACO_ANTS):
            if (logger is not None) and (logger.fe >= MAX_TOTAL_FE):
                break
            plan = construct_ant_solution(all_nodes_solve, tau, eta, rng)
            obj, cost, viol = evaluate_global_plan(all_nodes_solve, edges_by_scenario, plan, BETA_PENALTY, logger=logger)

            # ---- NEW: FE hint print (optional) ----
            if PRINT_EVERY_FE_HINT and (logger is not None) and (logger.fe % PRINT_EVERY_FE_HINT == 0):
                print(
                    f"[Run {run_id:02d} | FE {logger.fe}] "
                    f"best_local_obj={logger.best_local_obj:.2f}  "
                    f"best_local_cost={logger.best_local_cost:.2f}  "
                    f"best_local_viol={logger.best_local_viol:.4f}",
                    flush=True
                )

            if obj < iter_best_obj - 1e-9:
                iter_best_obj, iter_best_cost, iter_best_viol = obj, cost, viol
                iter_best_plan = plan

        # update global best
        improved = False
        if iter_best_obj < best_obj - 1e-9:
            best_obj, best_cost, best_viol = iter_best_obj, iter_best_cost, iter_best_viol
            best_plan = dict(iter_best_plan)
            improved = True

        # log gen point (use iter best for "current")
        if logger is not None:
            logger.record_global_gen_point(iter_best_obj, iter_best_cost, iter_best_viol)

        # pheromone update
        tau = evaporate(tau)
        deposit(tau, iter_best_plan, all_nodes_solve, iter_best_obj)

        # ---- NEW: tqdm postfix (real-time) ----
        pbar.set_postfix({
            "iter_best": f"{iter_best_obj:.2e}",
            "iter_cost": f"{iter_best_cost:.2e}",
            "iter_viol": f"{iter_best_viol:.2e}",
            "best_obj": f"{best_obj:.2e}",
            "best_cost": f"{best_cost:.2e}",
            "best_viol": f"{best_viol:.2e}",
            "fe": (logger.fe if logger is not None else 0),
            "imp": ("Y" if improved else "N"),
        })

        # ---- NEW: periodic explicit print (like your GCPCC logs) ----
        if (PRINT_EVERY_IT is not None) and (PRINT_EVERY_IT > 0) and (it % PRINT_EVERY_IT == 0):
            print(
                f"[Run {run_id:02d} | Iter {it}] "
                f"iter_best: cost={iter_best_cost:.2f}  viol={iter_best_viol:.4f}  obj={iter_best_obj:.2f} | "
                f"best_so_far: cost={best_cost:.2f}  viol={best_viol:.4f}  obj={best_obj:.2f}",
                flush=True
            )

        fe_now = (logger.fe if logger is not None else 0)
        if (logger is not None) and (fe_now >= MAX_TOTAL_FE):
            logger.set_stop(
                reason="max_total_fe_reached",
                outer_gen=it,
            )
            print(f"\n[Stop] reached max_total_fe={MAX_TOTAL_FE}.", flush=True)
            break

    pbar.close()
    if logger is not None and logger.stop_reason == "not_set":
        logger.set_stop(
            reason="finished_without_explicit_stop",
            outer_gen=it,
        )

    print(f"\n[Run {run_id:02d}] best_global_cost={best_cost:.4f}, best_global_viol={best_viol:.6f}, best_global_obj={best_obj:.4f}", flush=True)

    # Output plan CSV with same columns
    out = node_df.copy()
    active_set_out = set(all_nodes_solve)

    def get_scale_id_for_output(nid: int) -> int:
        nid = int(nid)
        if nid in active_set_out:
            return int(best_plan[nid])
        return 0

    out["seg_bits"] = out["id"].astype(int).map(lambda nid: seg_to_str(id_to_seg(get_scale_id_for_output(nid))))
    out["scale_id"] = out["id"].astype(int).map(lambda nid: get_scale_id_for_output(nid))
    out["scale"] = out["scale_id"].map(lambda sid: SCALE_LABEL[int(sid)])

    for sname in SCENARIO_FILES.keys():
        out[f"cap_{sname}"] = out["scale_id"].map(lambda sid, t=sname: C_sess_day[int(sid)] * SCENARIO_DAYS[t])

    out["build_cost_usd"] = out["scale_id"].map(lambda sid: build_cost[int(sid)])
    total_days = sum(SCENARIO_DAYS.values())
    out["op_cost_usd_for_total_duration"] = out["scale_id"].map(
        lambda sid: op_cost_yr[int(sid)] * (total_days / 365.0)
    )

    out["is_active_solve"] = out["id"].astype(int).map(lambda nid: int(int(nid) in active_set_out))

    run_plan_path = os.path.join(LOG_DIR, f"run_{run_id:02d}_station_plan.csv")
    out.to_csv(run_plan_path, index=False, encoding="utf-8-sig")

    return best_plan, best_obj, best_cost, best_viol, run_plan_path


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

        best_plan, best_obj, best_cost, best_viol, plan_path = run_aco_global(
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
    parser = argparse.ArgumentParser(description="Run ACO global baseline.")
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

