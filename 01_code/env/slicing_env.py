"""
Lightweight O-RAN network-slicing environment for RAG-conditioned safe DRL (Theme A, Phase 0).

Pure-numpy. Exposes a gymnasium-style reset/step but does NOT import gymnasium, so the Phase-0
offline headroom oracle (the __main__ self-test) runs on CPU with zero extra installs. A thin
Gymnasium wrapper is added in Phase 1 for stable_baselines3.

Three slices share a fixed PRB pool:
  - eMBB   : throughput SLA   (wants many PRBs; high, bursty demand)
  - URLLC  : latency  SLA     (must meet demand *this slot* or it is delayed -> SLA violation)
  - mMTC   : access   SLA     (wants a served fraction of many small flows)

The headline safety concern: a reward-seeking controller, tempted by eMBB throughput, starves URLLC
during eMBB bursts -> URLLC latency-SLA violations. A safety *shield* reserves PRBs for URLLC. A FIXED
(static) reservation is either wasteful (low-URLLC regime) or unsafe (high-URLLC regime); a load-aware
DYNAMIC reservation (the oracle, later approximated by a RAG-grounded LLM from the SLA text + observed
load) reaches ~0 violations at lower reward cost. Phase 0 tests whether that headroom exists.

Run the offline oracle:
    /opt/anaconda3/bin/python 01_code/env/slicing_env.py
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field, asdict

import numpy as np

SCHEMA_VERSION = "safe_drl_v0"

# ----------------------------------------------------------------------------- config
SLICES = ("embb", "urllc", "mmtc")


@dataclass
class SliceSpec:
    name: str
    se: float                 # spectral efficiency, Mbps per PRB at unit channel gain
    base_demand: float        # baseline offered load (Mbps)
    sla_min: float            # eMBB/mMTC: min served (Mbps / fraction); URLLC: unused (latency below)


@dataclass
class EnvConfig:
    n_prb: int = 100
    prb_step: int = 10        # action granularity: integer PRB splits in multiples of prb_step
    episode_len: int = 200
    # per-slice physical model
    se: dict = field(default_factory=lambda: {"embb": 1.0, "urllc": 0.6, "mmtc": 0.4})
    base_demand: dict = field(default_factory=lambda: {"embb": 45.0, "urllc": 18.0, "mmtc": 12.0})
    # SLA targets
    embb_rate_min: float = 30.0      # eMBB throughput SLA (Mbps)
    mmtc_access_min: float = 0.80    # mMTC served-fraction SLA
    # URLLC latency SLA: demand must be served within the slot (cap >= demand+backlog). violation if not.
    urllc_backlog_tol: float = 0.0   # Mbps of tolerable unmet URLLC before it counts as a violation
    # reward weights (report components separately; scalar is only a summary)
    w_embb: float = 1.0
    w_mmtc: float = 0.6
    w_urllc_ok: float = 0.5
    beta_violation: float = 2.0      # URLLC violation penalty
    # channel
    channel_mean: float = 1.0
    channel_amp: float = 0.15        # slow sinusoid
    channel_noise: float = 0.05


# ----------------------------------------------------------------------------- traffic
REGIMES = ("balanced", "high_embb", "high_urllc", "bursty")


def _diurnal(t: int, period: int) -> float:
    return 1.0 + 0.25 * math.sin(2 * math.pi * t / max(period, 1))


def make_demand(cfg: EnvConfig, regime: str, t: int, rng: np.random.Generator) -> dict:
    """Per-slice offered load (Mbps) at step t for a given traffic regime."""
    d = dict(cfg.base_demand)
    diur = _diurnal(t, cfg.episode_len)
    if regime == "high_embb":
        d["embb"] *= 1.8
    elif regime == "high_urllc":
        d["urllc"] *= 2.2
    elif regime == "bursty":
        # correlated bursts: eMBB and URLLC spike together (the dangerous case for the shield)
        if rng.random() < 0.25:
            d["embb"] *= 2.2
            d["urllc"] *= 1.8
    # apply diurnal + multiplicative noise
    for k in d:
        d[k] *= diur * (1.0 + 0.10 * rng.standard_normal())
        d[k] = max(0.0, d[k])
    return d


# ----------------------------------------------------------------------------- env
class SlicingEnv:
    """gymnasium-style slicing env (numpy only)."""

    def __init__(self, cfg: EnvConfig | None = None, regime: str = "balanced", seed: int = 0):
        self.cfg = cfg or EnvConfig()
        self.regime = regime
        self.rng = np.random.default_rng(seed)
        self.actions = self._build_action_set()
        self.n_actions = len(self.actions)
        self.t = 0
        self.backlog = {s: 0.0 for s in SLICES}

    # --- action space: integer PRB allocations summing to n_prb, in multiples of prb_step ---
    def _build_action_set(self) -> np.ndarray:
        step, total = self.cfg.prb_step, self.cfg.n_prb
        units = total // step
        combos = [c for c in itertools.product(range(units + 1), repeat=len(SLICES)) if sum(c) == units]
        return np.array([[c[i] * step for i in range(len(SLICES))] for c in combos], dtype=int)

    def channel(self) -> float:
        c = self.cfg.channel_mean + self.cfg.channel_amp * math.sin(2 * math.pi * self.t / 40.0)
        c += self.cfg.channel_noise * self.rng.standard_normal()
        return float(max(0.2, c))

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.backlog = {s: 0.0 for s in SLICES}
        self._pending = self._sample_demand()
        return self._obs(), {}

    def _sample_demand(self) -> dict:
        return make_demand(self.cfg, self.regime, self.t, self.rng)

    def _obs(self) -> np.ndarray:
        g = self._last_channel if hasattr(self, "_last_channel") else self.cfg.channel_mean
        d = self._pending
        return np.array(
            [d["embb"], d["urllc"], d["mmtc"],
             self.backlog["embb"], self.backlog["urllc"], self.backlog["mmtc"],
             g, self.t / self.cfg.episode_len],
            dtype=np.float32,
        )

    def min_prb_needed(self, slice_name: str, g: float | None = None) -> int:
        """Load-aware PRBs to fully serve a slice's current demand+backlog this slot (oracle constraint)."""
        if g is None:
            g = self._last_channel if hasattr(self, "_last_channel") else self.cfg.channel_mean
        need = self._pending[slice_name] + self.backlog[slice_name]
        prb = need / max(self.cfg.se[slice_name] * g, 1e-6)
        return int(min(self.cfg.n_prb, math.ceil(prb)))

    def step(self, action_idx: int):
        cfg = self.cfg
        g = self.channel()
        self._last_channel = g
        alloc = self.actions[int(action_idx)]
        prb = {s: int(alloc[i]) for i, s in enumerate(SLICES)}
        d = self._pending

        served, viol = {}, {}
        for s in SLICES:
            cap = prb[s] * cfg.se[s] * g
            demand_tot = d[s] + self.backlog[s]
            srv = min(cap, demand_tot)
            served[s] = srv
            self.backlog[s] = max(0.0, demand_tot - cap)

        # URLLC: latency SLA -> demand must be met this slot
        urllc_unmet = (d["urllc"] + self.backlog["urllc"] * 0.0) - served["urllc"]
        urllc_violation = urllc_unmet > cfg.urllc_backlog_tol + 1e-6

        # SLA satisfaction per slice
        embb_ok = served["embb"] >= cfg.embb_rate_min
        mmtc_frac = served["mmtc"] / max(d["mmtc"], 1e-6)
        mmtc_ok = mmtc_frac >= cfg.mmtc_access_min

        # reward components (reported separately)
        embb_norm = served["embb"] / max(cfg.base_demand["embb"], 1e-6)
        mmtc_norm = min(1.0, mmtc_frac)
        reward = (cfg.w_embb * embb_norm + cfg.w_mmtc * mmtc_norm
                  + cfg.w_urllc_ok * (0.0 if urllc_violation else 1.0)
                  - cfg.beta_violation * (1.0 if urllc_violation else 0.0))

        comp = {
            "reward": float(reward),
            "embb_served": served["embb"], "urllc_served": served["urllc"], "mmtc_served": served["mmtc"],
            "urllc_violation": bool(urllc_violation),
            "embb_ok": bool(embb_ok), "mmtc_ok": bool(mmtc_ok),
            "prb_embb": prb["embb"], "prb_urllc": prb["urllc"], "prb_mmtc": prb["mmtc"],
            "channel": g,
        }

        self.t += 1
        done = self.t >= cfg.episode_len
        if not done:
            self._pending = self._sample_demand()
        return self._obs(), float(reward), done, False, comp


# ----------------------------------------------------------------------------- shields
def shield_none(env: SlicingEnv, action_idx: int) -> int:
    return action_idx


def _project_to_min_urllc(env: SlicingEnv, action_idx: int, min_urllc: int) -> int:
    """Return the nearest valid action that gives URLLC >= min_urllc, taking PRBs from the largest others."""
    min_urllc = int(min(env.cfg.n_prb, max(0, min_urllc)))
    cur = env.actions[int(action_idx)]
    if cur[1] >= min_urllc:
        return action_idx
    # choose, among actions meeting the URLLC floor, the one closest (L1) to the proposed alloc
    mask = env.actions[:, 1] >= min_urllc
    cand = env.actions[mask]
    d = np.abs(cand - cur).sum(axis=1)
    best = cand[np.argmin(d)]
    return int(np.where((env.actions == best).all(axis=1))[0][0])


def shield_static(min_urllc: int):
    def f(env: SlicingEnv, action_idx: int) -> int:
        return _project_to_min_urllc(env, action_idx, min_urllc)
    return f


def shield_dynamic_oracle(env: SlicingEnv, action_idx: int) -> int:
    """Minimal load-aware oracle: reserve exactly the PRBs URLLC needs this slot (quantized up).
    No reliability margin -> shows the *reward potential* of a load-aware constraint."""
    need = env.min_prb_needed("urllc")
    step = env.cfg.prb_step
    need_q = int(math.ceil(need / step) * step)
    return _project_to_min_urllc(env, action_idx, need_q)


def shield_dynamic_oracle_margin(reliability: float = 0.99):
    """Load-aware oracle WITH an SLA-reliability margin: reserve demand at a pessimistic channel
    quantile so in-slot channel dips don't breach the SLA. This is what the RAG-LLM shield will
    approximate -- the SLA's reliability target maps to the channel margin. Higher reliability ->
    more pessimistic channel -> larger reservation -> fewer violations at some reward cost."""
    def f(env: SlicingEnv, action_idx: int) -> int:
        cfg = env.cfg
        # pessimistic channel: mean minus (slow amplitude + a reliability-scaled noise allowance)
        z = 1.0 + 1.5 * reliability            # ~99% -> ~2.5 sigma of channel noise
        g_pess = max(0.2, cfg.channel_mean - cfg.channel_amp - z * cfg.channel_noise)
        need = env.min_prb_needed("urllc", g=g_pess)
        step = cfg.prb_step
        need_q = int(math.ceil(need / step) * step)
        return _project_to_min_urllc(env, action_idx, need_q)
    return f


# ----------------------------------------------------------------------------- policies
def policy_random(env: SlicingEnv, obs, rng) -> int:
    return int(rng.integers(env.n_actions))


def policy_throughput_greedy(env: SlicingEnv, obs, rng) -> int:
    """Maximize immediate throughput utility, IGNORING the URLLC penalty (mimics unsafe exploration)."""
    g = env._last_channel if hasattr(env, "_last_channel") else env.cfg.channel_mean
    d = env._pending
    best_i, best_u = 0, -1e9
    for i, alloc in enumerate(env.actions):
        prb = {s: int(alloc[j]) for j, s in enumerate(SLICES)}
        embb = min(prb["embb"] * env.cfg.se["embb"] * g, d["embb"] + env.backlog["embb"])
        mmtc = min(prb["mmtc"] * env.cfg.se["mmtc"] * g, d["mmtc"] + env.backlog["mmtc"])
        u = env.cfg.w_embb * embb / max(env.cfg.base_demand["embb"], 1e-6) + env.cfg.w_mmtc * min(
            1.0, mmtc / max(d["mmtc"], 1e-6))
        if u > best_u:
            best_u, best_i = u, i
    return best_i


def policy_reward_greedy(env: SlicingEnv, obs, rng) -> int:
    """Myopic argmax of the TRUE immediate reward (knows the penalty, but only one-step)."""
    g = env._last_channel if hasattr(env, "_last_channel") else env.cfg.channel_mean
    d, cfg = env._pending, env.cfg
    best_i, best_r = 0, -1e9
    for i, alloc in enumerate(env.actions):
        prb = {s: int(alloc[j]) for j, s in enumerate(SLICES)}
        embb = min(prb["embb"] * cfg.se["embb"] * g, d["embb"] + env.backlog["embb"])
        mmtc = min(prb["mmtc"] * cfg.se["mmtc"] * g, d["mmtc"] + env.backlog["mmtc"])
        urllc_cap = prb["urllc"] * cfg.se["urllc"] * g
        viol = (d["urllc"] - urllc_cap) > cfg.urllc_backlog_tol + 1e-6
        r = (cfg.w_embb * embb / max(cfg.base_demand["embb"], 1e-6)
             + cfg.w_mmtc * min(1.0, mmtc / max(d["mmtc"], 1e-6))
             + cfg.w_urllc_ok * (0.0 if viol else 1.0) - cfg.beta_violation * (1.0 if viol else 0.0))
        if r > best_r:
            best_r, best_i = r, i
    return best_i


POLICIES = {
    "random": policy_random,
    "throughput_greedy": policy_throughput_greedy,
    "reward_greedy": policy_reward_greedy,
}


# ----------------------------------------------------------------------------- rollout + metrics
def jain(xs) -> float:
    xs = np.asarray(xs, dtype=float)
    if xs.sum() <= 0:
        return 0.0
    return float((xs.sum() ** 2) / (len(xs) * (xs ** 2).sum() + 1e-12))


def run_episode(cfg, regime, policy_fn, shield_fn, seed):
    env = SlicingEnv(cfg, regime=regime, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 9999)
    rew, viol, embb_ok, mmtc_ok = [], [], [], []
    prb_used = []
    done = False
    while not done:
        a = policy_fn(env, obs, rng)
        a = shield_fn(env, a)
        obs, r, done, _, comp = env.step(a)
        rew.append(r)
        viol.append(comp["urllc_violation"])
        embb_ok.append(comp["embb_ok"])
        mmtc_ok.append(comp["mmtc_ok"])
        prb_used.append([comp["prb_embb"], comp["prb_urllc"], comp["prb_mmtc"]])
    prb_used = np.array(prb_used)
    return {
        "reward": float(np.mean(rew)),
        "urllc_violation_rate": float(np.mean(viol)),
        "embb_sla_rate": float(np.mean(embb_ok)),
        "mmtc_sla_rate": float(np.mean(mmtc_ok)),
        "fairness": jain(prb_used.mean(axis=0)),
    }


def evaluate(cfg, regime, policy_name, shield_fn, seeds=(42, 43, 44)):
    runs = [run_episode(cfg, regime, POLICIES[policy_name], shield_fn, s) for s in seeds]
    keys = runs[0].keys()
    return {k: float(np.mean([r[k] for r in runs])) for k in keys}


# ----------------------------------------------------------------------------- Phase-0 headroom oracle
def _dominated_by_static(point: dict, static: dict, static_grid) -> bool:
    """Is `point` Pareto-dominated by ANY static point? (a static with >= reward AND <= violation)."""
    for p in static_grid:
        s = static[str(p)]
        if (s["reward"] >= point["reward"] - 1e-6
                and s["urllc_violation_rate"] <= point["urllc_violation_rate"] + 1e-6
                and (s["reward"] > point["reward"] + 1e-6
                     or s["urllc_violation_rate"] < point["urllc_violation_rate"] - 1e-6)):
            return True
    return False


def phase0_headroom(out_path: str | None = None) -> dict:
    cfg = EnvConfig()
    policy = "throughput_greedy"   # the unsafe, reward-seeking behavior DRL exploration mimics
    static_grid = list(range(0, 61, 10))
    regimes = ["high_embb", "high_urllc"]

    results = {"schema_version": SCHEMA_VERSION, "paper_usable": False, "kind": "offline_headroom_oracle",
               "policy": policy, "static_grid": static_grid, "regimes": {}}

    for regime in regimes:
        none = evaluate(cfg, regime, policy, shield_none)
        static = {p: evaluate(cfg, regime, policy, shield_static(p)) for p in static_grid}
        oracle_min = evaluate(cfg, regime, policy, shield_dynamic_oracle)
        oracle_margin = evaluate(cfg, regime, policy, shield_dynamic_oracle_margin(reliability=0.99))
        # safe static = smallest reservation reaching <=1% violations, else the lowest-violation one
        safe_ps = [p for p in static_grid if static[p]["urllc_violation_rate"] <= 0.01]
        safe_p = (min(safe_ps) if safe_ps
                  else min(static_grid, key=lambda p: (round(static[p]["urllc_violation_rate"], 4),
                                                       -static[p]["reward"])))
        results["regimes"][regime] = {
            "none": none,
            "static": {str(p): static[p] for p in static_grid},
            "oracle_min": oracle_min,
            "oracle_margin": oracle_margin,
            "safe_static_p": safe_p,
            "min_static_violation": float(min(static[p]["urllc_violation_rate"] for p in static_grid)),
        }

    # ---- gate checks ----
    g = results["regimes"]
    # (1) unshielded reward-seeking is genuinely unsafe
    none_bad = all(g[r]["none"]["urllc_violation_rate"] >= 0.15 for r in regimes)
    # (2a) no single fixed reservation is best across regimes
    static_regime_specific = len({g[r]["safe_static_p"] for r in regimes}) > 1
    # (2b) deploying one regime's safe reservation in the other loses safety or reward
    rA, rB = regimes
    pA, pB = g[rA]["safe_static_p"], g[rB]["safe_static_p"]
    cross_regime_gap = (g[rA]["static"][str(pB)]["reward"] < g[rA]["static"][str(pA)]["reward"] - 1e-6
                        or g[rB]["static"][str(pA)]["urllc_violation_rate"] > 0.01) if pA != pB else False
    # (3) a load-aware oracle lies on/beyond the static Pareto frontier (not dominated) in every regime,
    #     and the margin-oracle reaches near-safety better than the best static can
    oracle_on_frontier = all(
        not _dominated_by_static(g[r]["oracle_min"], g[r]["static"], static_grid)
        and not _dominated_by_static(g[r]["oracle_margin"], g[r]["static"], static_grid)
        for r in regimes
    )
    # margin-oracle is at least as safe as the best static can be, in every regime (often far safer)
    oracle_safer = all(
        g[r]["oracle_margin"]["urllc_violation_rate"] <= g[r]["min_static_violation"] + 1e-6
        for r in regimes
    )
    gate_pass = bool(none_bad and static_regime_specific and cross_regime_gap
                     and oracle_on_frontier and oracle_safer)

    results["gate"] = {
        "none_violation_ge_15pct": bool(none_bad),
        "best_static_differs_by_regime": bool(static_regime_specific),
        "cross_regime_fixed_reservation_suboptimal": bool(cross_regime_gap),
        "oracle_on_or_beyond_static_frontier": bool(oracle_on_frontier),
        "margin_oracle_at_least_as_safe_as_best_static": bool(oracle_safer),
        "PASS": gate_pass,
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


def _print_report(res: dict):
    print(f"\n{'='*78}\nPHASE 0 — slicing headroom oracle  (policy={res['policy']}, seeds 42/43/44)\n{'='*78}")
    for regime, g in res["regimes"].items():
        print(f"\n### regime = {regime}")
        print(f"{'shield':<22}{'reward':>9}{'urllc_viol':>12}{'embb_sla':>10}{'mmtc_sla':>10}{'fair':>7}")
        def row(name, m):
            print(f"{name:<22}{m['reward']:>9.3f}{m['urllc_violation_rate']:>12.3f}"
                  f"{m['embb_sla_rate']:>10.3f}{m['mmtc_sla_rate']:>10.3f}{m['fairness']:>7.3f}")
        row("none", g["none"])
        for p in res["static_grid"]:
            tag = f"static({p})"
            if p == g["safe_static_p"]:
                tag += " *safe"
            row(tag, g["static"][str(p)])
        row("oracle_min", g["oracle_min"])
        row("oracle_margin", g["oracle_margin"])
    gate = res["gate"]
    print(f"\n{'-'*78}\nGATE")
    for k, v in gate.items():
        if k != "PASS":
            print(f"  [{'x' if v else ' '}] {k}")
    print(f"\n  ==> PHASE 0 GATE: {'PASS — headroom exists, proceed to DRL baselines' if gate['PASS'] else 'FAIL — report static-shield-suffices negative'}")
    print(f"{'-'*78}")


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "..", "04_results", "phase0_headroom.json")
    out = os.path.abspath(out)
    res = phase0_headroom(out_path=out)
    _print_report(res)
    print(f"\nartifact: {out}")
