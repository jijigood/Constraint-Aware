"""
Phase 1 eval + gate. Reads the per-run JSONs written by train_baselines.py, adds the rule-based
(proportional-fair) baseline and the cross-regime distribution-shift eval, aggregates across seeds
(mean/std/bootstrap-CI), computes the Phase-1 gate (with the honest "DRL internalized safety" branch),
and renders figures. Writes summary.json + analysis.json + 05_figures/phase1/*.png.

Run after the training grid:
  .venv/bin/python 01_code/drl/eval_baselines.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DRL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DRL_DIR)
import drl_common as C  # noqa: E402
from slicing_env import EnvConfig, SLICES, run_episode, shield_none  # noqa: E402
from stable_baselines3 import PPO, DQN  # noqa: E402

ALGOS = {"ppo": PPO, "dqn": DQN}
RUNS_DIR = os.path.join(C.PROJ, "04_results", "phase1", "runs")
OUT_DIR = os.path.join(C.PROJ, "04_results", "phase1")
FIG_DIR = os.path.join(C.PROJ, "05_figures", "phase1")
METRICS = ["reward", "urllc_violation_rate", "embb_sla_rate", "mmtc_sla_rate", "fairness",
           "shield_correction_rate"]


# ---------------- rule-based baseline: proportional-fair (demand-share) allocation ----------------
def policy_proportional_fair(env, obs, rng=None):
    need = np.array([env._pending[s] + env.backlog[s] for s in SLICES], dtype=float)
    tot = need.sum() or 1.0
    target = need / tot * env.cfg.n_prb
    return int(np.argmin(np.abs(env.actions - target).sum(axis=1)))


def eval_rule_based(regime, seeds=(42, 43, 44)):
    runs = [run_episode(EnvConfig(), regime, policy_proportional_fair, shield_none, s) for s in seeds]
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


# ---------------- aggregation ----------------
def agg(values, seed=0):
    m, lo, hi = C.bootstrap_ci(values, rng_seed=seed)
    return {"mean": m, "std": float(np.std(values)), "lo": lo, "hi": hi, "n": len(values)}


def load_runs():
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def aggregate_in_regime(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["algo"], r["regime"], r["shield"]), []).append(r)
    out = {}
    for (algo, regime, shield), rs in groups.items():
        arm = {}
        for k in METRICS:
            arm[k] = agg([r["eval_metrics"][k] for r in rs])
        arm["train_violation_rate"] = agg([r["train_metrics"]["train_violation_rate"] for r in rs])
        arm["train_cum_urllc_violations"] = agg([r["evidence"]["train_cum_urllc_violations"] for r in rs])
        arm["n_seeds"] = len(rs)
        arm["all_paper_usable"] = all(r["paper_usable"] for r in rs)
        out[f"{algo}_{regime}_{shield}"] = arm
    return out


# ---------------- cross-regime distribution-shift eval ----------------
def cross_regime(runs):
    groups = {}
    for r in runs:
        tr = r["regime"]
        er = [x for x in C.REGIMES if x != tr]
        if not er:
            continue
        er = er[0]
        try:
            model = ALGOS[r["algo"]].load(r["evidence"]["checkpoint_path"], device="cpu")
            norm_fn = C.load_norm_stats(r["evidence"]["vecnorm_path"])
        except Exception as e:  # noqa: BLE001
            print(f"  cross-regime skip {r['algo']}_{tr}_{r['shield']}_s{r['seed']}: {e}")
            continue
        # static shield "travels" with its train-regime floor; oracle_margin/none are regime-agnostic
        floor = C.static_floor(tr)
        ev = C.deterministic_eval(model, norm_fn, er, r["shield"], floor, n_episodes=20)
        groups.setdefault((r["algo"], tr, er, r["shield"]), []).append(ev["means"])
    out = {}
    for (algo, tr, er, shield), evs in groups.items():
        out[f"{algo}_{tr}->{er}_{shield}"] = {
            "train_regime": tr, "eval_regime": er, "algo": algo, "shield": shield,
            "urllc_violation_rate": agg([e["urllc_violation_rate"] for e in evs]),
            "reward": agg([e["reward"] for e in evs]),
        }
    return out


# ---------------- gate ----------------
def compute_gate(in_reg, rule, regimes, algos):
    checks, notes = {}, []
    per_regime = {}
    drl_beats_rule = {}
    less_safe = {}
    internalizes = {}
    oracle_ub = {}
    for regime in regimes:
        rb = rule.get(regime, {}).get("reward")
        beats, safer, intern, ub = [], [], [], []
        for algo in algos:
            none = in_reg.get(f"{algo}_{regime}_none")
            if none is None:
                continue
            # (1) DRL-only beats rule-based reward (CI lower bound over the rule mean, or mean+margin)
            if rb is not None:
                beats.append(none["reward"]["lo"] > rb or none["reward"]["mean"] > rb + 0.02)
            v_none = none["urllc_violation_rate"]["mean"]
            tv_none = none["train_violation_rate"]["mean"]
            for sh in ("static", "oracle_margin"):
                arm = in_reg.get(f"{algo}_{regime}_{sh}")
                if arm is None:
                    continue
                v_sh = arm["urllc_violation_rate"]["mean"]
                tv_sh = arm["train_violation_rate"]["mean"]
                conv_gap = v_none > v_sh + 0.05
                explore_gap = (tv_none > 3 * max(tv_sh, 1e-6)) or (tv_none > 0.10 and tv_sh < 0.02)
                safer.append(conv_gap or explore_gap)
                intern.append((v_none <= v_sh + 0.02) and (v_none < 0.05))
                if sh == "oracle_margin":
                    v_static = in_reg.get(f"{algo}_{regime}_static", {}).get(
                        "urllc_violation_rate", {}).get("mean", 1.0)
                    ub.append(v_sh <= min(v_none, v_static) + 1e-3)
        per_regime[regime] = {
            "drl_beats_rule_based_reward": (all(beats) and len(beats) > 0),
            "drl_only_less_safe_than_shield": (any(safer) and len(safer) > 0),
            "drl_only_internalizes_safety": (all(intern) and len(intern) > 0),
            "oracle_is_safety_upper_bound": (all(ub) and len(ub) > 0),
        }
        drl_beats_rule[regime] = per_regime[regime]["drl_beats_rule_based_reward"]
        less_safe[regime] = per_regime[regime]["drl_only_less_safe_than_shield"]
        internalizes[regime] = per_regime[regime]["drl_only_internalizes_safety"]
        oracle_ub[regime] = per_regime[regime]["oracle_is_safety_upper_bound"]

    checks["drl_beats_rule_based_reward_all_regimes"] = all(drl_beats_rule.values()) and len(drl_beats_rule) > 0
    # shield value required at least in high_urllc (static-insufficient regime)
    key_regime = "high_urllc" if "high_urllc" in less_safe else (regimes[0] if regimes else None)
    checks["shield_value_in_key_regime"] = bool(less_safe.get(key_regime, False))
    checks["oracle_is_safety_upper_bound"] = all(oracle_ub.values()) and len(oracle_ub) > 0

    any_internalizes = any(internalizes.values())
    checks["drl_only_internalizes_safety_somewhere"] = any_internalizes

    gate_pass = bool(checks["drl_beats_rule_based_reward_all_regimes"]
                     and checks["shield_value_in_key_regime"]
                     and checks["oracle_is_safety_upper_bound"])

    if any_internalizes:
        notes.append("HONEST BRANCH: in >=1 regime the converged DRL-only policy internalizes safety "
                     "(viol<0.05, ~= shielded). Do NOT claim converged-safety as the shield's value there; "
                     "the shield's value is SAFE EXPLORATION (training-time violations) + distribution-shift "
                     "robustness. To probe a converged gap, raise env difficulty (demand stochasticity / "
                     "burst correlation / shorter episodes) -- NOT beta_violation (which makes safety easier). "
                     "This is a Phase-0-style re-validation, not forced in Phase 1.")
    if not gate_pass:
        notes.append("Gate did not pass on the primary axes; if neither converged nor safe-exploration gap "
                     "holds, the honest finding is 'shield gives no measurable benefit with a learner in this "
                     "sim' -> revisit env difficulty before Phase 2.")
    verdict = ("PASS -- DRL beats rule-based, shield reduces URLLC unsafety (converged and/or during "
               "exploration), oracle is the safety upper bound; proceed to Phase 2 (RAG-LLM shield)."
               if gate_pass else "INCONCLUSIVE/NEGATIVE -- see notes.")
    return {"checks": checks, "per_regime": per_regime, "PASS": gate_pass,
            "key_regime": key_regime, "verdict": verdict, "notes": notes}


# ---------------- figures (read aggregated numbers only) ----------------
def fig_safety_bars(in_reg, rule, regimes, algos, phase0):
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 4.2), squeeze=False)
    for j, regime in enumerate(regimes):
        ax = axes[0][j]
        labels, vals, errs = [], [], []
        for algo in algos:
            for sh in ("none", "static", "oracle_margin"):
                arm = in_reg.get(f"{algo}_{regime}_{sh}")
                if arm:
                    labels.append(f"{algo}\n{sh}")
                    vals.append(arm["urllc_violation_rate"]["mean"])
                    errs.append(arm["urllc_violation_rate"]["std"])
        x = np.arange(len(labels))
        ax.bar(x, vals, yerr=errs, color="#c0504d", alpha=0.85, capsize=3)
        if regime in rule:
            ax.axhline(rule[regime]["urllc_violation_rate"], ls="--", c="gray",
                       label=f"rule-based PF ({rule[regime]['urllc_violation_rate']:.2f})")
        oracle = phase0["regimes"].get(regime, {}).get("oracle_margin", {}).get("urllc_violation_rate")
        if oracle is not None:
            ax.axhline(oracle, ls=":", c="green", label=f"Phase0 oracle ({oracle:.2f})")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"URLLC violation rate — {regime}"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_safety_bars.png"), dpi=130); plt.close(fig)


def fig_safe_exploration(runs, regimes):
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 4.2), squeeze=False)
    for j, regime in enumerate(regimes):
        ax = axes[0][j]
        for r in runs:
            if r["regime"] != regime or r["seed"] != 42:
                continue
            sc = np.array(r["train_metrics"]["safety_curve"])
            if len(sc):
                ax.plot(sc[:, 0], sc[:, 1], label=f"{r['algo']}/{r['shield']}")
        ax.set_title(f"cumulative training URLLC violations — {regime}")
        ax.set_xlabel("timesteps"); ax.set_ylabel("cum. violations"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_safe_exploration.png"), dpi=130); plt.close(fig)


def fig_convergence(runs, regimes, algos):
    fig, axes = plt.subplots(len(algos), len(regimes), figsize=(6 * len(regimes), 4 * len(algos)),
                             squeeze=False)
    for i, algo in enumerate(algos):
        for j, regime in enumerate(regimes):
            ax = axes[i][j]
            for r in runs:
                if r["algo"] != algo or r["regime"] != regime or r["seed"] != 42:
                    continue
                rc = np.array(r["train_metrics"]["reward_curve"])
                if len(rc):
                    ax.plot(rc[:, 0], rc[:, 1], label=r["shield"])
            ax.set_title(f"{algo} — {regime}"); ax.set_xlabel("timesteps")
            ax.set_ylabel("train ep reward"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_convergence.png"), dpi=130); plt.close(fig)


def fig_pareto(in_reg, regimes, algos, phase0):
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 4.2), squeeze=False)
    for j, regime in enumerate(regimes):
        ax = axes[0][j]
        p0 = phase0["regimes"].get(regime, {})
        st = p0.get("static", {})
        if st:
            xs = [st[k]["reward"] for k in st]; ys = [st[k]["urllc_violation_rate"] for k in st]
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], "-o", c="gray", ms=3,
                    label="Phase0 static frontier", alpha=0.6)
        if "oracle_margin" in p0:
            ax.scatter(p0["oracle_margin"]["reward"], p0["oracle_margin"]["urllc_violation_rate"],
                       marker="*", s=160, c="green", label="Phase0 oracle_margin", zorder=5)
        marks = {"none": "x", "static": "s", "oracle_margin": "^"}
        for algo in algos:
            for sh in ("none", "static", "oracle_margin"):
                arm = in_reg.get(f"{algo}_{regime}_{sh}")
                if arm:
                    ax.scatter(arm["reward"]["mean"], arm["urllc_violation_rate"]["mean"],
                               marker=marks[sh], s=70, label=f"{algo}/{sh}")
        ax.set_title(f"reward vs URLLC violation — {regime}")
        ax.set_xlabel("reward (higher better)"); ax.set_ylabel("URLLC violation (lower better)")
        ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_reward_safety_pareto.png"), dpi=130); plt.close(fig)


def fig_cross_regime(cross):
    if not cross:
        return
    keys = list(cross)
    labels = [k.replace("_", "\n") for k in keys]
    v = [cross[k]["urllc_violation_rate"]["mean"] for k in keys]
    e = [cross[k]["urllc_violation_rate"]["std"] for k in keys]
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(keys)), 4.2))
    x = np.arange(len(keys))
    colors = ["#4f81bd" if cross[k]["shield"] == "oracle_margin" else "#c0504d" for k in keys]
    ax.bar(x, v, yerr=e, color=colors, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("URLLC violation rate (eval regime)"); ax.set_ylim(0, 1.05)
    ax.set_title("Cross-regime (train→eval) safety: static (red) breaks under shift, oracle_margin (blue) holds")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_cross_regime.png"), dpi=130); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("no run JSONs found in", RUNS_DIR); sys.exit(1)
    regimes = sorted({r["regime"] for r in runs}, key=lambda x: (x != "high_embb", x))
    algos = sorted({r["algo"] for r in runs})
    print(f"[eval] {len(runs)} runs; regimes={regimes} algos={algos}")

    with open(C.PHASE0_JSON) as f:
        phase0 = json.load(f)

    in_reg = aggregate_in_regime(runs)
    rule = {reg: eval_rule_based(reg) for reg in regimes}
    cross = cross_regime(runs)

    lib = runs[0]["lib"]
    summary = {"schema_version": C.SCHEMA_VERSION, "kind": "phase1_summary", "lib": lib,
               "device": "cpu", "regimes": regimes, "algos": algos,
               "in_regime": in_reg, "rule_based": rule, "cross_regime": cross,
               "all_paper_usable": all(r["paper_usable"] for r in runs)}
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    gate = compute_gate(in_reg, rule, regimes, algos)
    analysis = {"schema_version": C.SCHEMA_VERSION, "kind": "phase1_analysis", "gate": gate}
    with open(os.path.join(OUT_DIR, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    fig_safety_bars(in_reg, rule, regimes, algos, phase0)
    fig_safe_exploration(runs, regimes)
    fig_convergence(runs, regimes, algos)
    fig_pareto(in_reg, regimes, algos, phase0)
    fig_cross_regime(cross)

    # console report
    print("\n=== in-regime (seed mean) ===")
    for regime in regimes:
        print(f"\n# {regime}   rule-based PF: reward={rule[regime]['reward']:.3f} "
              f"urllc_viol={rule[regime]['urllc_violation_rate']:.3f}")
        print(f"{'arm':<26}{'reward':>9}{'urllc_viol':>12}{'train_viol':>12}{'embb_sla':>10}{'mmtc_sla':>10}")
        for algo in algos:
            for sh in ("none", "static", "oracle_margin"):
                a = in_reg.get(f"{algo}_{regime}_{sh}")
                if a:
                    print(f"{algo+'/'+sh:<26}{a['reward']['mean']:>9.3f}"
                          f"{a['urllc_violation_rate']['mean']:>12.3f}"
                          f"{a['train_violation_rate']['mean']:>12.3f}"
                          f"{a['embb_sla_rate']['mean']:>10.3f}{a['mmtc_sla_rate']['mean']:>10.3f}")
    print("\n=== GATE ===")
    for k, v in gate["checks"].items():
        print(f"  [{'x' if v else ' '}] {k}")
    print(f"\n  PASS={gate['PASS']}  key_regime={gate['key_regime']}")
    print(f"  verdict: {gate['verdict']}")
    for n in gate["notes"]:
        print(f"  note: {n}")
    print(f"\nartifacts: {OUT_DIR}/summary.json, analysis.json ; figures: {FIG_DIR}/")


if __name__ == "__main__":
    main()
