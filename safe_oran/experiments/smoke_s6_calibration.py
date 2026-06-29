"""Smoke-check that S6 sits between easy SLA dynamics and S3 saturation."""

from __future__ import annotations

import json

import numpy as np

from safe_oran.envs.factory import make_constraint_env
from safe_oran.envs.legacy import EnvConfig
from safe_oran.experiments.configs import SCENARIOS


def collect_pmin_ratios(scenario: str, *, seed: int = 42) -> dict:
    env = make_constraint_env("M5_constraint_aware", scenario, seed=seed)
    obs, _ = env.reset(seed=seed)
    ratios = [float(obs[-1])]
    channels = []
    slas = []
    violations = []
    dproj = []
    terminated = truncated = False
    # Use a deliberately eMBB-heavy raw action; the shield reveals the active
    # reservation without letting backlog dominate the calibration smoke.
    raw_action = 0
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(raw_action)
        ratios.append(float(info["p_min_next"]) / float(EnvConfig().n_prb))
        channels.append(float(info["state_channel"]))
        slas.append(float(info["sla"]))
        violations.append(int(bool(info.get("urllc_violation", False))))
        dproj.append(float(info.get("D_proj", 0.0)))
    env.close()
    arr = np.asarray(ratios, dtype=float)
    ch = np.asarray(channels, dtype=float)
    return {
        "scenario": scenario,
        "n": int(arr.size),
        "mean_pmin_ratio": float(arr.mean()),
        "p05_pmin_ratio": float(np.percentile(arr, 5)),
        "p50_pmin_ratio": float(np.percentile(arr, 50)),
        "p95_pmin_ratio": float(np.percentile(arr, 95)),
        "moderate_share_0p3_0p8": float(np.mean((arr >= 0.3) & (arr <= 0.8))),
        "saturation_share_ge_0p95": float(np.mean(arr >= 0.95)),
        "channel_start": float(ch[0]) if ch.size else None,
        "channel_end": float(ch[-1]) if ch.size else None,
        "channel_monotone_nonincreasing": bool(np.all(np.diff(ch) <= 1e-9)) if ch.size else False,
        "sla_values": sorted({round(float(x), 4) for x in slas}),
        "violation_rate": float(np.mean(violations)) if violations else 0.0,
        "mean_D_proj": float(np.mean(dproj)) if dproj else 0.0,
    }


def main() -> int:
    if "S6_moderate_decay" not in SCENARIOS:
        raise SystemExit("S6_moderate_decay is missing from SCENARIOS")
    s6 = collect_pmin_ratios("S6_moderate_decay")
    s3 = collect_pmin_ratios("S3_channel_decay")
    passed = bool(
        s6["moderate_share_0p3_0p8"] >= 0.45
        and s6["saturation_share_ge_0p95"] < s3["saturation_share_ge_0p95"]
        and s6["saturation_share_ge_0p95"] <= 0.35
        and s6["channel_end"] <= 0.36
        and 0.999 in s6["sla_values"]
    )
    out = {"passed": passed, "S6_moderate_decay": s6, "S3_channel_decay": s3}
    print(json.dumps(out, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("S6 calibration smoke FAILED")
    print("S6 calibration smoke PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
