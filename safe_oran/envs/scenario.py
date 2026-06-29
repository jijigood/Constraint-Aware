"""Scenario-specific wrappers layered on top of the legacy Gym env."""

from __future__ import annotations

from collections.abc import Mapping

from .legacy import EnvConfig, SlicingGymEnv


def linear_decay_value(t: int, *, start: float, end: float, start_t: int, end_t: int) -> float:
    """Piecewise-linear channel profile used by S3/S5 V4 scenarios."""
    if t <= start_t:
        return float(start)
    if t >= end_t:
        return float(end)
    frac = (t - start_t) / max(end_t - start_t, 1)
    return float(start + frac * (end - start))


class ScenarioGymEnv(SlicingGymEnv):
    """Legacy env with optional scenario-level channel override.

    The wrapper changes only the channel process for the new ``safe_oran``
    scenarios. Legacy ``01_code`` experiments still use the original env.
    """

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        *,
        regime: str = "balanced",
        scenario_cfg: Mapping | None = None,
        shield_fn=None,
        seed: int | None = None,
    ):
        self.scenario_cfg = dict(scenario_cfg or {})
        super().__init__(cfg=cfg, regime=regime, shield_fn=shield_fn, seed=seed)
        self._install_demand_profile()
        self._install_channel_profile()

    def _channel_profile_value(self, t: int) -> float | None:
        if self.scenario_cfg.get("channel_profile", "default") != "linear_decay":
            return None
        return linear_decay_value(
            int(t),
            start=float(self.scenario_cfg["channel_start"]),
            end=float(self.scenario_cfg["channel_end"]),
            start_t=int(self.scenario_cfg["channel_start_t"]),
            end_t=int(self.scenario_cfg["channel_end_t"]),
        )

    def _install_channel_profile(self) -> None:
        if self.scenario_cfg.get("channel_profile", "default") != "linear_decay":
            return

        def channel_override() -> float:
            return float(self._channel_profile_value(self.inner.t))

        self.inner.channel = channel_override
        self.inner._last_channel = float(self._channel_profile_value(self.inner.t))

    def _install_demand_profile(self) -> None:
        if self.scenario_cfg.get("demand_profile", "default") != "mild_urllc_burst":
            return
        if not hasattr(self.inner, "_scenario_base_sample_demand"):
            self.inner._scenario_base_sample_demand = self.inner._sample_demand
        base_sample = self.inner._scenario_base_sample_demand

        def demand_override() -> dict:
            d = dict(base_sample())
            t = int(self.inner.t)
            start_t = int(self.scenario_cfg.get("urllc_burst_start_t", 0))
            end_t = int(self.scenario_cfg.get("urllc_burst_end_t", start_t))
            if start_t <= t <= end_t:
                d["urllc"] *= float(self.scenario_cfg.get("urllc_burst_scale", 1.0))
            return d

        self.inner._sample_demand = demand_override

    def reset(self, *, seed: int | None = None, options=None):
        self._install_demand_profile()
        if self.scenario_cfg.get("channel_profile", "default") == "linear_decay":
            # Keep reset's first observation aligned with the scenario channel.
            self.inner._last_channel = float(self._channel_profile_value(0))
        obs, info = super().reset(seed=seed, options=options)
        self._install_channel_profile()
        return obs, info
