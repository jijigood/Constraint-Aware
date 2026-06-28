"""Shared V4 baseline and scenario definitions."""

BASELINE_CONFIGS = {
    "M1_vanilla": dict(use_shield=False, static_p_min=None, use_state_aug=False, proj_penalty=0.0, z_mode="oracle", verifier_on=True),
    "M2_static": dict(use_shield=True, static_p_min=50, use_state_aug=False, proj_penalty=0.0, z_mode="oracle", verifier_on=True),
    "M3_dynamic_no_aug": dict(use_shield=True, static_p_min=None, use_state_aug=False, proj_penalty=0.0, z_mode="oracle", verifier_on=True),
    "M4_penalty_no_aug": dict(use_shield=True, static_p_min=None, use_state_aug=False, proj_penalty=0.01, z_mode="oracle", verifier_on=True),
    "M5_constraint_aware": dict(use_shield=True, static_p_min=None, use_state_aug=True, proj_penalty=0.0, z_mode="oracle", verifier_on=True),
    "M6_full_cer": dict(use_shield=True, static_p_min=None, use_state_aug=True, proj_penalty=0.0, z_mode="cer", verifier_on=True),
    "M6_noisy_vrf_on": dict(use_shield=True, static_p_min=None, use_state_aug=True, proj_penalty=0.0, z_mode="noisy", verifier_on=True),
    "M6_noisy_vrf_off": dict(use_shield=True, static_p_min=None, use_state_aug=True, proj_penalty=0.0, z_mode="noisy", verifier_on=False),
}

SCENARIOS = {
    "S1_stable": {"legacy_regime": "balanced", "sla_schedule": {0: 0.99}, "channel_profile": "default"},
    "S2_urllc_burst": {"legacy_regime": "bursty", "sla_schedule": {0: 0.99}, "channel_profile": "default"},
    "S3_channel_decay": {
        "legacy_regime": "high_urllc",
        "sla_schedule": {0: 0.99},
        "channel_profile": "linear_decay",
        "channel_start": 0.8,
        "channel_end": 0.15,
        "channel_start_t": 100,
        "channel_end_t": 400,
        "episode_len": 500,
    },
    "S4_sla_upgrade": {"legacy_regime": "high_urllc", "sla_schedule": {0: 0.99, 100: 0.9999}, "channel_profile": "default"},
    "S5_combined": {
        "legacy_regime": "bursty",
        "sla_schedule": {0: 0.99, 150: 0.9999},
        "channel_profile": "linear_decay",
        "channel_start": 0.7,
        "channel_end": 0.12,
        "channel_start_t": 120,
        "channel_end_t": 420,
        "episode_len": 500,
    },
    "high_embb": {"legacy_regime": "high_embb", "sla_schedule": {0: 0.99}, "channel_profile": "default"},
    "high_urllc": {"legacy_regime": "high_urllc", "sla_schedule": {0: 0.99}, "channel_profile": "default"},
}
