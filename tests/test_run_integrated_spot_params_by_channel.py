from __future__ import annotations

from drivers.run_integrated import _build_slice0_params_for_channel


def test_build_slice0_params_for_channel_applies_overrides() -> None:
    cfg = {
        "spot_zR": 344.5,
        "spot_lambda_nm": 667.0,
        "spot_q_min": 1.0,
        "spot_u0_min": 30.0,
        "spot_params_by_channel": {
            1: {
                "spot_radius_nm": 270.0,
                "spot_q_min": 6.182,
                "spot_u0_min": 0.0,
            }
        },
    }

    p1 = _build_slice0_params_for_channel(cfg, pixel_size_nm=65.0, channel=1)
    assert p1.spot_radius_nm == 270.0
    assert p1.q_min == 6.182
    assert p1.u0_min == 0.0

    p2 = _build_slice0_params_for_channel(cfg, pixel_size_nm=65.0, channel=2)
    assert p2.q_min == 1.0
    assert p2.u0_min == 30.0
    assert p2.spot_radius_nm is None
