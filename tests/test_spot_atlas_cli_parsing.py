from __future__ import annotations

from drivers.generate_spot_atlas_pptx import _parse_channel_float_map, _parse_channels_csv


def test_parse_channel_float_map_defaults_to_zero() -> None:
    default, per = _parse_channel_float_map([])
    assert default == 0.0
    assert per == {}


def test_parse_channel_float_map_bare_number_sets_default() -> None:
    default, per = _parse_channel_float_map(["30"])
    assert default == 30.0
    assert per == {}


def test_parse_channel_float_map_channel_overrides() -> None:
    default, per = _parse_channel_float_map(["1=0", "2:30"])
    assert default == 0.0
    assert per == {1: 0.0, 2: 30.0}


def test_parse_channels_csv() -> None:
    assert _parse_channels_csv("1,2") == (1, 2)
    assert _parse_channels_csv("  ") == ()
