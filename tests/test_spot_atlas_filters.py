from __future__ import annotations

import pandas as pd

from src.qc_spot_atlas import SpotAtlasParams, _filter_spots_by_u0, _filter_spots_to_required_nuclei


def test_filter_spots_by_u0_per_channel() -> None:
    df = pd.DataFrame(
        [
            {
                "y_px": 0.0,
                "x_px": 0.0,
                "intensity": 1.0,
                "background": 0.0,
                "spot_channel": 1,
                "output_dir": "runA",
            },
            {
                "y_px": 1.0,
                "x_px": 1.0,
                "intensity": 5.0,
                "background": 0.0,
                "spot_channel": 2,
                "output_dir": "runA",
            },
            {
                "y_px": 2.0,
                "x_px": 2.0,
                "intensity": 20.0,
                "background": 0.0,
                "spot_channel": 2,
                "output_dir": "runA",
            },
        ]
    )

    params = SpotAtlasParams(u0_min=0.0, u0_min_by_channel={2: 10.0})
    out = _filter_spots_by_u0(df, params)

    assert set(out["spot_channel"].astype(int).tolist()) == {1, 2}
    # Channel 2 should keep only the u0 >= 10 row.
    assert out[out["spot_channel"].astype(int) == 2]["intensity"].astype(float).tolist() == [20.0]


def test_filter_spots_to_required_nuclei() -> None:
    df = pd.DataFrame(
        [
            # Nucleus 1 has both channels
            {
                "y_px": 0.0,
                "x_px": 0.0,
                "intensity": 10.0,
                "background": 0.0,
                "spot_channel": 1,
                "output_dir": "runA",
                "input_path": "fileA.ims",
                "nucleus_label": 1,
            },
            {
                "y_px": 0.0,
                "x_px": 1.0,
                "intensity": 10.0,
                "background": 0.0,
                "spot_channel": 2,
                "output_dir": "runA",
                "input_path": "fileA.ims",
                "nucleus_label": 1,
            },
            # Nucleus 2 has only channel 1
            {
                "y_px": 1.0,
                "x_px": 0.0,
                "intensity": 10.0,
                "background": 0.0,
                "spot_channel": 1,
                "output_dir": "runA",
                "input_path": "fileA.ims",
                "nucleus_label": 2,
            },
            # Unassigned (label 0)
            {
                "y_px": 2.0,
                "x_px": 2.0,
                "intensity": 10.0,
                "background": 0.0,
                "spot_channel": 2,
                "output_dir": "runA",
                "input_path": "fileA.ims",
                "nucleus_label": 0,
            },
        ]
    )

    out = _filter_spots_to_required_nuclei(df, required_spot_channels=(1, 2))
    assert not out.empty
    assert set(out["nucleus_label"].astype(int).tolist()) == {1}
    assert set(out["spot_channel"].astype(int).tolist()) == {1, 2}
