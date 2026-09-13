import json
from pathlib import Path


def test_ladder_table_reports_gap_per_width(tmp_path: Path):
    from scripts.ladder_summary import ladder_table
    for d, plain, head in ((64, 2.0, 1.9), (128, 1.8, 1.6)):
        for s in (0, 1):
            (tmp_path / f"ladder_mingru_code_d{d}_s{s}.json").write_text(json.dumps({
                "run_name": f"ladder_mingru_code_d{d}_s{s}", "d_model": d, "seed": s, "task": "code",
                "arms": [{"arm": "mingru", "val_bpb": plain + 0.01 * s, "val_bpb_copyable": 1.0, "trainable_scalars": 1000 * d},
                         {"arm": "mingru_sparse_post", "val_bpb": head + 0.01 * s, "val_bpb_copyable": 0.5, "trainable_scalars": 1000 * d + 40000}]}))
    rows = ladder_table(tmp_path, backbone="mingru", corpus="code")
    assert [r["d_model"] for r in rows] == [64, 128]
    assert abs(rows[0]["gap"] - 0.1) < 1e-9 and abs(rows[1]["gap"] - 0.2) < 1e-9
    assert rows[0]["n_seeds"] == 2 and abs(rows[0]["plain"] - 2.005) < 1e-9
    assert abs(rows[0]["gap_copyable"] - 0.5) < 1e-9
