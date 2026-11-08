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


def test_cross_eval_rebuild_loads_weights_for_every_backbone(tmp_path):
    import torch
    from flybrain.model.hybrid import GRUHybrid
    from scripts.cross_eval import rebuild
    for arm, kw in (("mingru", dict(where="none")), ("transformer", dict(where="none")), ("mingru_sparse_post", dict(where="post", key="sparse"))):
        bb = arm.split("_")[0]
        torch.manual_seed(1)
        m = GRUHybrid(d_model=16, hidden=16, backbone=bb, n_layers=1, d_ff=32, n_key=64, top_k=4, max_len=1024, **kw)
        args = dict(d_model=16, gru_hidden=16, n_layers=1, d_ff=32, learned_key_dim=16, key_conv=2, seed=1, seq_len=64,
                    n_key=64, top_k=4, claws=4, theta=1.0, normalize=True, decay=1.0, gate=False)
        torch.manual_seed(2)
        r = rebuild({"arm": arm, "args": args, "state": m.state_dict()})
        x = torch.randint(0, 256, (1, 64))
        m.eval(); r.eval()
        assert torch.allclose(m(x)[0], r(x)[0], atol=1e-5), arm


def test_rebuild_carries_mem_every():
    import torch
    from flybrain.model.hybrid import GRUHybrid
    from scripts.cross_eval import rebuild
    torch.manual_seed(1)
    m = GRUHybrid(d_model=16, backbone="mingru", where="post", key="sparse", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2, normalize=True, gate=True, mem_every=2, chunk=8)
    args = dict(d_model=16, gru_hidden=16, n_layers=4, d_ff=32, learned_key_dim=16, key_conv=2, seed=1, seq_len=64,
                n_key=64, top_k=4, claws=4, theta=1.0, normalize=True, decay=1.0, gate=True, mem_every=2, chunk=8)
    r = rebuild({"arm": "mingru_sparse_post", "args": args, "state": m.state_dict()})
    assert r.n_heads == 2 and all(h.chunk == 8 for h in r.all_heads())
    x = torch.randint(0, 256, (1, 64)); m.eval(); r.eval()
    assert torch.allclose(m(x)[0], r(x)[0], atol=1e-5)
