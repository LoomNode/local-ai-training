import pytest
import torch
from torch.nn import functional as F

from flybrain.model.hybrid import GRUHybrid


def _hybrid(**kw):
    torch.manual_seed(0)
    args = dict(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2,
                normalize=True, gate=True, chunk=8, mem_every=2, local_attn=8, key_convs=(2, 4))
    args.update(kw)
    return GRUHybrid(**args).eval()


def test_read_in_segments_scores_like_forward_and_merge_sums_agents():
    from flybrain.group import merge, read
    m = _hybrid()
    x = torch.randint(0, 256, (3, 50))
    ce, st = read(m, x, seg=16)
    with torch.no_grad():
        logits, _ = m(x)
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1), reduction="none").view(3, 49)
    assert ce.shape == (3, 49) and torch.allclose(ce, ref, atol=1e-4)
    c = merge(st)
    assert len(c.S) == len(st.heads) == 2 and c.S[0].shape == (64, 16) and c.Ksum[0].shape == (64,)
    assert torch.allclose(c.S[0], st.heads[0].S.sum(0)) and torch.allclose(c.Ksum[1], st.heads[1].Ksum.sum(0))
    singles = [read(m, x[i : i + 1], seg=16)[1] for i in range(3)]   # three agents one at a time == one batch of three
    c2 = merge(singles)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(c.S, c2.S))
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(c.Ksum, c2.Ksum))
    with pytest.raises(ValueError):
        merge([])


def test_preload_zero_is_alone_and_a_handoff_gives_the_lowest_head_what_reading_would():
    from flybrain.group import merge, preload, read
    m = _hybrid()
    a = torch.randint(0, 256, (1, 30)); b = torch.randint(0, 256, (1, 26))
    _, st_a = read(m, a)
    with torch.no_grad():
        alone, _ = m.stream(b, m.init_state(1))
        zero, _ = m.stream(b, preload(m.init_state(1), merge(st_a), 0.0))
        assert torch.equal(alone, zero)
        fresh = m.init_state(1)
        handed_logits, handed = m.stream(b, preload(fresh, merge(st_a), 1.0))
        _, sequential = m.stream(b, st_a.new_document())
        assert float(fresh.heads[0].S.abs().sum()) == 0.0                 # preload returned a copy
        assert torch.allclose(handed.heads[1].S, sequential.heads[1].S, atol=1e-5)
        assert not torch.allclose(alone, handed_logits, atol=1e-4)        # the handoff changes what the agent predicts


def test_preload_gives_every_agent_the_same_commons():
    from flybrain.group import merge, preload, read
    m = _hybrid()
    _, st = read(m, torch.randint(0, 256, (2, 24)))
    c = merge(st)
    two = preload(m.init_state(2), c, 0.5)
    assert torch.allclose(two.heads[0].S[0], 0.5 * c.S[0]) and torch.equal(two.heads[0].S[0], two.heads[0].S[1])


def test_commons_file_is_bound_to_the_weights_that_wrote_it(tmp_path):
    from flybrain.group import load_commons, merge, read, save_commons, weights_hash
    m = _hybrid()
    c = merge(read(m, torch.randint(0, 256, (2, 20)))[1])
    p = tmp_path / "commons.pt"
    save_commons(p, c, m)
    back = load_commons(p, m)
    assert all(torch.equal(a, b) for a, b in zip(c.S, back.S)) and all(torch.equal(a, b) for a, b in zip(c.Ksum, back.Ksum))
    assert weights_hash(m) == weights_hash(_hybrid())                     # same seed, same weights, same hash
    other = _hybrid()
    with torch.no_grad():
        other.decoder.bias.add_(1.0)
    assert weights_hash(other) != weights_hash(m)
    with pytest.raises(ValueError):
        load_commons(p, other)


def _one_pass_ce(m, x):
    """Cross-entropy of byte t + 1 given bytes up to t, [B, T - 1], from one forward pass over all of x."""
    with torch.no_grad():
        logits, _ = m(x)
    return F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1), reduction="none").view(x.size(0), -1)


def _tensors(st):
    """Every tensor of an AgentState in a fixed order (None where a field is empty)."""
    return list(st.h) + [t for hd in st.heads for t in (hd.S, hd.Ksum, hd.last_idx, hd.last_val, hd.conv_tail)] + [t for c in st.attn for t in (c.k, c.v)]


def _snapshot(st):
    """Clones of every tensor of an AgentState, and the attention offsets."""
    return [None if t is None else t.clone() for t in _tensors(st)], [c.offset for c in st.attn]


def _unchanged(st, snap):
    """Every tensor of `st` is bit-equal to its clone (same dtype, None where it was None), and the attention offsets are the same."""
    now = _tensors(st)
    return [c.offset for c in st.attn] == snap[1] and len(now) == len(snap[0]) and all(
        (p is None and q is None) or (p is not None and q is not None and p.dtype == q.dtype and torch.equal(p, q)) for p, q in zip(now, snap[0]))


def _tables(st):
    return [t for hd in st.heads for t in (hd.S, hd.Ksum)]


def _shares(p, q):
    return p.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()


def test_a_continued_read_scores_like_one_pass_except_for_its_own_first_byte():
    from flybrain.group import read
    m = _hybrid()
    x = torch.randint(0, 256, (3, 50))
    ref = _one_pass_ce(m, x)
    ce_a, st = read(m, x[:, :20], seg=16)
    ce_b, _ = read(m, x[:, 20:], state=st, seg=16)
    assert ce_a.shape == (3, 19) and ce_b.shape == (3, 29)              # the loss of byte 20 given bytes up to 19 is returned by neither read
    assert torch.allclose(ce_a, ref[:, :19], atol=1e-4) and torch.allclose(ce_b, ref[:, 20:], atol=1e-4)


def test_read_leaves_the_state_it_is_given_alone():
    from flybrain.group import read
    m = _hybrid()
    x = torch.randint(0, 256, (3, 50))
    _, st = read(m, x[:, :20], seg=16)                                  # a state that is not fresh
    kept = _snapshot(st)
    assert all(t is not None for t in kept[0])                          # nothing is empty, so every field is checked
    ce1, s1 = read(m, x[:, 20:], state=st, seg=16)
    ce2, s2 = read(m, x[:, 20:], state=st, seg=16)                      # the same state object, the same streams
    assert torch.equal(ce1, ce2)
    assert all(torch.equal(a, b) for a, b in zip(_tables(s1), _tables(s2)))
    assert _unchanged(st, kept)                                         # h, all five fields of every head, k and v of the cache and its offset
    assert s1 is not st and s2 is not st and s1 is not s2


def test_a_stream_of_one_byte_or_none_scores_nothing_and_leaves_a_usable_state():
    from flybrain.group import read
    m = _hybrid()
    x = torch.randint(0, 256, (2, 30))
    ref = _one_pass_ce(m, x)
    for T in (1, 0):
        ce, st = read(m, x[:, :T], seg=16)
        assert ce.shape == (2, 0) and ce.dtype == ref.dtype and st.batch == 2     # no byte has a next byte to score
        more, _ = read(m, x[:, T:], state=st, seg=16)                   # the state reads on from where it is
        assert torch.allclose(more, ref[:, T:], atol=1e-4), T
    _, empty = read(m, x[:, :0], seg=16)
    from_empty, _ = read(m, x, state=empty, seg=16)
    from_fresh, _ = read(m, x, seg=16)
    assert torch.equal(from_empty, from_fresh)                          # what an empty read hands back reads exactly like a fresh agent


def test_merge_and_preload_leave_their_inputs_alone_and_return_tables_of_their_own():
    from flybrain.group import merge, preload, read
    m = _hybrid()
    for B in (1, 2):                                                    # at B = 1 a sum over the agents could hand back a view of the table
        _, st = read(m, torch.randint(0, 256, (B, 24)), seg=16)
        _, st2 = read(m, torch.randint(0, 256, (B, 24)), seg=16)
        fresh = m.init_state(B)
        tables = lambda: [t for s in (st, st2, fresh) for t in _tables(s)]
        kept = [t.clone() for t in tables()]
        c, both = merge(st), merge([st, st2])                           # one state, and a sum that must not be accumulated into the first of them
        for com in (c, both):
            assert not any(_shares(a, b) for a in com.S + com.Ksum for s in (st, st2) for b in _tensors(s) if b is not None)
        c_kept = [t.clone() for t in c.S + c.Ksum]
        for alpha in (0.0, 0.5, 1.0):                                   # 0 and 1 are the values an implementation might special-case
            pre = preload(fresh, c, alpha)
            assert pre is not fresh
            assert not any(_shares(a, b) for a in _tables(pre) for b in _tensors(fresh) + c.S + c.Ksum if b is not None)
        assert all(torch.equal(a, b) for a, b in zip(tables(), kept))   # the tables of st, st2 and fresh are as they were
        assert all(torch.equal(a, b) for a, b in zip(c.S + c.Ksum, c_kept))


def test_preload_refuses_commons_that_do_not_fit_the_state():
    from flybrain.group import Commons, merge, preload, read
    two, one = _hybrid(), _hybrid(mem_every=0)                          # two heads, and the post head alone
    c2 = merge(read(two, torch.randint(0, 256, (1, 20)), seg=16)[1])
    c1 = merge(read(one, torch.randint(0, 256, (1, 20)), seg=16)[1])
    assert len(c2.S) == 2 and len(c1.S) == 1
    with pytest.raises(ValueError, match="heads"):
        preload(one.init_state(1), c2)                                  # more heads in the commons than in the state: zip would drop them
    with pytest.raises(ValueError, match="heads"):
        preload(two.init_state(1), c1)                                  # fewer: zip would leave the state's other heads empty
    with pytest.raises(ValueError, match="heads"):
        preload(two.init_state(1), Commons(S=c2.S, Ksum=c2.Ksum[:1]))   # the commons' own two lists disagree
    assert len(preload(two.init_state(1), c2).heads) == 2               # and commons that fit still go in


@pytest.mark.parametrize("T, seg", [(50, 1), (48, 16), (17, 16), (16, 16), (5, 100), (2, 1), (33, 32), (64, 32)])
def test_read_scores_like_one_pass_at_the_segment_boundaries(T, seg):
    from flybrain.group import read
    m = _hybrid()
    x = torch.randint(0, 256, (2, T))
    ce, _ = read(m, x, seg=seg)                                         # seg 1, T a multiple of seg, a one-byte tail, seg longer than the stream
    assert ce.shape == (2, T - 1) and torch.allclose(ce, _one_pass_ce(m, x), atol=1e-4)


def test_preload_adds_alpha_times_both_tables_to_every_head_and_agent():
    from flybrain.group import merge, preload, read
    m = _hybrid()
    c = merge(read(m, torch.randint(0, 256, (2, 24)), seg=16)[1])
    base = read(m, torch.randint(0, 256, (2, 18)), seg=16)[1]           # a state that has read: other tables per agent, nothing is compared against zeros
    for alpha in (0.5, 2.0):
        pre = preload(base, c, alpha)
        for i, (p, b) in enumerate(zip(pre.heads, base.heads)):
            assert torch.allclose(p.S - b.S, alpha * c.S[i][None].expand_as(p.S), atol=1e-5), (alpha, i)
            assert torch.allclose(p.Ksum - b.Ksum, alpha * c.Ksum[i][None].expand_as(p.Ksum), atol=1e-5), (alpha, i)


def test_one_reader_handed_to_a_fresh_agent_predicts_like_reading_in_sequence():
    from flybrain.group import merge, preload, read
    m = _hybrid()
    a = torch.randint(0, 256, (1, 30)); b = torch.randint(0, 256, (1, 26))
    _, st_a = read(m, a)
    handed, _ = read(m, b, state=preload(m.init_state(1), merge(st_a), 1.0))
    sequential, _ = read(m, b, state=st_a.new_document())               # one reader's commons at alpha 1 is what the same agent keeps across a new document
    assert torch.allclose(handed, sequential, atol=1e-5)


def test_preload_keeps_the_reading_position_of_a_state_that_has_read():
    from flybrain.group import merge, preload, read
    m = _hybrid()
    c = merge(read(m, torch.randint(0, 256, (1, 24)), seg=16)[1])
    _, warm = read(m, torch.randint(0, 256, (1, 20)), seg=16)
    x = torch.randint(0, 256, (1, 15))
    plain, _ = read(m, x, state=warm, seg=16)
    zero, _ = read(m, x, state=preload(warm, c, 0.0), seg=16)
    one, _ = read(m, x, state=preload(warm, c, 1.0), seg=16)
    assert torch.equal(plain, zero)                                     # nothing preloaded: it reads on as before (carries, last keys, conv tails, cache)
    assert not torch.allclose(plain, one, atol=1e-4)                    # and a real alpha does change what it predicts


def test_preload_casts_the_commons_to_the_dtype_of_the_state():
    from flybrain.group import Commons, merge, preload, read
    m = _hybrid()
    c = merge(read(m, torch.randint(0, 256, (1, 20)), seg=16)[1])
    c64 = Commons(S=[t.double() for t in c.S], Ksum=[t.double() for t in c.Ksum])
    pre = preload(m.init_state(1), c64, 1.0)
    assert all(h.S.dtype == torch.float32 and h.Ksum.dtype == torch.float32 for h in pre.heads)


def test_read_builds_no_graph_although_the_model_has_gradients_on():
    from flybrain.group import read
    m = _hybrid()
    assert all(p.requires_grad for p in m.parameters())                 # eval() does not switch them off: only read's no_grad keeps the graph away
    ce, st = read(m, torch.randint(0, 256, (2, 20)), seg=16)
    assert not ce.requires_grad and not any(t.requires_grad for t in _tensors(st) if t is not None)


def test_weights_hash_covers_buffers_and_places_and_ignores_registration_order():
    import collections
    import copy
    from flybrain.group import weights_hash
    m = _hybrid(); h = weights_hash(m)
    other = _hybrid()
    with torch.no_grad():
        other.mem.code.proj[0, 0] += 1.0                                # a buffer (the frozen address table), not a parameter
    assert weights_hash(other) != h
    swapped = _hybrid()
    with torch.no_grad():
        swapped.embed.weight[[0, 1]] = swapped.embed.weight[[1, 0]]     # the same values and the same sum, in other places
    assert weights_hash(swapped) != h
    shuffled = copy.deepcopy(m)                                         # the same weights, registered in the opposite order
    for mod in shuffled.modules():
        for attr in ("_modules", "_parameters", "_buffers"):
            setattr(mod, attr, collections.OrderedDict(reversed(list(getattr(mod, attr).items()))))
    assert list(shuffled.state_dict()) != list(m.state_dict()) and weights_hash(shuffled) == h


def test_commons_file_keeps_every_head(tmp_path):
    from flybrain.group import load_commons, merge, read, save_commons
    m = _hybrid()
    c = merge(read(m, torch.randint(0, 256, (2, 20)))[1])
    save_commons(tmp_path / "c.pt", c, m)
    back = load_commons(tmp_path / "c.pt", m)
    assert len(back.S) == len(c.S) == 2 and len(back.Ksum) == len(c.Ksum) == 2     # the brief's round trip zips, so a dropped head would pass it


@pytest.mark.parametrize("heads", [(1, 2), (2, 1), (2, 2, 1)])
def test_merge_refuses_states_with_different_numbers_of_heads(heads):
    from flybrain.group import merge, read
    models = {2: _hybrid(), 1: _hybrid(mem_every=0)}                    # two heads, and the post head alone
    states = [read(models[n], torch.randint(0, 256, (1, 20)), seg=16)[1] for n in heads]
    with pytest.raises(ValueError, match="heads"):                      # the count used to come from the first state: extra heads of a later one were dropped, fewer raised IndexError
        merge(states)


@pytest.mark.parametrize("seg", [0, -1, -16])
def test_read_refuses_a_segment_length_below_one(seg):
    from flybrain.group import read
    with pytest.raises(ValueError, match="seg"):                        # 0 used to fail with range()'s message, below 0 returned an empty loss and the unread state
        read(_hybrid(), torch.randint(0, 256, (2, 20)), seg=seg)
