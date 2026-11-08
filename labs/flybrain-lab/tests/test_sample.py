import torch

from flybrain.model.hybrid import GRUHybrid


def _tiny():
    torch.manual_seed(0)
    return GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4,
                     key_conv=2, normalize=True, gate=True, chunk=8, mem_every=2, local_attn=8, key_convs=(2, 4)).eval()


def test_sample_next_is_greedy_at_zero_temperature_and_respects_top_k():
    from scripts.sample import sample_next
    logits = torch.tensor([0.1, 2.0, 0.3, 1.9] + [-5.0] * 252)
    gen = torch.Generator().manual_seed(0)
    assert sample_next(logits, 0.0, 0, gen) == 1
    assert all(sample_next(logits, 5.0, 1, gen) == 1 for _ in range(20))       # top-1 is the argmax at any temperature
    assert {sample_next(logits, 5.0, 2, gen) for _ in range(200)} == {1, 3}     # top-2 never leaves the two best

    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        return [sample_next(logits, 1.0, 0, g) for _ in range(8)]

    assert draw(7) == draw(7)


def test_generate_recompute_matches_a_manual_greedy_loop_and_is_deterministic():
    from scripts.sample import generate
    m = _tiny()
    prompt = b"int main(void) {"
    out = generate(m, prompt, 12, temperature=0.0)
    buf = list(prompt)
    for _ in range(12):
        with torch.no_grad():
            logits, _ = m(torch.tensor([buf]))
        buf.append(int(logits[0, -1].argmax()))
    assert out == bytes(buf[len(prompt):]) and len(out) == 12
    assert generate(m, prompt, 12, temperature=0.9, seed=3) == generate(m, prompt, 12, temperature=0.9, seed=3)


def test_generate_uses_only_the_last_window_bytes_and_starts_an_empty_prompt_from_a_newline():
    from scripts.sample import generate
    m = _tiny()
    prompt = b"abcdefghij"
    out = generate(m, prompt, 6, temperature=0.0, window=4)
    buf = list(prompt)
    for _ in range(6):
        with torch.no_grad():
            logits, _ = m(torch.tensor([buf[-4:]]))
        buf.append(int(logits[0, -1].argmax()))
    assert out == bytes(buf[len(prompt):])
    seen = []
    assert generate(m, b"", 5, temperature=0.0, on_byte=seen.append) == generate(m, b"\n", 5, temperature=0.0)
    assert len(seen) == 5


def test_utf8_printer_prints_split_characters_once_and_replaces_invalid_bytes():
    from scripts.sample import Utf8Printer
    got = []
    p = Utf8Printer(got.append)
    for b in "é".encode("utf-8"):      # two bytes, one character
        p.feed(b)
    p.feed(0xFF)                        # never valid in UTF-8
    p.feed(ord("x"))
    p.close()
    assert "".join(got) == "é�x"
    got.clear()
    p = Utf8Printer(got.append)
    p.feed(0xC3)                        # a character cut off at the end
    p.close()
    assert "".join(got) == "�"


def test_streamed_and_recompute_samplers_agree_greedily():
    from scripts.sample import generate, streamable
    m = _tiny()
    prompt = b"static int probe(struct device *dev)\n{\n"
    assert streamable(m)
    assert generate(m, prompt, 40, temperature=0.0, stream=True) == generate(m, prompt, 40, temperature=0.0, stream=False)
    assert generate(m, prompt, 20, temperature=0.9, seed=5, stream=True) == generate(m, prompt, 20, temperature=0.9, seed=5, stream=True)
    seen = []
    assert len(generate(m, b"", 3, temperature=0.0, stream=True, on_byte=seen.append)) == 3 and len(seen) == 3
    assert not streamable(GRUHybrid(d_model=16, backbone="transformer", where="none", n_layers=2, d_ff=32))


def test_generating_zero_bytes_returns_nothing_on_either_path_never_calls_on_byte_and_reads_no_prompt(monkeypatch):
    import pytest
    from scripts.sample import generate
    m = _tiny()
    monkeypatch.setattr(m, "stream", lambda *a, **kw: pytest.fail("the prompt was read to draw no bytes"))
    seen = []
    for stream in (True, False):
        assert generate(m, b"int ", 0, stream=stream, on_byte=seen.append) == b""
        assert generate(m, b"", 0, stream=stream, on_byte=seen.append) == b""
    assert seen == []


def test_streamed_and_recompute_samplers_agree_past_the_attention_window_and_across_head_chunks():
    from scripts.sample import generate
    m = _tiny()                                   # local attention window 8 (local_attn=8), head chunk 8 (chunk=8)
    prompt = b"#include <linux/io.h>\n"           # 22 bytes: well past the window, and it ends part-way through a head chunk
    out = generate(m, prompt, 30, temperature=0.0, stream=True)       # positions 22..51: the head chunk ends at 23, 31, 39 and 47 fall inside
    assert len(out) == 30 and out == generate(m, prompt, 30, temperature=0.0, stream=False)
    assert generate(m, prompt, 30, temperature=0.0, window=4, stream=True) == out     # the streamed path ignores window: the state is the context


def test_streamed_and_recompute_samplers_agree_under_sampling():
    from scripts.sample import generate
    m = _tiny()
    prompt = b"static int probe(struct device *dev)\n{\n"
    for t, k, seed in ((0.9, 5, 5), (1.2, 40, 3), (1.0, 1, 2)):          # top_k 1 forces the best byte at any temperature; the others draw among several
        assert generate(m, prompt, 25, t, k, seed, stream=True) == generate(m, prompt, 25, t, k, seed, stream=False), (t, k, seed)


def test_each_path_is_deterministic_for_a_seed_and_on_byte_gets_the_returned_bytes_in_order():
    from scripts.sample import generate
    m = _tiny()
    prompt = b"int main(void) {"
    for stream in (True, False):
        out = generate(m, prompt, 25, temperature=0.9, seed=5, stream=stream)
        assert len(out) == 25 and out == generate(m, prompt, 25, temperature=0.9, seed=5, stream=stream)
        assert out != generate(m, prompt, 25, temperature=0.9, seed=6, stream=stream)         # the seed reaches the sampler on this path
        seen = []
        assert generate(m, prompt, 25, temperature=0.9, seed=5, stream=stream, on_byte=seen.append) == out
        assert seen == list(out)                                                              # each byte once, in the order drawn


def _spy_on_stream(monkeypatch, m):
    """Wrap m.stream. The returned list gets one (tokens in the call, autograd on during it, state passed in, state returned) per call."""
    calls, real = [], m.stream

    def spy(tokens, state):
        grad = torch.is_grad_enabled()
        logits, new = real(tokens, state)
        calls.append((tokens.size(1), grad, state, new))
        return logits, new

    monkeypatch.setattr(m, "stream", spy)
    return calls


def _state_tensors(st):
    """Every tensor an AgentState holds."""
    ts = [h for h in st.h if h is not None]
    for hd in st.heads:
        ts += [t for t in (hd.S, hd.Ksum, hd.last_idx, hd.last_val, hd.conv_tail) if t is not None]
    for a in st.attn:
        ts += [t for t in (a.k, a.v) if t is not None]
    return ts


def test_the_streamed_prompt_is_read_in_segments_that_carry_the_state(monkeypatch):
    import scripts.sample as sample
    m = _tiny()
    prompt = b"static int probe(struct device *dev)\n{\n\treturn 0;\n}\n\n"          # 53 bytes
    one_segment = sample.generate(m, prompt, 20, temperature=0.0, stream=True)       # PROMPT_SEGMENT as it is: the prompt in one call
    recompute = sample.generate(m, prompt, 20, temperature=0.0, stream=False)
    calls = _spy_on_stream(monkeypatch, m)
    for seg in (16, 13):                       # 16 is two head chunks (chunk 8); 13 ends part-way through a chunk and through the attention window (8)
        monkeypatch.setattr(sample, "PROMPT_SEGMENT", seg)
        calls.clear()
        out = sample.generate(m, prompt, 20, temperature=0.0, stream=True)
        sizes = [min(seg, len(prompt) - c0) for c0 in range(0, len(prompt), seg)] + [1] * 19       # no call over seg tokens, then one byte per step
        assert [c[0] for c in calls] == sizes
        assert all(calls[k][2] is calls[k - 1][3] for k in range(1, len(calls)))     # each call continues the state the one before returned
        assert out == one_segment == recompute, seg


def test_the_streamed_path_runs_without_autograd_and_carries_a_state_with_no_graph(monkeypatch):
    from scripts.sample import generate
    m = _tiny()
    calls = _spy_on_stream(monkeypatch, m)
    generate(m, b"int main(void) {", 6, temperature=0.0, stream=True)
    assert len(calls) == 6                                  # the prompt, then five single-byte steps
    for _, grad, _, new in calls:
        assert grad is False                                # a graph on the carried state would grow with every byte
        tensors = _state_tensors(new)
        assert tensors and not any(t.requires_grad for t in tensors)


def test_streamable_answers_as_init_state_does_without_building_a_state(monkeypatch):
    import pytest
    from scripts.sample import streamable
    base = dict(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, chunk=8, mem_every=2)
    cells = [{}, dict(chunk=0), dict(where="pre", mem_every=0), dict(count_head=True), dict(shuffle_keys=True),
             dict(backbone="transformer", where="none", mem_every=0), dict(backbone="gru", mem_every=0)]
    answers = []
    for cell in cells:
        m = GRUHybrid(**{**base, **cell}).eval()
        try:
            m.init_state(1); expected = True
        except ValueError:
            expected = False
        monkeypatch.setattr(m, "init_state", lambda *a, **kw: pytest.fail("streamable built an agent state"))
        answers.append(streamable(m))
        assert answers[-1] is expected, cell
    assert answers == [True, False, False, False, False, False, False]                       # the first cell is the one that streams
    assert streamable(torch.nn.Linear(2, 2)) is False                                         # a model with no init_state at all


def _main_calls_generate_with(monkeypatch, ckpt, *flags):
    """The (positional, keyword) arguments main hands to generate for these flags; generate itself is replaced, so nothing is sampled.
    Also checks that on_byte is the feed of the one printer main made."""
    import scripts.sample as sample
    calls, printers = [], []

    class Recorded(sample.Utf8Printer):
        def __init__(self, write):
            super().__init__(write)
            printers.append(self)

    with monkeypatch.context() as mp:
        mp.setattr(sample, "Utf8Printer", Recorded)
        mp.setattr(sample, "generate", lambda *a, **kw: calls.append((a, kw)))
        mp.setattr("sys.argv", ["sample.py", "--ckpt", str(ckpt), "--prompt", "int ", "--n", "3", *flags])
        sample.main()
    (call,), (printer,) = calls, printers
    assert call[1]["on_byte"] == printer.feed
    return call


def test_the_command_line_streams_by_default_and_recomputes_under_no_stream_or_for_a_model_that_cannot_stream(monkeypatch, tiny_checkpoint):
    mingru, transformer = tiny_checkpoint(), tiny_checkpoint("transformer")
    args, kw = _main_calls_generate_with(monkeypatch, mingru, "--temperature", "0.5", "--top-k", "7", "--seed", "9", "--window", "64")
    assert kw["stream"] is True
    assert args[1:] == (b"int ", 3, 0.5, 7, 9, 64)                      # the rest of the command line reaches generate as it did before
    assert _main_calls_generate_with(monkeypatch, mingru, "--no-stream")[1]["stream"] is False
    assert _main_calls_generate_with(monkeypatch, transformer)[1]["stream"] is False                # it cannot stream: no failure, the recompute path
    assert _main_calls_generate_with(monkeypatch, transformer, "--no-stream")[1]["stream"] is False


def test_a_window_below_one_is_refused_before_the_checkpoint_loads(monkeypatch, tiny_checkpoint, capsys):
    """--window 0 or below would slice buf[-0:] (the whole buffer) or drop the start of the context on the recompute path, silently."""
    import pytest
    import scripts.sample as sample
    ckpt = tiny_checkpoint()
    for w in ("0", "-5"):
        with monkeypatch.context() as mp:
            mp.setattr(sample, "load_model", lambda *a, **kw: pytest.fail("the checkpoint was loaded"))
            mp.setattr(sample, "generate", lambda *a, **kw: pytest.fail("a byte was sampled"))
            mp.setattr("sys.argv", ["sample.py", "--ckpt", str(ckpt), "--prompt", "int ", "--window", w])
            with pytest.raises(SystemExit) as refused:
                sample.main()
        assert refused.value.code == 2 and "--window must be at least 1" in capsys.readouterr().err, w
    assert _main_calls_generate_with(monkeypatch, ckpt, "--window", "1")[0][-1] == 1             # the smallest window is still allowed
