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
