import torch

from local_ai_training.data import batch_from_starts, build_char_corpus, make_batch_schedule


def test_build_char_corpus_uses_deterministic_tail_validation_split() -> None:
    corpus = build_char_corpus("abcde" * 20, validation_fraction=0.1)

    assert corpus.train_text == "abcde" * 18
    assert corpus.validation_text == "abcde" * 2
    assert corpus.vocabulary == tuple("abcde")
    assert corpus.decode(corpus.train_ids[:5]) == "abcde"


def test_batch_schedule_is_seeded_and_stays_inside_next_token_range() -> None:
    first = make_batch_schedule(
        data_length=100, steps=4, batch_size=3, block_size=8, seed=17
    )
    second = make_batch_schedule(
        data_length=100, steps=4, batch_size=3, block_size=8, seed=17
    )

    assert torch.equal(first, second)
    assert first.shape == (4, 3)
    assert first.min().item() >= 0
    assert first.max().item() <= 100 - 8 - 1


def test_batch_from_starts_returns_shifted_character_targets() -> None:
    data = torch.arange(20)

    inputs, targets = batch_from_starts(data, torch.tensor([0, 5]), block_size=4)

    assert inputs.tolist() == [[0, 1, 2, 3], [5, 6, 7, 8]]
    assert targets.tolist() == [[1, 2, 3, 4], [6, 7, 8, 9]]

