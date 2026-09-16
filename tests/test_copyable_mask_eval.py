from __future__ import annotations

import torch

from scripts.copyable_mask_eval import (
    ARMS,
    BIN_LABELS,
    SEEDS,
    build_parser,
    gap_mass_per_bin,
    longest_earlier_match,
    match_bin,
    mean_and_sample_sd,
)


def _encode(rows: list[str]) -> torch.Tensor:
    vocabulary = tuple(sorted(set("".join(rows))))
    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    return torch.tensor(
        [[char_to_id[character] for character in row] for row in rows], dtype=torch.long
    )


def test_longest_earlier_match_abcabc() -> None:
    result = longest_earlier_match(_encode(["abcabc"]))
    assert result.tolist() == [[0, 0, 0, 1, 2, 3]]


def test_longest_earlier_match_aaaa() -> None:
    result = longest_earlier_match(_encode(["aaaa"]))
    assert result.tolist() == [[0, 1, 2, 3]]


def test_longest_earlier_match_abab() -> None:
    result = longest_earlier_match(_encode(["abab"]))
    assert result.tolist() == [[0, 0, 1, 2]]


def test_longest_earlier_match_abcd() -> None:
    result = longest_earlier_match(_encode(["abcd"]))
    assert result.tolist() == [[0, 0, 0, 0]]


def test_longest_earlier_match_batch_matches_single_rows() -> None:
    batch = _encode(["aaaa", "abab"])
    result = longest_earlier_match(batch)
    assert result.tolist() == [[0, 1, 2, 3], [0, 0, 1, 2]]


def test_match_bin_boundaries() -> None:
    m = torch.tensor([0, 1, 2, 3, 4, 7, 8, 20])
    result = match_bin(m)
    assert result.tolist() == [0, 1, 2, 2, 3, 3, 4, 4]
    assert BIN_LABELS == ("0", "1", "2-3", "4-7", "8+")


def test_match_bin_preserves_shape_for_2d_input() -> None:
    m = torch.tensor([[0, 1, 8], [3, 4, 20]])
    result = match_bin(m)
    assert result.tolist() == [[0, 1, 4], [2, 3, 4]]


def test_mean_and_sample_sd_three_values() -> None:
    mean, sd = mean_and_sample_sd([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert abs(sd - 1.0) < 1e-9


def test_mean_and_sample_sd_single_value_has_zero_sd() -> None:
    mean, sd = mean_and_sample_sd([5.0])
    assert mean == 5.0
    assert sd == 0.0


def test_gap_mass_per_bin_sums_to_overall_gap() -> None:
    fractions = {"a": 0.25, "b": 0.75}
    gaps = {"a": 0.1, "b": 0.3}
    overall_gap = fractions["a"] * gaps["a"] + fractions["b"] * gaps["b"]
    mass = gap_mass_per_bin(fractions, gaps)
    assert set(mass) == set(fractions)
    assert abs(sum(mass.values()) - overall_gap) < 1e-12


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.arms == list(ARMS)
    assert args.seeds == list(SEEDS)
    assert args.device == "cuda:1"
    assert args.batch_size == 64
    assert args.limit_windows is None
