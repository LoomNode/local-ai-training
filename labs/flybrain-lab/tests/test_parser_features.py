"""checks/parser_features.py: what the parser pseudo-orders say at each position of a small C file."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tree_sitter_language_pack")
CHECKS = Path(__file__).resolve().parents[1] / "notes" / "speed-round" / "checks"
sys.path.insert(0, str(CHECKS))
spec = importlib.util.spec_from_file_location("parser_features", CHECKS / "parser_features.py")
pf = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)

SRC = b"""struct ring { int head; int tail; };

static int ring_push(struct ring *r, int value)
{
\tint next = r->head + 1;
\tr->head = next;
\treturn value;
}
"""


def test_the_completion_features_fire_inside_names_with_the_next_byte_of_an_earlier_name():
    kinds = {"c": {"identifier": 1, "field_identifier": 2, "compound_statement": 3}}
    pred, conf = pf.file_features(SRC, "c", kinds)
    assert pred.shape == (len(SRC), 4) and conf.shape == pred.shape
    i = SRC.index(b"r->head = next") + 3            # 'h' typed, "head" was defined in the struct
    assert conf[i + 1, 2] == 2 and pred[i + 1, 2] == ord("e")     # after "h": e
    assert conf[i + 3, 2] == 4 and pred[i + 3, 2] == ord("d")     # after "hea": d
    assert conf[i, 2] == 0                                          # before the name starts: nothing
    j = SRC.index(b"return value") + 7              # "value" is a parameter: defined; and used before
    assert pred[j + 1, 2] == ord("a") and pred[j + 1, 3] == ord("a")
    assert pred[j + 4, 2] == ord("e") and conf[j + 4, 2] == 5
    # the node kind and depth come from the node holding the PREVIOUS byte; position 0 has none
    assert conf[0, 0] == 0 and conf[1, 0] == 1 and (pred[:, 1][conf[:, 1] > 0] >= 128).all()
    assert pred[i + 2, 0] == 2                      # inside "head": a field_identifier holds the previous byte


def test_no_parser_means_no_parser_features_but_the_name_completions_still_work():
    pred, conf = pf.file_features(SRC, None, {})
    assert (conf[:, :2] == 0).all() and (conf[:, 2] == 0).all()     # no tree: no kinds, no definitions
    j = SRC.index(b"return value") + 7
    assert pred[j + 1, 3] == ord("a") and conf[j + 1, 3] == 2       # used names need no parser
