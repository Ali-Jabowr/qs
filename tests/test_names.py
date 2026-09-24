"""The frame-name rule of docs/12-data-model.md §2: one format/parse pair."""

import pytest

from qs.names import FrameName, FrameNameError, format_name, parse_name

# The five examples of docs/12 §2, verbatim.
EXAMPLES = {
    "u007_good_G2_r0.png": FrameName("u007", "good", "G2", 0, None),
    "u007_dmg_G2_r0.png": FrameName("u007", "dmg", "G2", 0, None),
    "c02_good_G2_r1_pass3.png": FrameName("c02", "good", "G2", 1, 3),
    "p04_good_G1_r2.png": FrameName("p04", "good", "G1", 2, None),
    "x03_empty_G2_r0.png": FrameName("x03", "empty", "G2", 0, None),
}


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_the_docs_examples_parse_to_their_parts(name: str) -> None:
    assert parse_name(name) == EXAMPLES[name]


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_the_docs_examples_round_trip_exactly(name: str) -> None:
    assert format_name(parse_name(name)) == name


@pytest.mark.parametrize(
    ("name", "part"),
    [
        ("u7_good_G2_r0.png", "unit"),
        ("u0007_good_G2_r0.png", "unit"),
        ("c2_good_G2_r0_pass1.png", "unit"),
        ("u000_good_G2_r0.png", "unit"),
        ("q01_good_G2_r0.png", "unit"),
        ("u007_empty_G2_r0.png", "state"),
        ("x03_good_G2_r0.png", "state"),
        ("c01_dmg_G2_r0_pass1.png", "state"),
        ("u007_good_G2_r0_pass1.png", "pass"),
        ("x03_empty_G2_r0_pass1.png", "pass"),
        ("c01_good_G2_r0.png", "pass"),
        ("c01_good_G2_r0_pass4.png", "pass"),
        ("c01_good_G2_r0_pass0.png", "pass"),
        ("u007_good_G2_r3.png", "rotation"),
        ("u007_good_G4_r0.png", "geometry"),
        ("u007_good_g2_r0.png", "geometry"),
        ("u007_good_G2_r0.jpg", "suffix"),
        ("u007_good_G2_r0.PNG", "suffix"),
        ("u007_good_G2.png", "parts"),
        ("u007_good_G2_r0_pass1_x.png", "parts"),
    ],
)
def test_a_bad_name_is_refused_with_the_part_named(name: str, part: str) -> None:
    with pytest.raises(FrameNameError) as refused:
        parse_name(name)
    assert refused.value.part == part
    assert str(refused.value).startswith(f"{part}:")


def test_format_refuses_what_parse_refuses() -> None:
    with pytest.raises(FrameNameError) as refused:
        format_name(FrameName("u007", "good", "G2", 0, 2))
    assert refused.value.part == "pass"
    with pytest.raises(FrameNameError) as refused:
        format_name(FrameName("u007", "good", "G2", 3, None))
    assert refused.value.part == "rotation"
