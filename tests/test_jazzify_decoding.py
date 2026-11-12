from fractions import Fraction

from homr.jazzify.decoding import bar_lengths, bar_penalty, reference_bar_length


def test_bar_penalty_is_proportional_to_the_difference() -> None:
    assert bar_penalty(Fraction(9, 2), Fraction(4), 3.0) == 1.5
    assert bar_penalty(Fraction(3), Fraction(4), 3.0) == 3.0
    assert bar_penalty(Fraction(4), None, 3.0) == 0.0


def test_first_bar_may_be_a_short_pickup_but_not_too_long() -> None:
    assert bar_penalty(Fraction(1), Fraction(4), 3.0, may_be_pickup=True) == 0.0
    # A misread triplet makes the bar longer; that is still penalised on the first bar.
    assert bar_penalty(Fraction(9, 2), Fraction(4), 3.0, may_be_pickup=True) == 1.5


def test_reference_needs_a_clear_majority_of_plain_bars() -> None:
    four = [(Fraction(4), False)] * 3
    assert reference_bar_length([*four, (Fraction(3), False)]) == 4
    assert reference_bar_length([(Fraction(4), False)] * 2) is None
    assert reference_bar_length([(Fraction(4), True)] * 5) is None  # tuplet bars do not vote


def test_bar_lengths_reads_closed_bars() -> None:
    tokens = [("note_4", "upper")] * 4 + [("barline", ".")] + [("note_2", "upper")] * 2
    assert [length for length, _ in bar_lengths(tokens)] == [4, 4]
