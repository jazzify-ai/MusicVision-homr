"""Jazzify decoding changes for the HOMR transformer (B, C, E).

Ported unchanged from MusicVision-ml-private ``musicvision_ml/homr_decoding.py``, where they were
developed and measured as process-local hooks on upstream HOMR. Production enables them through
``homr.jazzify.jazzify_processing``; the hooks are installed for one ``process_image`` call and
removed afterwards. The research write-up is MusicVision-ml-private
``docs/homr_decoding_experiments.md`` (it holds the exact bar-penalty scoring rule).

B. Clef/key carry-forward
   Lead sheets often print the clef and key signature on the first system only. The transformer
   reads every staff on its own, so a staff without a printed clef gets a guessed clef/key and
   wrong pitches. When the segmentation finds no clef/key anchor at a staff's left edge, the clef
   and key read on the last staff that showed them are *forced* as the decoder's first tokens.

C. Consistent heads and bar-length-aware beam search
   The decoder has one output head per field (rhythm, pitch, lift, ...). Upstream takes each
   head's argmax independently, so "note" + "no pitch" can be emitted and later becomes a rest.
   Here a candidate symbol is always built consistently (notes get a real pitch, rests get none)
   and scored jointly. Optionally a beam search keeps several readings and applies a soft penalty
   whenever a bar does not add up to the page's reference bar length. See
   MusicVision-ml-private ``docs/homr_decoding_experiments.md`` for the scoring rule.

E. Cleanup
   - A slur stop with no open slur whose previous note has the same pitch becomes a tie.
   - Low-confidence ornaments/fermatas are replaced by the best plain articulation.
   - Repeated time signatures that do not change the bar length are dropped; denominator-only
     ``/2`` readings of four-quarter bars are written as 4/4 (see the doc for the trade-off).
   - Eighth notes and shorter are beamed by beat groups in the written MusicXML.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

# Tokens which close a bar in the decoder's rhythm stream.
BAR_CLOSERS = frozenset(
    {"barline", "doublebarline", "bolddoublebarline", "repeatEnd", "repeatEndStart"}
)
# Ornament-like articulation components which are hallucinated on staccato dots in jazz fonts.
GATED_ARTICULATIONS = frozenset(
    {
        "fermata",
        "trill",
        "staccatissimo",
        "mordent",
        "turn",
        "breathMark",
        "caesura",
        "doit",
        "arpeggiate",
        "tremolo",
    }
)
NONOTE = "."
EMPTY = "_"


@dataclass
class DecodeOptions:
    carry_clef_key: bool = False
    # Clef forced on an unmarked staff when nothing has been carried yet ("" = none).
    default_clef: str = "clef_G2"
    # Keep the printed clef/key context from one page to the next page of the same document.
    carry_across_pages: bool = False
    # Printed clefs believed on single-line lead sheets; other readings are re-decoded.
    trusted_clefs: tuple[str, ...] = ("clef_G2", "clef_F4")
    consistent_heads: bool = False
    beam_width: int = 1
    bar_penalty: float = 0.0
    # Cost of keeping a time signature after the meter is established (beam search only).
    meter_change_penalty: float = 1.0
    ornament_min_prob: float | None = None
    cleanup: bool = False
    # Run the re-implemented decoder in pure upstream mode, to prove token-for-token parity.
    parity_check: bool = False

    @property
    def changes_decoding(self) -> bool:
        return (
            self.parity_check
            or self.carry_clef_key
            or self.consistent_heads
            or self.beam_width > 1
            or self.ornament_min_prob is not None
        )

    @property
    def enabled(self) -> bool:
        return self.changes_decoding or self.cleanup


# --------------------------------------------------------------------------------------------
# Bar accounting (pure, unit tested)
# --------------------------------------------------------------------------------------------


def symbol_quarters(rhythm: str) -> Fraction | None:
    """Duration of a note/rest token in quarter notes; ``None`` when not measurable."""
    if not rhythm.startswith(("note_", "rest_")):
        return Fraction(0)
    kern = rhythm.split("_", 1)[1]
    if kern.endswith("m") or "G" in kern:
        # Multi-measure rests have no fixed length; grace notes take no time.
        return None if kern.endswith("m") else Fraction(0)
    digits = "".join(ch for ch in kern if ch.isdigit())
    dots = kern.count(".")
    if not digits:
        return None
    base = int(digits)
    whole = Fraction(2) if base == 0 else Fraction(1, base)  # kern 0 = breve
    total = whole
    add = whole / 2
    for _ in range(dots):
        total += add
        add /= 2
    return 4 * total


@dataclass
class BarState:
    """Running bar length for one hypothesis."""

    current: Fraction = Fraction(0)
    measurable: bool = True
    after_chord: bool = False
    polyphonic: bool = False

    def copy(self) -> BarState:
        return BarState(self.current, self.measurable, self.after_chord, self.polyphonic)

    def add(self, rhythm: str, position: str) -> Fraction | None:
        """Consume one token. Returns the bar length when the token closes a measurable bar."""
        if position in ("lower", "lower2", "upper2"):
            self.polyphonic = True
        if rhythm == "chord":
            self.after_chord = True
            return None
        if rhythm.startswith(("note_", "rest_")):
            if self.after_chord:
                self.after_chord = False
                return None
            quarters = symbol_quarters(rhythm)
            if quarters is None:
                self.measurable = False
            else:
                self.current += quarters
            return None
        self.after_chord = False
        closes = rhythm in BAR_CLOSERS or rhythm in {"EOS", "repeatStart"}
        if not closes:
            return None
        length = self.current if (self.measurable and self.current > 0) else None
        self.current = Fraction(0)
        self.measurable = True
        return None if self.polyphonic else length


def bar_lengths(rhythms: Iterable[tuple[str, str]]) -> list[tuple[Fraction, bool]]:
    """(length, has_tuplet) for every measurable bar in a token stream of (rhythm, position)."""
    state = BarState()
    result = []
    tuplet = False
    for rhythm, position in [*rhythms, ("EOS", NONOTE)]:
        if rhythm.startswith(("note_", "rest_")):
            digits = "".join(ch for ch in rhythm.split("_", 1)[1] if ch.isdigit())
            tuplet = tuplet or (digits != "" and int(digits) % 3 == 0 and int(digits) > 0)
        length = state.add(rhythm, position)
        if rhythm in BAR_CLOSERS or rhythm in ("EOS", "repeatStart"):
            if length is not None:
                result.append((length, tuplet))
            tuplet = False
    return result


def reference_bar_length(bars: Iterable[tuple[Fraction, bool]]) -> Fraction | None:
    """The page's typical bar length, using Codex's rule from ``timing_diagnostics``.

    Only bars without tuplets vote; the winner needs at least three votes and must be unique.
    """
    counts = Counter(length for length, tuplet in bars if not tuplet)
    ranked = counts.most_common()
    if ranked and ranked[0][1] >= 3 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
        return ranked[0][0]
    return None


def bar_penalty(length: Fraction, reference: Fraction | None, weight: float) -> float:
    """Soft penalty (in log-probability units) for a closed bar of ``length`` quarters.

    Every bar is held to the page reference, including the first: lead sheets in the
    evaluation sets write pickups as full bars padded with rests, and exempting short first
    bars (tried 2026-11-12) only let the beam drop rests on continuation pages.
    """
    if reference is None or weight <= 0:
        return 0.0
    return weight * float(abs(length - reference))


# --------------------------------------------------------------------------------------------
# Consistent symbol construction (pure apart from numpy, unit tested)
# --------------------------------------------------------------------------------------------


def log_softmax(values: Any) -> Any:
    import numpy as np

    shifted = values - values.max()
    return shifted - np.log(np.exp(shifted).sum())


@dataclass
class Vocab:
    rhythm: dict[str, int]
    pitch: dict[str, int]
    lift: dict[str, int]
    articulation: dict[str, int]
    slur: dict[str, int]
    position: dict[str, int]
    inv_rhythm: dict[int, str] = field(init=False)
    inv_pitch: dict[int, str] = field(init=False)
    inv_lift: dict[int, str] = field(init=False)
    inv_articulation: dict[int, str] = field(init=False)
    inv_slur: dict[int, str] = field(init=False)
    inv_position: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        for name in ("rhythm", "pitch", "lift", "articulation", "slur", "position"):
            setattr(self, f"inv_{name}", {v: k for k, v in getattr(self, name).items()})
        self.real_pitch_ids = [i for t, i in self.pitch.items() if t not in (NONOTE, EMPTY)]
        self.control_pitch_ids = [self.pitch[t] for t in (NONOTE, EMPTY) if t in self.pitch]
        self.note_lift_ids = [i for t, i in self.lift.items() if t != NONOTE]
        self.plain_articulation_ids = [
            i for t, i in self.articulation.items() if not (set(t.split("_")) & GATED_ARTICULATIONS)
        ]
        self.special_rhythm_ids = {self.rhythm[t] for t in ("PAD", "BOS") if t in self.rhythm}


def _best(logp: Any, ids: list[int]) -> tuple[int, float]:
    best = max(ids, key=lambda i: logp[i])
    return best, float(logp[best])


@dataclass
class Candidate:
    rhythm: int
    pitch: int
    lift: int
    articulation: int
    slur: int
    position: int
    score: float  # joint log-probability used for ranking
    gated: dict | None = None


def build_candidates(
    vocab: Vocab,
    heads: dict[str, Any],
    *,
    consistent: bool,
    top_k: int,
    ornament_min_prob: float | None,
    forced_rhythm: int | None = None,
) -> list[Candidate]:
    """Turn one decoder step's head outputs into ranked candidate symbols.

    ``heads`` holds log-probabilities for the last position of each head. Without ``consistent``
    the fields are the heads' independent argmaxes (upstream behavior). With ``consistent`` each
    candidate rhythm gets a matching pitch: notes take the most likely *real* pitch, rests take
    "no pitch" and control symbols a placeholder. Either way the score is the rhythm
    log-probability only: the rhythm head decides what kind of symbol it is, the pitch head only
    which pitch a note has.

    (A first version added the pitch log-probability to the score. Where the pitch head wrongly
    says "no pitch", that made every non-note token look better than the real note, and it
    inserted a phantom ``dynamic_f`` at Airegin m26. So the pitch head no longer votes on type.)
    """
    rhythm_lp = heads["rhythm"]
    if forced_rhythm is not None:
        rhythm_ids = [forced_rhythm]
    else:
        order = rhythm_lp.argsort()[::-1]
        rhythm_ids = [int(i) for i in order if int(i) not in vocab.special_rhythm_ids][:top_k]

    import numpy as np

    lift_argmax = int(np.argmax(heads["lift"]))
    pitch_argmax = int(np.argmax(heads["pitch"]))
    articulation = int(np.argmax(heads["articulation"]))
    slur = int(np.argmax(heads["slur"]))
    position = int(np.argmax(heads["position"]))

    gated = None
    if ornament_min_prob is not None:
        token = vocab.inv_articulation[articulation]
        probability = float(np.exp(heads["articulation"][articulation]))
        if set(token.split("_")) & GATED_ARTICULATIONS and probability < ornament_min_prob:
            replacement, _ = _best(heads["articulation"], vocab.plain_articulation_ids)
            gated = {
                "token": token,
                "probability": probability,
                "replacement": vocab.inv_articulation[replacement],
            }
            articulation = replacement
        elif set(token.split("_")) & GATED_ARTICULATIONS:
            gated = {"token": token, "probability": probability, "replacement": None}

    candidates = []
    for rhythm in rhythm_ids:
        name = vocab.inv_rhythm[rhythm]
        score = float(rhythm_lp[rhythm])
        pitch, lift = pitch_argmax, lift_argmax
        if consistent:
            if name.startswith("note_"):
                pitch, _ = _best(heads["pitch"], vocab.real_pitch_ids)
                lift, _ = _best(heads["lift"], vocab.note_lift_ids)
            elif name.startswith("rest_"):
                pitch = vocab.pitch[EMPTY]
                lift = vocab.lift[EMPTY] if EMPTY in vocab.lift else lift
            else:
                pitch, _ = _best(heads["pitch"], vocab.control_pitch_ids)
        candidates.append(
            Candidate(rhythm, pitch, lift, articulation, slur, position, score, gated)
        )
    return sorted(candidates, key=lambda c: c.score, reverse=True)


# --------------------------------------------------------------------------------------------
# Token-level cleanups (pure, unit tested)
# --------------------------------------------------------------------------------------------


def pair_orphan_slur_stops(symbols: list[Any]) -> list[dict]:
    """Give a lone slur stop its start on the previous same-pitch note, so it exports as a tie.

    Upstream only turns a slur into a tie when both ends were predicted. A missed start leaves a
    dangling slur stop; when the previous note in the same staff position has the same pitch and
    alteration and no slur is open, that stop can only be the end of a tie.
    """
    changes = []
    open_slurs: Counter[str] = Counter()
    previous: dict[str, Any] = {}
    after_chord = False
    for index, symbol in enumerate(symbols):
        rhythm = symbol.rhythm
        if rhythm == "chord":
            after_chord = True
            continue
        if not rhythm.startswith(("note_", "rest_")):
            after_chord = False
            continue
        key = "lower" if str(symbol.position).startswith("lower") else "upper"
        slur = symbol.slur or ""
        if "slurStop" in slur:
            before = previous.get(key)
            if (
                open_slurs[key] == 0
                and before is not None
                and rhythm.startswith("note_")
                and before.rhythm.startswith("note_")
                and before.pitch == symbol.pitch
                and before.lift == symbol.lift
                and "slurStart" not in (before.slur or "")
            ):
                before.slur = "slurStart_slurStop" if before.slur == "slurStop" else "slurStart"
                changes.append({"symbol_index": index, "pitch": symbol.pitch})
            else:
                open_slurs[key] = max(0, open_slurs[key] - 1)
        if "slurStart" in slur:
            open_slurs[key] += 1
        if not after_chord:
            previous[key] = symbol
        after_chord = False
    return changes


def normalize_meter(symbols: list[Any]) -> list[dict]:
    """Drop time signatures which do not change the bar length; prefer x/4 over x/2.

    The vocabulary stores only a time signature's denominator; the exporter infers the numerator
    from bar lengths. A later signature is kept only when the bars after it have a different
    typical length than the bars before it. A ``/2`` reading whose bars are an even whole number
    of quarters is written as ``/4`` (4/4 rather than 2/2): in lead sheets 4/4 is far more common,
    and both have identical durations, so only the printed signature changes. Modifies
    ``symbols`` in place.
    """
    segments: list[tuple[int | None, list[tuple[str, str]]]] = [(None, [])]
    for index, symbol in enumerate(symbols):
        if symbol.rhythm.startswith("timeSignature"):
            segments.append((index, []))
        else:
            segments[-1][1].append((symbol.rhythm, str(symbol.position)))

    changes: list[dict] = []
    drop: set[int] = set()
    previous_length = None
    seen_signature = False
    for signature_index, tokens in segments:
        counts = Counter(length for length, _ in bar_lengths(tokens))
        length = counts.most_common(1)[0][0] if counts else None
        if signature_index is not None:
            if seen_signature and (length is None or length == previous_length):
                drop.add(signature_index)
                changes.append(
                    {"symbol_index": signature_index, "dropped": symbols[signature_index].rhythm}
                )
            else:
                seen_signature = True
                if (
                    symbols[signature_index].rhythm == "timeSignature/2"
                    and length is not None
                    and length.denominator == 1
                    and length >= 4
                    and length % 2 == 0
                ):
                    symbols[signature_index].rhythm = "timeSignature/4"
                    changes.append(
                        {"symbol_index": signature_index, "rewritten": "timeSignature/2->/4"}
                    )
        if length is not None:
            previous_length = length
    symbols[:] = [s for i, s in enumerate(symbols) if i not in drop]
    return changes


# --------------------------------------------------------------------------------------------
# XML beaming (pure, unit tested)
# --------------------------------------------------------------------------------------------

_BEAMABLE = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4}
_AFTER_BEAM = {"notations", "lyric", "play", "listen"}


def _group_length(beats: int, beat_type: int) -> Fraction:
    """Beam group length in quarter notes for a time signature."""
    if beat_type == 8 and beats % 3 == 0:
        return Fraction(3, 2)
    if beat_type == 4 and beats % 2 == 0:
        return Fraction(2)  # 4/4, 2/4: eighths beam in half bars
    if beat_type == 2:
        return Fraction(2)
    return Fraction(4, beat_type)


def _insert_beam(note: ET.Element, number: int, value: str) -> None:
    beam = ET.Element("beam", number=str(number))
    beam.text = value
    for index, child in enumerate(list(note)):
        if child.tag in _AFTER_BEAM:
            note.insert(index, beam)
            return
    note.append(beam)


def add_beams(root: ET.Element) -> int:
    """Beam monophonic eighth-and-shorter notes within beat groups. Returns notes beamed."""
    beamed = 0
    for part in root.iter("part"):
        division = 1
        beats, beat_type = 4, 4
        for measure in part.findall("measure"):
            for attributes in measure.findall("attributes"):
                division = int(attributes.findtext("divisions") or division)
                if attributes.find("time") is not None:
                    beats = int(attributes.findtext("time/beats") or beats)
                    beat_type = int(attributes.findtext("time/beat-type") or beat_type)
            if measure.find("backup") is not None or measure.find(".//beam") is not None:
                continue
            group = _group_length(beats, beat_type)
            time = Fraction(0)
            runs: list[list[ET.Element]] = [[]]
            current_group = None
            for note in measure.findall("note"):
                if note.find("chord") is not None:
                    continue
                duration = Fraction(int(note.findtext("duration") or 0), division)
                level = _BEAMABLE.get(note.findtext("type") or "")
                group_index = int(time // group)
                if (
                    note.find("rest") is not None
                    or note.find("grace") is not None
                    or level is None
                    or group_index != current_group
                ):
                    runs.append([])
                if note.find("rest") is None and note.find("grace") is None and level:
                    runs[-1].append(note)
                current_group = group_index
                time += duration
            for run in runs:
                if len(run) < 2:
                    continue
                levels = [_BEAMABLE[n.findtext("type") or ""] for n in run]
                for position, note in enumerate(run):
                    for number in range(1, levels[position] + 1):
                        left = position > 0 and levels[position - 1] >= number
                        right = position < len(run) - 1 and levels[position + 1] >= number
                        if number == 1:
                            value = "begin" if position == 0 else "end" if not right else "continue"
                        elif left and right:
                            value = "continue"
                        elif right:
                            value = "begin"
                        elif left:
                            value = "end"
                        else:
                            value = "forward hook" if position == 0 else "backward hook"
                        _insert_beam(note, number, value)
                    beamed += 1
    return beamed


# --------------------------------------------------------------------------------------------
# Runtime hooks
# --------------------------------------------------------------------------------------------


@dataclass
class _Hyp:
    symbols: list[Any]
    last: tuple[int, int, int, int, int]
    cache: list[Any]
    score: float
    bar: BarState
    penalties: list[dict]
    done: bool = False
    alternatives: list[list] = field(default_factory=list)
    reinterpretations: list[dict] = field(default_factory=list)


class _PageState:
    def __init__(self) -> None:
        self.pass_name = "final"
        self.reference: Fraction | None = None
        self.carried: list[str] | None = None
        self.prefix: list[str] | None = None
        self.single_line = True
        self.meter_seen = False
        self.staffs: list[dict] = []
        self.encoder_cache: dict[str, Any] = {}


CURRENT: _PageState | None = None


def current_pass() -> str:
    return CURRENT.pass_name if CURRENT is not None else "final"


def _make_generate(options: DecodeOptions, page: _PageState, evidence: dict):
    import numpy as np

    def generate(self, start_tokens, nonote_tokens, **kwargs):
        from homr.transformer.vocabulary import EncodedSymbol

        vocab = getattr(self, "_mv_vocab", None)
        if vocab is None:
            c = self.config
            vocab = Vocab(
                c.rhythm_vocab,
                c.pitch_vocab,
                c.lift_vocab,
                c.articulation_vocab,
                c.slur_vocab,
                c.position_vocab,
            )
            self._mv_vocab = vocab
        context_full = kwargs["context"]
        context_reduced = context_full[:, :1]
        # init_cache() allocates the empty cache on the session's device (GPU when CUDA is
        # active). Every hypothesis keeps its cache there as OrtValues and passes it back in
        # through io_binding, so the cache never travels to host memory between steps. Only
        # the small per-step outputs (logits, attention) are copied back.
        empty_cache, kv_in, kv_out = self.init_cache()
        output_names = self.output_names + kv_out
        start = int(np.asarray(start_tokens).reshape(-1)[0])
        nonote = int(np.asarray(nonote_tokens).reshape(-1)[0])
        prefix_ids = [vocab.rhythm[t] for t in page.prefix or [] if t in vocab.rhythm]
        beam_width = max(1, options.beam_width) if page.pass_name == "final" else 1
        weight = options.bar_penalty if page.pass_name == "final" else 0.0
        reference = page.reference
        reinterpret_meter = beam_width > 1 and weight > 0 and reference is not None
        gated_log: list[dict] = []

        beams = [
            _Hyp(
                [],
                (start, nonote, nonote, nonote, nonote),
                empty_cache,
                0.0,
                BarState(),
                [],
            )
        ]
        for step in range(self.max_seq_len):
            if all(b.done for b in beams):
                break
            expanded: list[_Hyp] = []
            for hyp in beams:
                if hyp.done:
                    expanded.append(hyp)
                    continue
                r, p, lf, a, sl = hyp.last
                # Keep the host arrays referenced until the run returns.
                host_inputs = {
                    "rhythms": np.array([[r]], dtype=np.int64),
                    "pitchs": np.array([[p]], dtype=np.int64),
                    "lifts": np.array([[lf]], dtype=np.int64),
                    "articulations": np.array([[a]], dtype=np.int64),
                    "slurs": np.array([[sl]], dtype=np.int64),
                    "context": np.ascontiguousarray(
                        context_full if step == 0 else context_reduced
                    ),
                    "cache_len": np.array([step], dtype=np.int64),
                }
                # A fresh binding per run is required: reusing self.io_binding lets ORT write a
                # later run's outputs into buffers that sibling hypotheses still hold as their
                # cache, which silently changed the decoded music.
                binding = self.net.io_binding()
                for name, value in host_inputs.items():
                    binding.bind_cpu_input(name, value)
                for name, value in zip(kv_in, hyp.cache, strict=True):
                    binding.bind_ortvalue_input(name, value)
                for name in output_names:
                    binding.bind_output(name, self.device, self.device_id)
                self.net.run_with_iobinding(iobinding=binding)
                outputs = binding.get_outputs()
                cache = outputs[7:]
                logits = [outputs[i].numpy() for i in range(6)]
                heads = {
                    "rhythm": log_softmax(logits[0][0, -1].astype(np.float64)),
                    "pitch": log_softmax(logits[1][0, -1].astype(np.float64)),
                    "lift": log_softmax(logits[2][0, -1].astype(np.float64)),
                    "position": log_softmax(logits[3][0, -1].astype(np.float64)),
                    "articulation": log_softmax(logits[4][0, -1].astype(np.float64)),
                    "slur": log_softmax(logits[5][0, -1].astype(np.float64)),
                }
                attention = outputs[6].numpy()
                top = heads["rhythm"].argsort()[::-1][:4]
                alts = [
                    [vocab.inv_rhythm[int(i)], round(float(heads["rhythm"][i]), 2)] for i in top
                ]
                forced = prefix_ids[step] if step < len(prefix_ids) else None
                candidates = build_candidates(
                    vocab,
                    heads,
                    consistent=options.consistent_heads,
                    top_k=max(beam_width, 5 if options.consistent_heads else 1),
                    ornament_min_prob=options.ornament_min_prob,
                    forced_rhythm=forced,
                )
                if not options.consistent_heads and beam_width == 1:
                    candidates = candidates[:1]
                expansions = []
                for cand in candidates[: max(beam_width, 1) + (4 if beam_width > 1 else 0)]:
                    name = vocab.inv_rhythm[cand.rhythm]
                    reinterpret = (
                        reinterpret_meter
                        and name.startswith("timeSignature/")
                        and (
                            page.meter_seen
                            or any(s.rhythm.startswith(("note_", "rest_")) for s in hyp.symbols)
                        )
                    )
                    if reinterpret:
                        # Jazz-font quarter rests are misread as "/4" time signatures. Let the bar
                        # length decide between the three readings; the decoder keeps seeing its
                        # own token either way.
                        expansions += [(cand, name), (cand, None), (cand, "rest_4")]
                    else:
                        expansions.append((cand, name))
                for cand, emitted in expansions:
                    name = vocab.inv_rhythm[cand.rhythm]
                    bar = hyp.bar.copy()
                    position = vocab.inv_position[cand.position]
                    reinterpretations = hyp.reinterpretations
                    if emitted != name:
                        reinterpretations = [
                            *reinterpretations,
                            {"step": step, "decoded": name, "emitted": emitted},
                        ]
                    if emitted is None:
                        expanded.append(
                            _Hyp(
                                hyp.symbols,
                                (
                                    cand.rhythm,
                                    cand.pitch,
                                    cand.lift,
                                    cand.articulation,
                                    cand.slur,
                                ),
                                cache,
                                hyp.score + cand.score,
                                bar,
                                hyp.penalties,
                                alternatives=[*hyp.alternatives, alts],
                                reinterpretations=reinterpretations,
                            )
                        )
                        continue
                    closed = bar.add(emitted, position)
                    penalty = bar_penalty(closed, reference, weight) if closed is not None else 0
                    if reinterpret_meter and emitted == name and name.startswith("timeSignature/"):
                        if page.meter_seen or any(
                            s.rhythm.startswith(("note_", "rest_")) for s in hyp.symbols
                        ):
                            # Prior: a second time signature on a lead sheet is rare.
                            penalty += options.meter_change_penalty
                    penalties = hyp.penalties
                    if penalty:
                        penalties = [
                            *penalties,
                            {
                                "step": step,
                                "bar_quarters": str(closed),
                                "penalty": penalty,
                            },
                        ]
                    if cand.gated is not None and cand is candidates[0]:
                        gated_log.append({"step": step, **cand.gated})
                    if cand.rhythm == self.eos_token:
                        expanded.append(
                            _Hyp(
                                hyp.symbols,
                                hyp.last,
                                cache,
                                hyp.score + cand.score - penalty,
                                bar,
                                penalties,
                                done=True,
                                alternatives=[*hyp.alternatives, alts],
                                reinterpretations=reinterpretations,
                            )
                        )
                        continue
                    if emitted == name:
                        symbol = EncodedSymbol(
                            rhythm=name,
                            pitch=vocab.inv_pitch[cand.pitch],
                            lift=vocab.inv_lift[cand.lift],
                            articulation=vocab.inv_articulation[cand.articulation],
                            slur=vocab.inv_slur[cand.slur],
                            position=position,
                            coordinates=attention,
                        )
                    else:
                        symbol = EncodedSymbol(
                            emitted,
                            EMPTY,
                            EMPTY,
                            EMPTY,
                            EMPTY,
                            "upper",
                            coordinates=attention,
                        )
                    expanded.append(
                        _Hyp(
                            [*hyp.symbols, symbol],
                            (
                                cand.rhythm,
                                cand.pitch,
                                cand.lift,
                                cand.articulation,
                                cand.slur,
                            ),
                            cache,
                            hyp.score + cand.score - penalty,
                            bar,
                            penalties,
                            alternatives=[*hyp.alternatives, alts],
                            reinterpretations=reinterpretations,
                        )
                    )
            expanded.sort(key=lambda h: h.score, reverse=True)
            beams = expanded[:beam_width]
        best = max(beams, key=lambda h: h.score)
        page.staffs[-1]["decode"] = {
            "pass": page.pass_name,
            "beam_width": beam_width,
            "bar_penalty": weight,
            "reference_bar_quarters": str(reference) if reference is not None else None,
            "score": best.score,
            "penalties": best.penalties,
            "gated_articulations": gated_log,
            "rhythm_alternatives": best.alternatives,
            "reinterpretations": best.reinterpretations,
        }
        if any(s.rhythm.startswith("timeSignature") for s in best.symbols):
            page.meter_seen = True
        return best.symbols

    return generate


@contextmanager
def decoding_experiment(hm, options: DecodeOptions, carry: dict | None = None) -> Iterator[dict]:
    """Install B/C/E hooks into an upstream HOMR process; restore everything afterwards.

    ``carry`` is shared between consecutive pages of one document: the last printed clef/key
    of a page becomes the starting context of the next page.
    """
    global CURRENT  # noqa: PLW0603
    evidence: dict = {
        "options": options.__dict__.copy(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if not options.enabled:
        yield evidence
        return
    from homr import staff_parsing
    from homr.jazzify.refinement import _leading_clef_key_anchor
    from homr.transformer import decoder_inference, encoder_inference

    page = _PageState()
    if carry is not None and carry.get("clef_key"):
        page.carried = list(carry["clef_key"])
    evidence["carried_in"] = list(page.carried) if page.carried else None
    CURRENT = page
    originals: list[tuple[Any, str, Any]] = []

    def replace(owner, name, function):
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, function)

    anchors: dict[str, list] = {"clefs": []}
    predict_symbols = hm.predict_symbols

    def capture_symbols(*args, **kwargs):
        symbols = predict_symbols(*args, **kwargs)
        anchors["clefs"] = list(symbols.clefs_keys)
        return symbols

    replace(hm, "predict_symbols", capture_symbols)

    if options.changes_decoding:
        replace(
            decoder_inference.ScoreDecoder,
            "generate",
            _make_generate(options, page, evidence),
        )
        encode = encoder_inference.Encoder.generate

        def cached_encode(self, x):
            key = hashlib.sha1(x.tobytes()).hexdigest()  # noqa: S324 - cache key only
            if key not in page.encoder_cache:
                page.encoder_cache[key] = encode(self, x)
            return page.encoder_cache[key]

        replace(encoder_inference.Encoder, "generate", cached_encode)

    parse_staff_image = staff_parsing.parse_staff_image

    def staff_hook(debug, index, staff, image, regions, config, *rest, **kwargs):
        has_anchor = any(_leading_clef_key_anchor(box, staff) for box in anchors["clefs"])
        may_force = (
            options.carry_clef_key
            and page.single_line
            and not staff.is_grandstaff
            and not has_anchor
        )
        if may_force and page.carried is not None:
            page.prefix = list(page.carried)
        elif may_force and options.default_clef:
            # Nothing printed and nothing carried (e.g. a continuation page on its own): force
            # only the clef and let the model read the key signature itself.
            page.prefix = [options.default_clef]
        else:
            page.prefix = None
        record = {
            "staff_index": index,
            "leading_clef_key_anchor": has_anchor,
            "forced_prefix": page.prefix,
        }
        page.staffs.append(record)
        result = parse_staff_image(debug, index, staff, image, regions, config, *rest, **kwargs)
        head = []
        for symbol in result:
            if symbol.rhythm.startswith(("note", "rest", "barline")):
                break
            if symbol.rhythm.startswith(("clef", "keySignature")):
                head.append(symbol.rhythm)
        record["decoded_head"] = head
        clef = next((h for h in head if h.startswith("clef")), None)
        key = next((h for h in head if h.startswith("keySignature")), None)
        if (
            options.carry_clef_key
            and has_anchor
            and page.single_line
            and not staff.is_grandstaff
            and clef is not None
            and (
                clef not in options.trusted_clefs
                # Once a document has a trusted clef, a different reading on a later printed
                # clef is treated as a misread (Omnibook: treble read as bass on 15 staffs).
                or (page.carried is not None and clef != page.carried[0])
            )
        ):
            # A printed clef read as alto/tenor on a single-line lead sheet is almost always a
            # misread hand-drawn treble clef (Omnibook pages 3 and 38). Re-decode the staff with
            # the carried clef, or the default, and let the model read the key itself.
            fallback = (page.carried or [options.default_clef])[0]
            if fallback:
                record["rejected_clef"] = clef
                page.prefix = [fallback]
                record["forced_prefix"] = page.prefix
                result = parse_staff_image(
                    debug, index, staff, image, regions, config, *rest, **kwargs
                )
                head = [
                    s.rhythm for s in result[:4] if s.rhythm.startswith(("clef", "keySignature"))
                ]
                record["decoded_head"] = head
                clef = next((h for h in head if h.startswith("clef")), None)
                key = next((h for h in head if h.startswith("keySignature")), None)
        # Only a staff with a printed clef/key may set the carried context. (An earlier version
        # also adopted the first staff's *guess* when nothing was carried yet; on Omnibook
        # continuation pages that forced alto and bass clefs onto treble music.)
        if options.carry_clef_key and has_anchor:
            if clef and key and clef in options.trusted_clefs:
                page.carried = [clef, key]
                if carry is not None:
                    carry["clef_key"] = list(page.carried)
        page.prefix = None
        return result

    replace(staff_parsing, "parse_staff_image", staff_hook)

    clean_up_voice = staff_parsing._clean_up_voice

    def clean_hook(symbols):
        result = clean_up_voice(symbols)
        if options.cleanup and page.pass_name == "final":
            evidence.setdefault("cleanup", []).append(
                {
                    "orphan_slur_ties": pair_orphan_slur_stops(result),
                    "meter": normalize_meter(result),
                }
            )
        return result

    replace(staff_parsing, "_clean_up_voice", clean_hook)

    parse_staffs = staff_parsing.parse_staffs

    def two_pass(debug, staffs, *args, **kwargs):
        page.single_line = all(len(row.staffs) == 1 for row in staffs)
        evidence["single_line_page"] = page.single_line
        needs_reference = options.beam_width > 1 and options.bar_penalty > 0
        if needs_reference:
            page.pass_name = "reference"
            carried = page.carried
            voices = parse_staffs(debug, staffs, *args, **kwargs)
            bars = [
                bar
                for voice in voices
                for bar in bar_lengths((s.rhythm, str(s.position)) for s in voice)
            ]
            page.reference = reference_bar_length(bars)
            evidence["reference_pass"] = {
                "staffs": page.staffs,
                "bar_lengths": [str(length) for length, _ in bars],
                "reference_bar_quarters": (
                    str(page.reference) if page.reference is not None else None
                ),
            }
            page.staffs = []
            page.carried = carried
            page.meter_seen = False
        page.pass_name = "final"
        voices = parse_staffs(debug, staffs, *args, **kwargs)
        evidence["staffs"] = page.staffs
        return voices

    replace(staff_parsing, "parse_staffs", two_pass)
    replace(hm, "parse_staffs", two_pass)
    try:
        yield evidence
    finally:
        for owner, name, function in reversed(originals):
            setattr(owner, name, function)
        CURRENT = None


def postprocess_xml(path: Path, options: DecodeOptions, accidentals_module: Any) -> dict:
    """Write displayed accidentals (A) and, with cleanup, beams (E) into the saved MusicXML."""
    tree = ET.parse(path)  # noqa: S314 - MusicXML written by this process.
    root = tree.getroot()
    result = {"accidentals_added": accidentals_module.add_display_accidentals(root)}
    if options.cleanup:
        result["beamed_notes"] = add_beams(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return result
