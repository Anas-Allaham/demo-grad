"""
Offline exercise-bank builder.

Tags each sentence in ``data/seed_sentences.txt`` with the app's own G2P
pipeline and loads it into the ``exercise_bank`` table so ``/practice/next``
has real, verified content to retrieve from.

Run once (and again whenever seed_sentences.txt changes):

    python scripts/build_exercise_bank.py

Imports ``g2p_service`` (not app.py), so building the bank never pulls in
torch / transformers / the Wav2Vec2 model -- only the lightweight G2P path.
Uses ``phoneme_vectors_professional`` as the single canonicalization source.
"""

import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import content  # noqa: E402
import db  # noqa: E402
from g2p_service import g2p_convert, load_g2p_engine  # noqa: E402
from phoneme_vectors_professional import (  # noqa: E402
    ASSESSABLE_INVENTORY,
    canonicalize_phoneme,
    validate_g2p_inventory,
)
from tokenization import ipa_to_tokens  # noqa: E402

SEED_PATH = BASE_DIR / "data" / "seed_sentences.txt"


def load_seed_sentences():
    if not SEED_PATH.exists():
        raise SystemExit(
            f"Seed file not found: {SEED_PATH}\n"
            "Create data/seed_sentences.txt (one sentence per line) and re-run."
        )
    with SEED_PATH.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _difficulty_band(level_proxy: float) -> str:
    d = content.normalize_difficulty(level_proxy)
    if d < 0.34:
        return "easy"
    if d < 0.67:
        return "medium"
    return "hard"


def retag_existing_bank():
    """Re-canonicalize every sentence already in the bank with the current G2P
    + tokenizer. Idempotent and content-only (no user history touched), so an
    old bank tagged by a previous tokenizer becomes consistent again."""
    retagged = 0
    for sentence in db.get_all_sentences():
        reference_ipa = g2p_convert(sentence["text"])
        counts = dict(Counter(ipa_to_tokens(reference_ipa)))
        if not counts:
            continue
        if counts != sentence["phoneme_counts"] or reference_ipa != sentence["reference_ipa"]:
            db.update_sentence_tags(sentence["id"], reference_ipa, counts)
            retagged += 1
    if retagged:
        print(f"Re-tagged {retagged} existing sentence(s) to the canonical inventory.")
    return retagged


def main():
    load_g2p_engine()
    db.init_db()

    retag_existing_bank()

    sentences = load_seed_sentences()
    inserted = 0
    skipped = 0
    duplicates = 0
    rejected_examples = []

    for text in sentences:
        tagged = content.tag_sentence(text, g2p_convert, ipa_to_tokens)
        if not content.is_valid_tagging(tagged):
            skipped += 1
            if len(rejected_examples) < 10:
                rejected_examples.append(text)
            continue

        sentence_id = db.insert_sentence(
            text=tagged["text"],
            reference_ipa=tagged["reference_ipa"],
            word_count=tagged["word_count"],
            level_proxy=tagged["level_proxy"],
            phoneme_counts=tagged["phoneme_counts"],
            source="retrieval",
        )
        if sentence_id is None:
            duplicates += 1
            continue
        inserted += 1

    # ---- Coverage / difficulty computed over the WHOLE bank ----
    coverage = Counter()
    difficulty_bands = Counter()
    all_phonemes = set()
    for sentence in db.get_all_sentences():
        difficulty_bands[_difficulty_band(sentence["level_proxy"])] += 1
        for phoneme in sentence["phoneme_counts"]:
            coverage[phoneme] += 1
            all_phonemes.add(phoneme)

    # ---- Report ----
    print(f"\nInserted:   {inserted}")
    print(f"Duplicates: {duplicates} (already in the bank)")
    print(f"Rejected:   {skipped} (untaggable / out-of-vocabulary)")
    if rejected_examples:
        print("  e.g. " + " | ".join(rejected_examples[:5]))
    print(f"Total in bank: {db.count_exercise_bank()}")

    print("\nDifficulty distribution (provisional readability proxy):")
    for band in ("easy", "medium", "hard"):
        print(f"  {band:>6}: {difficulty_bands.get(band, 0)}")

    report = validate_g2p_inventory(all_phonemes)
    if not report["ok"]:
        print("\nWARNING: phonemes outside the canonical scoring inventory were produced:")
        print("  ", report["unsupported"])

    print("\nCoverage per canonical assessable phoneme:")
    zero_coverage, low_coverage = [], []
    for phoneme in sorted(ASSESSABLE_INVENTORY):
        count = coverage.get(phoneme, 0)
        flag = ""
        if count == 0:
            zero_coverage.append(phoneme)
            flag = "  <-- ZERO COVERAGE"
        elif count < 3:
            low_coverage.append(phoneme)
            flag = "  <-- LOW COVERAGE"
        print(f"  {phoneme:>4}: {count:3d} sentence(s){flag}")

    if zero_coverage:
        print(f"\nZERO coverage (can never be targeted): {zero_coverage}")
    if low_coverage:
        print(f"LOW coverage (<3 sentences): {low_coverage}")
    if not zero_coverage and not low_coverage:
        print("\nEvery assessable phoneme has adequate (>=3) coverage.")


if __name__ == "__main__":
    main()
