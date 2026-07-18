"""
Offline exercise-bank builder: tags each sentence in data/seed_sentences.txt
with the app's own G2P pipeline and loads it into the exercise_bank table,
so `/practice/next` has real, verified content to retrieve from.

Run once (and again any time seed_sentences.txt changes):
    python scripts/build_exercise_bank.py

Safe to import app.py here: app.py only starts the Flask dev server and
loads the G2P/Wav2Vec2 models inside `if __name__ == "__main__":` or
lazily inside functions -- never at module import time -- so importing it
as a library never spins up the server or the audio model.
"""

import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import app as pronunciation_app  # noqa: E402
import content  # noqa: E402
import db  # noqa: E402
from phoneme_vectors import KNOWN_IPA_PHONEMES, canonicalize_phoneme  # noqa: E402

SEED_PATH = BASE_DIR / "data" / "seed_sentences.txt"


def load_seed_sentences():
    with SEED_PATH.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    pronunciation_app.load_g2p_engine()
    db.init_db()

    sentences = load_seed_sentences()
    inserted = 0
    skipped = 0
    duplicates = 0
    coverage = Counter()

    for text in sentences:
        tagged = content.tag_sentence(text, pronunciation_app.g2p_convert, pronunciation_app.ipa_to_tokens)
        if not content.is_valid_tagging(tagged):
            print(f"SKIP (no phonemes recognized): {text}")
            skipped += 1
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
        for phoneme in tagged["phoneme_counts"]:
            coverage[phoneme] += 1

    print(f"\nInserted {inserted} new sentences ({skipped} skipped as untaggable, {duplicates} already in the bank).")

    print("\nCoverage report (sentences containing each known phoneme):")
    zero_coverage = []
    known = sorted({canonicalize_phoneme(p) for p in KNOWN_IPA_PHONEMES if canonicalize_phoneme(p)})
    for phoneme in known:
        count = coverage.get(phoneme, 0)
        flag = "  <-- LOW/ZERO COVERAGE" if count < 3 else ""
        print(f"  {phoneme:>4}: {count:3d} sentence(s){flag}")
        if count == 0:
            zero_coverage.append(phoneme)

    if zero_coverage:
        print(f"\nWARNING: these phonemes have ZERO coverage and can never be targeted: {zero_coverage}")
        print("Add a few sentences containing them to data/seed_sentences.txt and re-run this script.")
    else:
        print("\nEvery known phoneme has at least one sentence.")


if __name__ == "__main__":
    main()
