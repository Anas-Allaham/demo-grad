from pathlib import Path
import time

from contextual_g2p import ContextAwareIpaG2p


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    heteronyms_path = base_dir / "heteronyms.json"
    ipa_dict_path = base_dir / "cmudict-0.7b-ipa.txt"

    if not heteronyms_path.exists():
        raise FileNotFoundError(f"heteronyms.json not found: {heteronyms_path}")

    if not ipa_dict_path.exists():
        raise FileNotFoundError(f"IPA dictionary not found: {ipa_dict_path}")

    g2p = ContextAwareIpaG2p(
        heteronyms_json_path=str(heteronyms_path),
        ipa_dict_path=str(ipa_dict_path),
    )

    examples = [
        # "I read the book everyday but you have read the book once and your sister read it last week",
        # "They record music at night.",
        # "The record was broken.",
        # "Please present the report.",
        # "The present is on the table.",
        # "Do you have a permit?",
        # "They permit visitors on weekends.",
        # "Schrödinger discussed physics in Göttingen."
        # "They traveled from Reykjavík to Ljubljana."
        # "The conference in Tübingen featured Noam Chomsky."
        # "She studied in Worcester before moving to Leicester."
        # "He lives near the Thames in Greenwich."
        # "The package was shipped to Arkansas via Illinois."

        # "The tokenization process affects generalization in transformers."
        # "We applied backpropagation to minimize cross-entropy loss."
        # "The model suffered from overfitting despite regularization."
        # "We evaluated phoneme error rate using a grapheme-to-phoneme model."
        # "The spectrogram was computed using mel-frequency scaling."
        # "Fine-tuning improved performance on out-of-vocabulary words."
        "aya is perfect "
    ]

    total_start = time.time()

    for text in examples:
        start = time.time()
        phonemes = g2p(text)
        elapsed = time.time() - start

        ipa_sentence = " ".join(phonemes)

        print("=" * 80)
        print("INPUT   :", text)
        print("IPA     :", ipa_sentence)
        print(f"EXECUTION TIME: {elapsed:.6f} seconds")

    total_elapsed = time.time() - total_start
    print("=" * 80)
    print(f"TOTAL EXECUTION TIME: {total_elapsed:.6f} seconds")


if __name__ == "__main__":
    main()