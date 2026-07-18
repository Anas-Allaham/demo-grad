"""Quick manual G2P smoke check: python test_g2p.py

Imports g2p_service directly (not app.py), so it runs without torch/Wav2Vec2.
"""

from g2p_service import g2p_convert, get_g2p_mode, load_g2p_engine

load_g2p_engine()

examples = [
    "school is open",
    "I read the book",
    "They record music",
    "The record was broken",
]

for text in examples:
    print("TEXT:", text)
    print("IPA :", g2p_convert(text))
    print()

print("G2P mode:", get_g2p_mode())
