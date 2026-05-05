from app import g2p_convert, g2p_mode, load_g2p_engine

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

print("G2P mode:", g2p_mode)
