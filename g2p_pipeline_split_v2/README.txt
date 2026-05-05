Files:
- heteronyms.json           : external lexicon with POS -> pronunciation mapping
- utils.py                  : shared helper functions
- pos_tagger.py             : POS tagging layer (spaCy)
- heteronym_lexicon.py      : loads and resolves heteronyms from JSON
- contextual_g2p.py         : main context-aware G2P engine
- main.py                   : example entry point

How to run:
1) Put all files in the same folder
2) Install dependencies:
   pip install spacy nemo_toolkit
   python -m spacy download en_core_web_sm
3) Run:
   python main.py
