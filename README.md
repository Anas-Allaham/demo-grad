# Pronunciation Feedback Web App — G2P Integrated

This is a local Flask web app for your graduation project pipeline:

```text
User text  -> bundled G2P -> expected IPA phonemes
User audio -> trained Wav2Vec2 -> spoken IPA phonemes
Expected + spoken phonemes -> DP alignment -> pronunciation feedback
```

## What is integrated now?

This version already includes your `g2p_pipeline_split_v2` folder.

The backend uses:

```text
g2p_pipeline_split_v2/contextual_g2p.py
g2p_pipeline_split_v2/heteronyms.json
g2p_pipeline_split_v2/cmudict-0.7b-ipa.txt
```

The G2P output is converted to this format:

```text
s k uː l | ɪ z | oʊ p ə n
```

That means:

- spaces separate phonemes
- `|` separates words
- the DP alignment compares phoneme tokens

## 1. Put your trained Wav2Vec2 model here

After unzipping this project, copy your downloaded trained model files into:

```text
model/my_wav2vec2_phoneme_model/
```

That folder should contain files like:

```text
config.json
vocab.json
tokenizer_config.json
preprocessor_config.json
model.safetensors
```

or:

```text
pytorch_model.bin
```

## 2. Install dependencies

### Recommended full install

This tries to use your context-aware G2P with spaCy + NeMo:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Easier minimal install

If NeMo is hard to install, use this:

```bash
pip install -r requirements-minimal.txt
```

The app will still work using the bundled IPA dictionary fallback.
The fallback is less context-aware, but it is good enough to run the demo.

## 3. Test G2P only

Before running the whole web app, test the text-to-phoneme part:

```bash
python test_g2p.py
```

Expected example style:

```text
TEXT: school is open
IPA : s k uː l | ɪ z | oʊ p ə n
```

## 4. Run the app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## 5. Optional: test G2P from the browser/backend

After running `python app.py`, you can test the G2P endpoint using curl:

```bash
curl -X POST http://127.0.0.1:5000/g2p \
  -H "Content-Type: application/json" \
  -d '{"text":"school is open"}'
```

## 6. If browser audio fails

Browser recordings are usually saved as `.webm`. `librosa` can load this if FFmpeg is installed.

If audio loading fails, install FFmpeg:

- Windows: install FFmpeg and add it to PATH
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

## Notes

This app runs locally on your computer. It does not upload audio to any online server unless you modify it.
