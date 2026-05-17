# Pronunciation Feedback System — Final Local Version

This is the final local Flask version of the pronunciation feedback project.

## Pipeline

```text
User text  -> G2P -> expected IPA phonemes
User audio -> Wav2Vec2 -> spoken IPA phonemes
Expected + spoken phonemes -> vector-weighted DP alignment -> feedback + weighted PER
Expected phonemes -> IPA reader guide
```

## Included features

- Text-to-IPA G2P using the bundled `g2p_pipeline_split_v2` folder.
- Local Wav2Vec2 phoneme model loading.
- Browser microphone recording.
- Noise reduction before Wav2Vec2 inference.
- Research-based phoneme distance using PanPhon articulatory feature edit distance.
- Weighted dynamic programming alignment.
- Minor / medium / major substitution labels.
- Weighted PER calculation from phoneme distances.
- IPA reader guide: explains how to read each expected phoneme.

## 1. Put your trained Wav2Vec2 model here

After unzipping this project, copy your downloaded trained model files into:

```text
model/my_wav2vec2_phoneme_model/
```

The folder should contain files such as:

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

## 2. Recommended environment

Using Conda on Windows:

```bash
conda create -n pronunciation-app python=3.10 -y
conda activate pronunciation-app
```

## 3. Install dependencies

Minimal demo install:

```bash
pip install -r requirements-minimal.txt
```

Full install with NeMo/spaCy context-aware G2P:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If NeMo is hard to install on Windows, use `requirements-minimal.txt`. The app will still use the bundled IPA dictionary fallback.

## 4. Install FFmpeg

Browser audio is recorded as `.webm`. The app tries to convert it to 16 kHz mono WAV using FFmpeg.

- Windows: install FFmpeg and add it to PATH.
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

Check:

```bash
ffmpeg -version
```

## 5. Test G2P only

```bash
python test_g2p.py
```

## 6. Run the app

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 7. Weighted PER formula

This version computes weighted PER as:

```text
Weighted PER = sum(error costs) / number of reference phonemes × 100
```

Where:

- correct phoneme = 0
- substitution = normalized PanPhon phoneme distance
- deletion = 1
- insertion = 1

This means close substitutions, such as similar vowels or similar consonants, receive less penalty than very different substitutions.

## 8. Research note for the report

The vectorized distance is based on PanPhon articulatory feature vectors:

> Mortensen, D. R., Littell, P., Bharadwaj, A., Goyal, K., Dyer, C., & Levin, L. (2016). PanPhon: A Resource for Mapping IPA Segments to Articulatory Feature Vectors. COLING 2016.

## 9. Local-only note

This app runs locally on your computer. It does not upload audio to an online server unless you modify it.
