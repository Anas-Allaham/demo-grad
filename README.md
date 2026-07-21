# Pronunciation Feedback System — Local Flask Version

Text → G2P → expected phonemes; audio → Wav2Vec2 → spoken phonemes; the two
are aligned and scored into a **provisional** pronunciation score and an
adaptive, confusion-aware per-phoneme practice loop.

> ### ⚠️ Scientific honesty note
> The scores here are **provisional**. They are built from PanPhon
> articulatory *distance* — a description of how similar two phonemes are —
> **not** from a calibrated acoustic Goodness-of-Pronunciation (GOP) model.
> Nothing in this app is a CEFR level or a research-grade pronunciation score.
> Every level/score is labelled provisional in the API and UI, and the level
> thresholds are configurable placeholders.

---

## Audit revisions (v2)

This version hardens the evidence pipeline:

1. **Speech-presence audio gate** — energy is no longer enough. Steady or
   amplitude-modulated noise/tones, a DC offset, silence, and clipping are rejected using spectral
   flatness, spectral bandwidth, zero-crossing rate, and syllable-rate envelope
   modulation. Validated against a real speech fixture (`tests/fixtures/`).
2. **Scoring provenance & trust** — every attempt stores `scoring_engine`,
   `scoring_trusted`, and `mastery_updated`; events store their engine too.
   Assessment coverage, prompt counts, confusion pairs, and diagnostic progress
   count **trusted, mastery-updating recordings only**. Legacy rows are treated
   as untrusted/unknown unless explicitly migrated.
3. **PanPhon validated at startup** — PanPhon must vectorize *every* assessable
   phoneme or the engine reports `fallback_features` (untrusted). It never
   mixes PanPhon and fallback distances within an attempt.
   **Trusted scoring requires PanPhon** (`pip install panphon`).
4. **Audio quality = fractional evidence** — `quality_weight` scales the Beta
   update symmetrically (less certain), never converting low quality into a
   pronunciation failure.
5. **Trusted contextual G2P** — all 72 bundled heteronyms are loaded before
   either backend. Resolution works without NeMo, and unresolved/OOV/unsupported
   references are exposed and never update mastery. `permit` is explicitly
   unsupported because its contrast collapses in the acoustic inventory.
6. **Real credible interval** — the macro-average now has a genuine Bayesian
   posterior **credible interval** via deterministic Monte-Carlo sampling of the
   per-phoneme Beta posteriors (renamed from the old fake "confidence interval").
7. **Functional exercise types** — mastery drives the actual material: minimal
   pairs / isolated words (low), short phrases, targeted sentences, or
   maintenance/connected speech. Retrieval is confusion-aware, not only the LLM.
8. **Insertions tracked separately** — epenthesis updates an utterance-level
   Beta state and affects the profile, but is never attributed to an expected phoneme.
9. **Private, temporary audio** — original, converted, and reduced recordings are
   deleted in an outer post-save `finally` unless `RETAIN_AUDIO=1`. No recordings ship.
10. **Atomic writes** — attempt + events + mastery + assignment completion happen
   in one SQLite transaction; a partial failure rolls back entirely.

---

## Pipeline

```text
audio quality gate
   → canonical phoneme tokenization
   → phoneme alignment (Needleman–Wunsch)
   → per-occurrence soft scores
   → per-recording phoneme mastery update (Beta posterior)
   → evidence-aware user pronunciation level
   → adaptive / confusion-aware exercises
```

An **unscorable** recording (silence, clipping, too short, internal dropout,
…) is rejected before any scoring and never updates mastery.

If **PanPhon is not installed**, the app falls back to a small, clearly
labelled articulatory-class distance so it keeps running, but it reports
`scoring_engine == "fallback_features"` and **refuses to update trusted
mastery** from those numbers.

---

## Architecture (one authoritative implementation per concern)

| Module | Responsibility |
|---|---|
| `phoneme_vectors_professional.py` | **Single source of truth**: canonicalization, the canonical inventory (derived from the model `vocab.json`), articulatory distance (PanPhon or labelled fallback), vowel/consonant classification, alignment cost + substitution labels, soft mastery evidence, inventory validation, `scoring_engine()`. |
| `phoneme_vectors.py` | Thin compatibility wrapper re-exporting the professional module (no second implementation). |
| `tokenization.py` | `normalize_ipa`, `split_ipa_word`, `tokenize_reference_ipa`, `tokenize_ctc_prediction`, `ipa_to_tokens`, reading guide. |
| `g2p_service.py` | Text → IPA plus reference trust metadata; heteronyms resolve before NeMo/dictionary fallback. No torch. |
| `scoring.py` | DP alignment + metrics. Keeps `articulatory_distance`, `alignment_cost`, and score strictly separate. |
| `mastery.py` | Soft, **per-recording** Beta posterior with correct half-life decay. |
| `audio_quality.py` | The scorability gate (`AudioQualityDecision`) + `should_update_mastery`. |
| `assessment.py` | Evidence-aware level, confusion aggregation, exercise-type-by-mastery, diagnostic coverage. |
| `services.py` | Ties assessment → exercise selection; shared by `/exercise` and `/practice/next`. |
| `content.py` | Retrieval scoring + LLM generation-with-verification. |
| `db.py` | SQLite persistence + **non-destructive** migrations. |
| `app.py` | Flask routes + the Wav2Vec2 acoustic step. |

---

## Canonical phoneme inventory

The canonical scoring inventory is derived from
`model/my_wav2vec2_phoneme_model/vocab.json` (the phonemes the acoustic model
can actually emit) plus application-level composite phonemes. The model vocab
has **no length symbol `ː`**, so expected long vowels are aliased onto emittable
units:

```
g → ɡ    r → ɹ    ɚ → ɝ    ɜː → ɝ
ɑː → ɑ   ɔː → ɔ   iː → i    uː → u
```

Composite (application-level) phonemes, kept as one scoring unit:

```
t+ʃ → tʃ   d+ʒ → dʒ
a+ɪ → aɪ   a+ʊ → aʊ   e+ɪ → eɪ   o+ʊ → oʊ   ɔ+ɪ → ɔɪ
```

`validate_g2p_inventory()` checks at startup (and on `/health`) that every
tagged phoneme maps into this inventory; unsupported phonemes never silently
enter scoring.

---

## Scoring formulas

### Articulatory distance (provisional similarity, **not** a score)
`articulatory_distance(a, b) ∈ [0, 1]` — normalized weighted-L1 distance over
PanPhon feature vectors (`0` = identical, `1` = maximally different). Composite
phonemes use the mean of their component-segment vectors.

### Alignment cost (used **only** by the DP alignment — not a distance)
```
correct                       0.00
very-close substitution       0.25   (distance ≤ 0.15)
close substitution            0.45   (distance ≤ 0.35)
medium substitution           0.75   (distance ≤ 0.65)
major substitution            1.05
vowel/consonant substitution  1.35   (major-class guard)
unknown substitution          1.20
deletion / insertion          0.85
```

### Metrics (insertions are preserved, not silently clamped)
```
raw_weighted_per         = Σ(alignment_cost of non-correct rows) / (#reference phonemes) × 100   # may exceed 100
display_error_percent    = clamp(raw_weighted_per, 0, 100)
display_accuracy_percent = clamp(100 − raw_weighted_per, 0, 100)
phoneme_error_rate       = display_error_percent   # backward-compatible field
```

### Soft mastery evidence (provisional, isolated for future GOP swap)
```
correct                       → 1.0
deletion                      → 0.0
unknown substitution          → 0.0
vowel/consonant substitution  → 0.0
other substitution            → clamp(1 − articulatory_distance, 0, 1)
```

### Per-phoneme mastery (Beta posterior, **one update per recording**)
For each recording, group alignment rows by expected phoneme, then:
```
mean_obs      = mean(soft evidence for that phoneme in this recording)
α  ← α_decayed + quality_weight × mean_obs
β  ← β_decayed + quality_weight × (1 − mean_obs)
occurrence_count    += (number of occurrences this recording)
independent_attempts += 1               # exactly once per recording
```
So three `/θ/` tokens in one sentence → `occurrence_count += 3`,
`independent_attempts += 1` (never three independent attempts, never instant
"mastered").

### Time decay (correct half-life)
```
gamma       = 0.5 ** (elapsed_days / half_life_days)      # 50% at one half-life
α_decayed   = 1 + gamma·(α − 1)
β_decayed   = 1 + gamma·(β − 1)
```
Decay is applied **on read** (ranking, display, level assessment, mastered
checks), so stale skills decay in real time.

### Evidence-aware level (provisional; **not** CEFR)
A phoneme is *level-eligible* only after **≥ 3 independent, trusted recordings**,
**≥ 2.0 quality-weighted effective recordings**, and **≥ 2 distinct prompts**.
Only scorable recordings with trusted PanPhon scoring and a trusted reference
G2P count. Insertions form a separate utterance-level posterior:
```
epenthesis_obs       = max(0, 1 − insertion_count / reference_unit_count)
α_utt               = 1 + Σ quality_weight × epenthesis_obs
β_utt               = 1 + Σ quality_weight × (1 − epenthesis_obs)
utterance_weight     = 0.20 × min(1, effective_recordings / 3)
profile_draw         = (1 − utterance_weight) × phoneme_macro_draw
                       + utterance_weight × Beta(α_utt, β_utt) draw
pronunciation_score  = 100 × E[profile_draw]
credible_interval    = 100 × [q2.5, q97.5] of profile_draw
overall_level        = one band only when the whole interval is inside it;
                       otherwise "uncertain" with borderline_levels
assessment_status    = insufficient_evidence | provisional | established
```
Too little coverage → `insufficient_evidence` and `overall_level = "unknown"`
(the app never invents a level). The Monte-Carlo is deterministic (fixed seed).

---

## Setup

```bash
conda create -n pronunciation-app python=3.10 -y
conda activate pronunciation-app

pip install -r requirements-minimal.txt        # demo install
# or full (optional NeMo/spaCy backend improvements):
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

1. **Model weights**: put your trained Wav2Vec2 files in
   `model/my_wav2vec2_phoneme_model/` (`config.json`, `vocab.json`,
   `tokenizer_config.json`, `preprocessor_config.json`, `model.safetensors`).
   The large `model.safetensors` is gitignored — copy it in.
2. **FFmpeg** (for browser `.webm`/`.ogg`/mobile audio -> 16 kHz mono WAV):
   the app can use the `imageio-ffmpeg` Python package from the requirements.
   If decoding still fails on Windows/Conda, install the system binary with:
   `conda install -c conda-forge ffmpeg -y`, then restart `python app.py`.
3. **Optional LLM generation**: copy `.env.example` → `.env` and add
   `GEMINI_API_KEY`. Without a key the app runs retrieval-only. **Never commit `.env`.**

---

## Build the exercise bank

The builder tags `data/seed_sentences.txt` with the app's own G2P pipeline,
re-canonicalizes any pre-existing bank sentences, and reports coverage:

```bash
python scripts/build_exercise_bank.py
```

It prints: inserted / duplicate / rejected (OOV) counts, difficulty
distribution, per-canonical-phoneme coverage, and zero/low-coverage phonemes.

---

## Run

```bash
python app.py
# http://127.0.0.1:5000
```

`GET /health` reports real readiness: model config + weight files, processor/
model loadability (`?deep=1` forces a full load), PanPhon availability, scoring
engine, G2P availability, exercise-bank population, and canonical-inventory
validation.

---

## Tests

Unit tests do **not** require the Wav2Vec2 weights or PanPhon — the acoustic/
model layer is bypassed and distances use deterministic inputs.

```bash
python -m compileall .
pytest -q
```

Covered: multiword-CTC & formatted-reference tokenization; long-vowel & composite
mapping; distance/cost/score separation; soft evidence; deletions; per-recording
attempt counting; exact-50%-at-half-life decay; stale-decay ranking;
insufficient-evidence level gating; continuing diagnostic coverage;
non-scorable-audio mastery gate; exercise repetition/difficulty selection;
seed-file location; single-canonicalizer identity; PanPhon-unavailable safety.

---

## Database migrations (non-destructive)

`db.init_db()` runs additive migrations via `PRAGMA table_info` + `ALTER TABLE`:
- `attempt_phoneme_events`: `articulatory_distance`, `alignment_cost` (the old
  single `distance` column is kept for compatibility).
- `phoneme_skill_state`: `occurrence_count` (separate from `attempts_count`,
  which now means **independent recordings**).
- `attempts`: `raw_weighted_per`, `quality_weight`, `scorable`, `rejected_reason`.

Plus a one-time, idempotent, **evidence-preserving** cleanup that folds legacy
non-canonical mastery keys (e.g. `r`, `iː`, `ɔr`) into their canonical form,
merging Beta evidence. No user data is dropped.

---

## Remaining scientific limitations

- The score is **provisional articulatory-distance evidence, not GOP**. PanPhon
  distance describes phoneme *similarity*, not acoustic pronunciation quality.
- The fallback distance (when PanPhon is absent) is coarser still and is never
  used to update trusted mastery.
- The difficulty proxy (`word_count + avg_word_len/2`) is a readability
  approximation, not a validated CEFR classifier.
- Level thresholds are configurable placeholders; they are **not** calibrated or
  validated against human raters.
- Confidence intervals come from the Beta posterior, not from acoustic
  measurement uncertainty.

## Research note

Distance is based on PanPhon articulatory feature vectors:

> Mortensen, D. R., Littell, P., Bharadwaj, A., Goyal, K., Dyer, C., & Levin, L.
> (2016). *PanPhon: A Resource for Mapping IPA Segments to Articulatory Feature
> Vectors.* COLING 2016.

This app runs locally and does not upload audio to any online server unless you
modify it.
