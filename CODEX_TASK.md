You are a senior Python, speech-processing, ASR, and adaptive-learning engineer.

The attached ZIP is the ONLY and latest project version. Ignore all previous
versions or reports. Extract it, inspect the real code independently, reproduce
the issues below, implement the fixes, run tests, and deliver a corrected ZIP.

Project goal:
Build a reliable English-pronunciation pipeline:

audio validation
→ context-aware reference G2P
→ CTC phoneme tokenization
→ phoneme alignment
→ soft per-recording mastery
→ evidence-aware user level
→ adaptive/confusion-aware exercises

Preserve:
- Flask UI and existing API compatibility
- SQLite user data and non-destructive migrations
- Wav2Vec2 integration
- PanPhon integration
- the current modular architecture
- existing exercise-bank and heteronym files

Do not include:
- .env or secrets
- app.db containing user data
- uploaded recordings
- model weights
- caches or temporary files

Known issues that must be independently verified and fixed:

1. Heteronym G2P
The minimal DictionaryIpaG2p fallback currently ignores heteronyms.json.
This causes incorrect reference IPA, for example:
- “I read every day” vs “I read it yesterday”
- “They record music” vs “The record was broken”
- “Please present the report” vs “The present is here”

Requirements:
- Decouple heteronym resolution from the NeMo import.
- Resolve heteronyms before dictionary fallback.
- Make context-aware heteronym handling available without requiring NeMo.
- Expose g2p_mode, heteronym_resolution_active, and reference_g2p_trusted.
- Do not update mastery when the reference pronunciation is unresolved or
  untrusted.
- Validate all 72 heteronym entries.
- Add context tests for read, record, present, use, refuse, close, and permit.
- “permit” currently collapses because ɚ→ɝ and lexical stress is stripped.
  Either support the distinction or explicitly report it as unsupported.

2. Audio-quality gate
The current gate accepts amplitude-modulated noise and tones because spectral
checks are only applied when envelope modulation is low. “clipping” is also
not fatal.

Requirements:
- Reject clipping above MAX_CLIPPING_RATIO.
- Reject amplitude-modulated white noise and modulated sine tones.
- Apply spectral/tonality checks consistently.
- Prefer a real VAD/speech-probability layer if feasible.
- Add tests for:
  silence, DC, steady noise, modulated noise, steady tone, modulated tone,
  clipped speech, real speech, real speech with DC offset, and mains hum.
- Rejected recordings must create no alignment/mastery evidence.

3. Audio cleanup
The uploaded file may remain when g2p_convert(), FFmpeg, process_recording(),
model loading, scoring, or DB writing raises before cleanup_paths is created.

Requirements:
- Initialize cleanup paths immediately after saving.
- Wrap the complete post-save path in an outer try/finally.
- Add failure-injection tests for all important failure points.
- RETAIN_AUDIO=false must leave no new files after every request outcome.

4. Assessment consistency
- /practice/next is evidence-aware, but /exercise?user=NAME currently falls
  back to a stateless mean-based assessment.
- Use assess_profile() for every saved-profile path.
- Keep raw stateless metrics clearly separate and provisional.
- Make systematic insertions/epenthesis influence the overall learner profile
  through an utterance-level state, without attaching insertions to a reference
  phoneme.
- Use posterior probability or a credible lower bound for level decisions.
- Return uncertain/borderline when an interval crosses a level threshold.
- Consider quality-weighted effective evidence, not only recording count.
- Keep the level explicitly provisional and not equivalent to CEFR.

5. Scoring provenance
If PanPhon inventory validation fails, do not mix PanPhon and fallback
distances while reporting one global fallback engine.
Use one engine consistently per attempt or store exact per-row provenance.
Fallback results must never update trusted mastery.

6. Exercise generation
Verify that:
- low mastery produces actual minimal-pair/isolated-word practice;
- known confusions affect retrieval, not only metadata;
- unknown phonemes continue receiving diagnostic coverage;
- difficulty and sentence length depend on evidence-aware level;
- repeated exercises are avoided;
- generated exercises are verified using the same trusted G2P path.

7. Packaging verification
The current TEST_OUTPUT says 62 tests, but the current source contains 64 test
functions. MANIFEST.txt also contains stale hashes.

Requirements:
- Run compileall and the complete pytest suite on the exact final source.
- Add the missing heteronym, adversarial-audio, cleanup-failure, and assessment
  consistency tests.
- Regenerate TEST_OUTPUT.txt after the final code changes.
- Generate MANIFEST.txt from the exact archived bytes after line-ending
  normalization.
- Validate every manifest hash after creating the final ZIP.

Work autonomously:
1. Inspect and reproduce first.
2. Implement the fixes.
3. Do not weaken tests just to make them pass.
4. Preserve unrelated working behavior.
5. Report any dependency/environment limitation honestly.
6. Do not claim calibrated GOP or CEFR accuracy without human-labelled data.

Final response must include:
- root causes found;
- changed files;
- final scoring/mastery/level formulas;
- heteronym context test table;
- adversarial audio results;
- exact compile/test output and number of tests;
- manifest validation result;
- remaining scientific limitations;
- link/path to the corrected ZIP.

Explain the final report to me in Arabic.