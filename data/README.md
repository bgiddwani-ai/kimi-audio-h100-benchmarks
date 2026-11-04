# Fixed AQA inputs

Source: [lmms-lab-audio/ClothoAQA](https://huggingface.co/datasets/lmms-lab-audio/ClothoAQA),
revision `57b51552af2fd57a033f515b4e9a4496005dd3c3`.

- 32 unique clips: `clotho_aqa` / `clotho_aqa_test_filtered`.
- 32 unique clips: `clotho_asqa_test_v2` / `test`.
- Dataset shuffle seed: `20261104`; duplicates skipped by original audio SHA-256.
- Audio converted to mono 16 kHz PCM16 WAV with `scipy.signal.resample_poly`.
- `manifest.jsonl` records the question, reference, duration, split, source hash,
  and normalized WAV hash. Only file paths were made relative for this repo.

The benchmark verifies every WAV hash before timing. Source audio and QA
attribution/licensing are documented in the linked upstream dataset.
