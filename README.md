# Quality-Aware Self-Correcting Speech Translation on an Edge Device

Code for my BSc Computer Science dissertation (University of Surrey, May 2026). It is an offline English→Spanish speech-to-speech translation pipeline that runs entirely on a 4 GB NVIDIA Jetson Nano. A quality-estimation (QE) score decides when to re-translate a sentence.

**Speech → ASR (Whisper tiny) → MT (Opus-MT en→es) → QE gate (mBERT) → [self-correction] → TTS (eSpeak-ng)**

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Jetson Nano 4GB](https://img.shields.io/badge/Jetson%20Nano-4GB-76b900.svg)](https://developer.nvidia.com/embedded/jetson-nano)

---

## Overview

Most speech translation systems run in the cloud and say nothing about whether a given translation is any good. This project tests whether a QE score can work as an **active control signal** on a small edge device. When the score for a translation is low, the system corrects it on the device, with no retraining, no cloud connection and no human input.

How each sentence is processed:

1. **ASR.** Whisper tiny transcribes the English audio, with the language forced to `en`.
2. **MT pass 1.** Opus-MT en→es translates the transcript with greedy decoding (`num_beams=1`).
3. **QE gate.** mBERT embeds the source and the translation separately, and their cosine similarity is the QE score.
   - If QE ≥ τ, the first-pass translation is accepted.
   - If QE < τ, the sentence goes into the self-correction branch.
4. **MT pass 2 (conditional).** Opus-MT runs beam search and returns an N-best list. One of three methods then picks the final candidate:
   - **M1: QE-Rerank.** Picks the candidate with the highest mBERT QE score.
   - **M2: Minimum Bayes Risk (MBR).** Picks the candidate with the highest average ChrF against the other candidates. **This was the best method.**
   - **M3: Constrained Beam Search (CBS).** Boosts the logits of anchor words taken from the first-pass output.
5. **TTS.** eSpeak-ng speaks the final Spanish translation.

The three transformer models can't all be kept in the Nano's 4 GB of shared RAM at once. The evaluation and M2 demo scripts therefore use a **sequential load → run → unload** pattern: `cleanup()`, then `gc.collect()`, then `time.sleep(3)` at each model boundary.

---

## Key Results

All numbers come from the dissertation. MT metrics use 1,012 FLORES-200 `devtest` sentences on clean text. The internal greedy baseline is BLEU 26.09, ChrF 54.85, COMET 0.8485.

| Config (τ = 0.90) | BLEU | ΔBLEU | ChrF | ΔChrF | COMET | ΔCOMET |
|---|---|---|---|---|---|---|
| Greedy baseline | 26.09 | — | 54.85 | — | 0.8485 | — |
| M1 QE-Rerank, N=3 | 26.45 | +0.36 | 55.04 | +0.19 | 0.8494 | +0.0009 |
| M2 MBR, N=3 | 26.68 | +0.59 | 55.22 | +0.37 | **0.8505** | **+0.0020** |
| M2 MBR, N=5 | 26.56 | +0.47 | 55.17 | +0.32 | 0.8497 | +0.0012 |
| **M2 MBR, N=10** | **26.76** | **+0.67** | **55.35** | **+0.50** | 0.8496 | +0.0011 |
| M1 QE-Rerank, N=10 | 25.96 | −0.13 | 54.74 | −0.11 | 0.8481 | −0.0004 |
| M3 CBS, N=10 | 25.14 | −0.95 | 53.99 | −0.86 | 0.8424 | −0.0061 |

- **Significance.** I used a paired bootstrap (1,000 resamples, seed 42). All three M2 configurations at τ = 0.90 give significant BLEU and ChrF gains (p ≤ 0.003). For M2 at N=10, ΔBLEU = +0.67 with p < 0.001 and a 95% CI of [+0.349, +1.025]. M1 at N=10 is not significant (p = 0.746). M3 at N=10 is significantly *worse*.
- **Threshold sensitivity.** The QE scores mostly fall between 0.82 and 0.94. Correction triggers on 0.1% of sentences at τ = 0.75, 1.3% at 0.80, 22.7% at 0.85 and 86.2% at 0.90. Gains only show up at τ ≥ 0.85.
- **Correction efficiency.** Gain-to-Edit Ratio is GER = ΔCOMET / TER(original, corrected). It is highest for M2 at N=5 (+0.0081) and negative at N=10 (−0.0019).
- **QE validity.** The mBERT QE score correlates moderately with COMET (Pearson r = 0.41, p < 0.001, M2 at N=5).
- **Component baselines.** Whisper tiny gets a WER of 31.59% on FLEURS. Opus-MT gets BLEU 26.09 and ChrF 54.94 on clean FLORES-200 (`correct_baseline.py`).
- **CBS diagnostic (Phase 5).** M3's failure is not a tuning problem. The anchors come from the same first-pass output the system is trying to correct, so they carry its errors forward.

### Latency and resource use (live spoken input, Jetson Nano ARM CPU)

| Pipeline | ASR | MT pass 1 | QE | MT pass 2 (MBR) | TTS | **Total / sentence** |
|---|---|---|---|---|---|---|
| Baseline (greedy) | 16,424 ms | 12,581 ms | 1,970 ms | — | 17,183 ms | **48,157 ms** |
| M2 (τ = 0.90, N = 5) | 14,558 ms | 16,276 ms | 4,388 ms | 23,873 ms | 23,698 ms | **82,793 ms** |

These are averages over four live runs for each pipeline. Most of the M2 overhead comes from reloading Opus-MT between passes, which the 4 GB limit forces. Across the eight live runs, RAM use stayed between 2.45 and 3.33 GB, and the device temperature stayed between 31.5 and 33.0 °C.

All inference runs on the ARM Cortex-A57 CPU. The Python 3.8 / `transformers` stack used on JetPack 4 doesn't provide the CUDA support these models need, so the Maxwell GPU isn't used.

---

## Hardware and Software Requirements

- **Device:** NVIDIA Jetson Nano Developer Kit, 4 GB. I powered it from the DC barrel jack (J48 jumper fitted) and added a 40 mm fan.
- **OS:** JetPack 4.x (L4T R32, Ubuntu 18.04).
- **Python:** 3.8, installed alongside the system Python 3.6.
- **Audio:** a USB microphone, plus a USB audio adapter and speakers for TTS output, since the Nano has no onboard audio jack.
- **Storage:** about 5 GB free for models, datasets and results.
- **Network:** only needed to download packages and model weights. Everything runs offline after that. For the dissertation, weights and datasets were copied to the Nano over USB.

The Python dependencies are listed in [`requirements.txt`](requirements.txt).

---

## Setup

### 1. Clone into `~/disso`

The scripts use hard-coded paths under `~/disso` (or `/home/zubair/disso`) for imports, datasets and results. The easiest option is to clone the repo to that location:

```bash
git clone https://github.com/juebae/speech-translation_edge_device.git ~/disso
cd ~/disso
```

If your username isn't `zubair`, edit the `/home/zubair/disso/...` constants at the top of the `phase4_*`, `phase5_*` and `correct_baseline.py` scripts.

### 2. System packages

```bash
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-dev python3.8-venv \
    espeak-ng alsa-utils ffmpeg libsndfile1 portaudio19-dev libportaudio2
```

- `espeak-ng` provides TTS and `aplay` (from `alsa-utils`) plays the audio.
- Whisper needs `ffmpeg` to read audio files.
- `soundfile` needs `libsndfile1`, and `sounddevice` needs PortAudio.

### 3. Python environment

```bash
python3.8 -m venv ~/disso/.venv
source ~/disso/.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Model weights (download once, then run offline)

The modules load models from fixed local paths:

| Model | Expected location |
|---|---|
| Whisper tiny | `~/.cache/whisper/tiny.pt`. This is downloaded automatically by `whisper.load_model("tiny")`, or you can copy it in. |
| Opus-MT en→es | `~/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-es/snapshots/5bc4493d463cf000c1f0b50f8d56886a392ed4ab/` |
| mBERT (`bert-base-multilingual-cased`) | `~/.cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots/mbert_pytorch_final/` |
| Opus-MT copy for `correct_baseline.py` | `~/disso/models/opus_mt_original/` (a `save_pretrained` copy of `Helsinki-NLP/opus-mt-en-es`) |

For example, on a machine with internet access:

```python
from huggingface_hub import snapshot_download
snapshot_download("Helsinki-NLP/opus-mt-en-es", revision="5bc4493d463cf000c1f0b50f8d56886a392ed4ab")
p = snapshot_download("bert-base-multilingual-cased")  # then copy/rename this snapshot dir to .../snapshots/mbert_pytorch_final
```

Then copy `~/.cache/huggingface` and `~/.cache/whisper` to the Nano. You can also pass a different path with `OpusMT(model_snapshot=...)` or `QualityEstimation(model_snapshot=...)`.

### 5. Datasets

The repo has no data, and `datasets/`, `results/` and `*.json` are git-ignored. The evaluation scripts expect these JSON files:

| File | Format | Used by |
|---|---|---|
| `~/disso/datasets/flores_test/samples.json` | `[{"id", "source_en", "reference_es"}, ...]` (1,012 FLORES-200 devtest pairs) | `phase4_*`, `phase5_*`, `error_analysis_m2.py` |
| `~/disso/datasets/fleurs_test/samples.json` | `[{"id", "audio_array", "sampling_rate", "transcription"}, ...]` (FLEURS en_us clips, with `id` matching the FLORES ids) | `phase4_baseline.py` |
| `~/disso/models/opus_mt_original/flores_100.json` | `[{"en", "es"}, ...]` | `correct_baseline.py` |

---

## Usage

### Quick module checks

```bash
python3 -c "from phase3a_asr_module import WhisperASR; a=WhisperASR('tiny'); print(a.load())"
python3 -c "from phase3a_mt_module import OpusMT; m=OpusMT(); print(m.load()); print(m.translate('Hello, how are you?'))"
python3 -c "from phase3a_qe_module import QualityEstimation; q=QualityEstimation(); print(q.load()); print(q.score('Hello', 'Hola'))"
python3 -c "from phase3a_tts_module import EspeakNGTTS; t=EspeakNGTTS(language='es'); print(t.load()); t.synthesize_and_play('Hola mundo')"
```

### Live demo (the scripts behind the dissertation's live latency results)

```bash
python3 demo_baseline.py        # greedy baseline: ASR → MT → QE → TTS
python3 demo_m2_correction.py   # M2 self-correction (τ = 0.90, N = 5), sequential load/unload
```

Speak several sentences, then press **Enter** to stop recording. The audio is split into sentences at silences (1.0–1.2 s gaps, 0.8–1.0 s minimum segment), and each sentence goes through the pipeline. Per-sentence transcripts, translations, QE scores, per-stage latencies, RAM use and temperature are saved to `~/disso/results/demo_*_log.{csv,json}`. Appendix B of the dissertation lists the sentences used in the live runs.

### Real-time interactive pipeline (Phase 3B)

```bash
python3 phase3b_main_FIXED.py --list-devices
python3 phase3b_main_FIXED.py --num-recordings 5
```

This script loads all four models once at startup. It then loops: press Enter, speak, and recording stops after silence (energy VAD at RMS > 0.015 with a 5 s timeout, plus `noisereduce` spectral gating calibrated on 2 s of ambient noise). Session logs go to `~/disso/phase3b_sessions/`. The pipeline is English→Spanish only, because the MT model is fixed to `opus-mt-en-es`.

### Evaluation (Phases 4 and 5)

Run these in order. Each one writes to `~/disso/results/`:

```bash
python3 phase4_baseline.py     # ASR on FLEURS (WER) + greedy MT + QE  → baseline_results.json
python3 correct_baseline.py    # MT-only BLEU/ChrF on clean FLORES     → correct_baseline.json
python3 phase4_selfcorrect.py  # 4×3 grid: τ∈{0.75,0.80,0.85,0.90} × N∈{3,5,10}, M1/M2/M3
                               #   → grid_search_results.json, hypotheses_tau*_n*.json, colab_inputs.json
python3 phase4_best_config.py  # re-run τ=0.90, N=5 and save B1/M1/M2/M3 hypotheses → best_config_hypotheses.json
python3 phase5_eval_cbs.py     # CBS diagnostic (boost sweep, strict anchors, CBS+MBR hybrid)
python3 phase5_cbs_expA.py     # Exp A: boost sweep 2.0 / 1.5 / 1.0
python3 phase5_cbs_expB.py     # Exp B: strict anchors (capitalised words / digits), boost 1.5
python3 phase5_cbs_expC.py     # Exp C: CBS candidates merged into the MBR pool
python3 error_analysis_m2.py   # per-sentence M2 error analysis (τ=0.90, N=10)
```

The full 1,012-sentence grid search took about 24 hours on the Nano. Two runs on consecutive days gave identical results.

COMET, GER and the bootstrap significance tests were **computed off-device on Google Colab** from `colab_inputs.json` and `best_config_hypotheses.json`. Those notebooks are not in this repo, and the COMET libraries are not in `requirements.txt`.

---

## Project Structure

```
.
├── phase3a_asr_module.py     # WhisperASR: Whisper tiny via openai-whisper (language forced to "en")
├── phase3a_mt_module.py      # OpusMT: translate(), translate_nbest(), translate_nbest_constrained() (CBS logit boost)
├── phase3a_qe_module.py      # QualityEstimation: mBERT embedding cosine similarity
├── phase3a_tts_module.py     # EspeakNGTTS: espeak-ng subprocess + aplay playback
├── phase3b_audio_module.py   # MicrophoneAudioCapture: sounddevice @16 kHz, energy VAD, noisereduce
├── phase3b_main_FIXED.py     # Real-time interactive orchestrator (all models loaded once)
├── demo_baseline.py          # Live demo: greedy baseline with RAM/temperature logging
├── demo_m2_correction.py     # Live demo: M2 MBR self-correction, sequential load/unload
├── phase4_baseline.py        # Baseline eval: FLEURS WER + greedy MT + QE (sequential loading)
├── correct_baseline.py       # MT-only BLEU/ChrF reference on clean FLORES text
├── phase4_selfcorrect.py     # Main grid search (τ × N × {M1, M2, M3})
├── phase4_best_config.py     # Saves per-sentence hypotheses for COMET/GER scoring
├── phase5_eval_cbs.py        # CBS diagnostic study (all experiments)
├── phase5_cbs_exp{A,B,C}.py  # CBS experiments A/B/C as standalone scripts
├── error_analysis_m2.py      # Per-sentence analysis of M2 corrections
├── config/                   # Earlier version of the audio module (no noise reduction)
├── Docker.cuda38             # Abandoned CUDA/faster-whisper container experiment (not used)
├── requirements.txt
└── README.md
```

---

## Implementation Notes

- **QE score.** `phase3a_qe_module.py` mean-pools mBERT's final hidden states for each sentence, computes their cosine similarity, and rescales it to [0, 1] as `(sim + 1) / 2`. COMET-Kiwi would be a stronger QE model, but its XLM-R backbone doesn't fit in 4 GB alongside Opus-MT. TinyBERT gave unstable scores.
- **MBR (M2).** `mbr_select()` scores each N-best candidate by its mean `corpus_chrf` against the other candidates and picks the highest. It doesn't call mBERT again, so it costs less than M1.
- **sacrebleu ChrF scale.** The scripts use **sacrebleu 1.x**, whose `corpus_chrf().score` is in [0, 1]. They multiply by 100 to report ChrF on the usual 0–100 scale. With sacrebleu ≥ 2.0 you would need to remove that `* 100`.
- **Memory.** Without `gc.collect()` and the 3 s sleep between model loads, loading Opus-MT right after unloading mBERT failed with out-of-memory errors. Keep them.
- **Threads.** `phase3b_main_FIXED.py` sets `OMP_NUM_THREADS=1` (and the related variables) so that NumPy and PyTorch don't oversubscribe the Nano's four cores.
- **Temp WAVs.** Recordings are written to uniquely timestamped temporary WAV files before ASR, to avoid a race condition between recordings.

---

## Limitations

- **One language pair.** Only English→Spanish was tested, on a single Jetson Nano. The architecture itself is language-agnostic.
- **Weak QE signal.** mBERT cosine similarity is only a proxy for quality (r = 0.41 with COMET), so many triggered sentences don't benefit from correction.
- **Threshold- and method-dependent gains.** Improvements only appear at τ ≥ 0.85 and only with MBR. CBS consistently makes translations worse.
- **Latency.** Running on the CPU and reloading models between passes makes the pipeline far from real-time (about 48 s per sentence for the baseline and 83 s for M2).
- **ASR errors.** Whisper tiny's WER is about 32%, and its errors carry through to the translation.
- **Domain.** FLORES-200 is Wikipedia text with professional translations, which is relatively easy for Opus-MT.

The dissertation's future work includes 8 GB hardware (such as the Jetson Orin Nano) to keep models resident, INT8 or KV-cache quantisation, a quantised COMET-Kiwi gate, QE integration at the encoder level, CBS with external anchors, and more language pairs.

---

## Troubleshooting

- **No GPU acceleration.** This is expected. All models run on the CPU.
- **Whisper error about `ffmpeg`.** Install it with `sudo apt-get install ffmpeg`.
- **Model "not found" or snapshot errors.** Check that the weights are in the exact paths listed under [Model weights](#4-model-weights-download-once-then-run-offline).
- **Out-of-memory errors while loading models.** Keep the `cleanup()`, `gc.collect()` and `time.sleep(3)` sequence. Close other processes, and consider adding swap.
- **Too many or too few sentences detected in the demos.** Adjust `SILENCE_THRESH`, `SILENCE_GAP_SEC` and `MIN_SENTENCE_SEC` at the top of the demo scripts.
- **No TTS audio.** Check that `espeak-ng --version` and `aplay -l` both work and that the USB audio device is the default output.

---

## Citation

```bibtex
@thesis{farooq2026speechtranslation,
  title       = {Quality-Aware Self-Correcting Speech Translation on an Edge Device},
  author      = {Farooq, Zubair Ajmal},
  type        = {BSc dissertation},
  institution = {Department of Computer Science, University of Surrey},
  year        = {2026},
  month       = may
}
```

---

## License

Academic research project. Please contact the author before reusing it.

Third-party models and datasets (Whisper, Opus-MT, mBERT, FLORES-200, FLEURS) are used under their own licences. eSpeak-ng (GPLv3) is called as an external binary and is not distributed with this repo.

---

## Acknowledgments

- Supervisor: Dr Diptesh Kanojia, University of Surrey
- Whisper (OpenAI), Opus-MT (Helsinki-NLP), multilingual BERT (Google), Hugging Face Transformers, eSpeak-ng, noisereduce, sacrebleu
- NVIDIA JetPack and the Jetson platform

## Contact

- **Author:** Zubair Ajmal Farooq
- **Institution:** University of Surrey
- **GitHub:** [@juebae](https://github.com/juebae)
