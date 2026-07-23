# Speech Translation Pipeline on Edge Device

Real-time speech translation system running entirely on NVIDIA Jetson Nano (4GB)

**English speech → ASR (Whisper tiny) → MT (Opus-MT) → QE (mBERT) → TTS (espeak-ng)**

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![JetPack 4.6](https://img.shields.io/badge/JetPack-4.6-green.svg)](https://developer.nvidia.com/embedded/jetpack)

---

## Overview

Privacy-preserving edge translation pipeline designed for resource-constrained devices. All processing happens on-device without cloud dependencies. The pipeline uses a sequential load-and-unload architecture to fit within the Jetson Nano's 4GB shared RAM, since all three transformer models cannot be held in memory simultaneously.

### Key Features

- **On-Device Processing:** No internet required, full privacy
- **Integrated Quality Estimation:** Real-time translation quality scoring using mBERT, used as an active self-correction trigger rather than a passive diagnostic
- **Noise Reduction:** Adaptive filtering for noisy environments (scipy + noisereduce)
- **Modular Architecture:** Easy to swap ASR/MT/QE/TTS components
- **Memory Optimized:** Runs on 4GB RAM via sequential model loading and unloading
- **Voice Activity Detection:** Automatic speech detection with silence-based segmentation

### Performance

Two operating modes are supported, with different latency profiles:

- **Real-time mode (all models resident):** ~1.8s net inference per utterance
- **Batch/self-correction mode (sequential load-unload):** ~35-83s per sentence, depending on whether correction is triggered — this cost is dominated by mandatory model reload between stages, required by the 4GB memory ceiling
- **Peak memory usage:** ~2.7-3.3 GB across live demo runs
- **Hardware:** NVIDIA Jetson Nano 4GB, 128-core Maxwell GPU (inference runs on ARM CPU; CUDA is not supported for these models under the JetPack 4 software stack)

---

## Hardware Requirements

- **Device:** NVIDIA Jetson Nano 4GB (or higher)
- **OS:** Ubuntu 18.04 (JetPack 4.6)
- **Python:** 3.8
- **Storage:** ~5GB free space (models + code)
- **Microphone:** USB or 3.5mm input

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/juebae/concurrent_translation.git
cd concurrent_translation
```

### 2. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y espeak-ng portaudio19-dev libsndfile1
```

### 3. Install Python Packages

```bash
pip3 install -r requirements.txt
```

### 4. Install Whisper ASR

```bash
pip3 install openai-whisper
```

The tiny model (~160MB) is used for its minimal footprint, making it deployable alongside Opus-MT and mBERT within the Jetson Nano's 4GB RAM budget. The model downloads automatically on first use via `whisper.load_model("tiny")`.

### 5. Test Installation

```bash
# Test full pipeline (1 recording)
python3 phase3b_main_FIXED.py --num-recordings 1
```

---

## Usage

### Interactive Mode (Continuous Translation)

```bash
python3 phase3b_main_FIXED.py
```

Press Enter to start recording, speak naturally, system auto-detects silence.

### Batch Mode (Fixed Number of Recordings)

```bash
python3 phase3b_main_FIXED.py --num-recordings 10
```

### Change Target Language

```bash
python3 phase3b_main_FIXED.py --target-lang fr  # Spanish by default
```

### List Audio Devices

```bash
python3 phase3b_main_FIXED.py --list-devices
```

---

## Architecture
```
┌─────────────┐
│  Microphone │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Noise Reduction    │  (scipy + noisereduce)
│  Voice Activity Det │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   ASR               │  English transcription
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  MT                 │  Translation to target lang
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  QE                 │  Quality scoring (0-1)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  TTS                │  Audio synthesis & playback
└─────────────────────┘
```

---

## Project Structure

```
concurrent_translation/
├── phase3a_asr_module.py # ASR component (Whisper tiny)
├── phase3a_mt_module.py # Machine Translation (Opus-MT)
├── phase3a_qe_module.py # Quality Estimation (mBERT)
├── phase3a_tts_module.py # Text-to-Speech (espeak-ng)
├── phase3b_audio_module.py # Microphone capture + VAD + noise reduction
├── phase3b_main_FIXED.py # Main pipeline orchestrator (real-time, single-load)
├── phase4_baseline.py # Batch evaluation: greedy baseline
├── phase4_selfcorrect.py # Batch evaluation: grid search over τ, N, methods
├── phase5_eval_cbs.py # Constrained Beam Search diagnostic study
├── correct_baseline.py # MT-only BLEU/ChrF baseline on clean text
├── requirements.txt # Python dependencies
├── .gitignore # Git exclusions
└── README.md # This file
```

---

## Technical Details

### Component Specifications

| Component | Model/Library         | Size    | Device | Purpose             |
| --------- | ---------------------- | ------- | ------ | -------------------- |
| **ASR**   | Whisper tiny            | ~160 MB | CPU    | Speech recognition  |
| **MT**    | Opus-MT-en-es           | ~180 MB | CPU    | Neural translation  |
| **QE**    | mBERT base              | ~680 MB | CPU    | Translation quality |
| **TTS**   | espeak-ng               | <10 MB  | CPU    | Speech synthesis    |

*All inference runs on ARM CPU. The Jetson Nano's Maxwell GPU cannot be used for these models under the JetPack 4 / Python 3.8 software stack, which lacks CUDA kernel support for the transformer libraries used here.*

### Why Whisper Tiny?

Whisper tiny was selected for ASR based on three criteria: multilingual capability, robustness to acoustic variation, and memory footprint. At ~160MB, it's the smallest model in the Whisper family that maintains reasonable word error rates on general-domain speech, making it deployable alongside the MT and QE components within the Jetson Nano's 4GB RAM.

### Noise Reduction Strategy

1. **Calibration:** 2-second ambient noise profile capture at startup
2. **Filtering:** noisereduce library with scipy signal processing
3. **Result:** Improved ASR accuracy in real-world noisy environments

### Memory Management

- **Sequential Loading:** Models loaded one at a time to fit in 4GB; each component's memory is released via cleanup() and gc.collect() before the next component loads
- **CPU-only inference:** All models run on ARM CPU due to software stack constraints
- **Peak Usage:** ~2.7-3.3 GB across live demo runs, well within the 4GB ceiling

---

## Performance Benchmarks

Figures below are from the real-time, single-load pipeline (all models resident in memory), profiled on Jetson Nano ARM CPU:

| Metric | Value | Notes |
|--------|-------|-------|
| ASR Latency | ~3.2s | Whisper tiny inference |
| MT Latency (greedy) | ~14.4s | Per sentence, first pass |
| QE Latency | ~16.1s | mBERT cosine similarity |
| TTS Latency | ~2.0s | Per sentence |

*For the batch self-correction evaluation (sequential load/unload architecture used in the formal grid search), total pipeline latency is substantially higher due to mandatory model reload costs — see the dissertation for full figures (baseline ~48s, self-correction ~83s per sentence).*

---

## Testing

### Unit Tests

```bash
python3 -c "from phase3a_asr_module import WhisperASR; print('ASR OK')"
python3 -c "from phase3a_mt_module import OpusMT; print('MT OK')"
```

### Integration Test

```bash
# Single end-to-end test
python3 phase3b_main_FIXED.py --num-recordings 1
```

---

## Dissertation Context

This project was submitted as a BSc final year dissertation:

> **"Quality-Aware Self-Correcting Speech Translation on an Edge Device"**

### Research Questions

1. Can quality estimation function as an active control signal to trigger self-correction in an edge-deployed speech translation pipeline?
2. Can integrated QE improve translation quality without major memory or architectural overhead?
3. Is quality-aware self-correction feasible on 4GB edge devices?

### Status

Completed. Full evaluation conducted over 1,012 FLORES-200 sentences on Jetson Nano hardware, with statistically significant BLEU and ChrF improvements achieved via QE-triggered Minimum Bayes Risk decoding (best configuration: τ=0.90, N=10, BLEU +0.67, ChrF +0.50, p<0.001).

---

## Known Issues & Limitations

- **Single language pair:** Currently English → Spanish only (architecture is language-agnostic and extensible)
- **CPU-only inference:** No CUDA support under the current JetPack 4 / Python 3.8 stack, limiting inference speed
- **Sequential load architecture:** Required by the 4GB memory ceiling; introduces model reload overhead in the batch self-correction pipeline
- **QE signal:** mBERT cosine similarity is a proxy for quality (moderate correlation with COMET, r=0.41); a stronger reference-free QE model (e.g. COMET-Kiwi) was not usable due to its larger memory footprint

---

## Troubleshooting

### "CUDA out of memory" / no GPU acceleration

This is expected — inference runs on ARM CPU only, since the JetPack 4 software stack doesn't support CUDA kernels for these models. There is no GPU fallback needed since GPU is never used.

### Poor ASR quality

- Check microphone input: `python3 phase3b_main_FIXED.py --list-devices`
- Increase noise reduction sensitivity in `phase3b_audio_module.py`

### Out-of-memory errors during model loading

Ensure `gc.collect()` and the cooldown sleep are not removed from the sequential loading logic — these are required to reclaim memory between model loads on the Jetson Nano.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@bachelorsthesis{farooq2026speechtranslation,
  title={Quality-Aware Self-Correcting Speech Translation on an Edge Device},
  author={Zubair Ajmal Farooq},
  year={2026},
  school={University of Surrey}
}
```

---

## License

Academic research project. Contact for usage permissions.

---

## Acknowledgments

- **Whisper:** ASR by OpenAI
- **Opus-MT:** Neural MT by Helsinki-NLP
- **Transformers:** Hugging Face library
- **NVIDIA:** JetPack SDK & Jetson hardware

---

## Contact

- **Author:** Zubair Ajmal Farooq
- **Institution:** University of Surrey
- **GitHub:** [@juebae](https://github.com/juebae)
