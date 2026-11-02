---
title: VoiceTutor
emoji: 🎧
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Voice-first study assistant that answers from your notes
---

# VoiceTutor

A voice-first study assistant for visually impaired and dyslexic students. Upload your notes (PDF, DOCX or TXT), ask a question out loud, and hear an answer grounded in your notes, with page numbers ("on page 4").

- **ASR:** Whisper on Groq (`whisper-large-v3-turbo`), with local **faster-whisper** as an automatic fallback.
- **LLM:** Llama on Groq (`llama-3.3-70b-versatile`). With no key, or when Groq fails, it falls back to an **extractive answer** that reads the best-matching passages aloud with their page numbers.
- **Retrieval:** PyMuPDF, BM25 and `bge-small-en-v1.5` (fastembed, ONNX), fused with reciprocal rank fusion.
- **TTS:** edge-tts, then gTTS, then pyttsx3 (optional).
- **UI:** Gradio 6 with a high-contrast theme, large fonts and labelled controls.

Your documents are processed only in memory for your browser session, and are never stored.

## Run locally (3 commands)

```bash
pip install -r requirements.txt
cp .env.example .env        # optional: add your GROQ_API_KEY
python app.py               # open http://127.0.0.1:7860
```

Without a key, the app runs fully offline for speech recognition (after a one-time model download). Answers are then extractive, and TTS still needs internet for edge-tts and gTTS.

## Run on the G14 with the GPU (RTX 4060)

In an Anaconda PowerShell prompt, from this folder:

```powershell
conda activate D:/conda_envs/fine_tune_env
powershell -ExecutionPolicy Bypass -File scripts/verify.ps1   # installs, checks CUDA, runs tests + evals
python app.py
```

`verify.ps1` backs up the environment to `env_backup.txt`, installs without upgrading existing packages, and runs `pip check`. faster-whisper uses CUDA through ctranslate2, which needs the cuBLAS and cuDNN 9 DLLs. The script reuses torch's DLL folder if torch is installed, and otherwise installs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`. If CUDA can't load, the app automatically uses the CPU and prints `[asr] CUDA load failed`.

To use a stronger local model on the GPU, set `LOCAL_WHISPER_MODEL=large-v3-turbo` in `.env`. It needs about 2 GB of VRAM in float16.

## Environment variables

All are optional; see `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | empty | Enables Groq ASR and LLM. If empty, the app runs in offline or extractive mode |
| `LOCAL_WHISPER_MODEL` | `base.en` | faster-whisper model for the fallback |
| `LOCAL_WHISPER_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `RETRIEVAL_MIN_CONFIDENCE` | `0.55` | Below this cosine score, the app says "I could not find that in your notes" |
| `ASR_REJECT_LOGPROB` / `ASR_ACCEPT_LOGPROB` | `-1.0` / `-0.5` | Below reject: asks you to repeat. Between the two: proceeds, saying "I heard: …" |
| `ASR_NO_SPEECH_REJECT` | `0.6` | Rejects the transcript if Whisper's no-speech probability is higher than this |
| `MAX_AUDIO_SECONDS`, `MAX_UPLOAD_MB`, `MAX_PAGES` | `30`, `20`, `300` | Input limits |
| `GROQ_RETRIES`, `GROQ_BREAKER_FAILURES`, `GROQ_BREAKER_COOLDOWN_S` | `2`, `3`, `60` | Groq resilience settings |

## Voice commands

Ask any question, or say:
- "summarise pages 2 to 4"
- "summarise the section on MFCCs"
- "read page 3"
- "read the section on beam search"
- "repeat"
- "slower" or "faster"
- "shorter" or "more detail"
- "help"

## Tests and evaluation

```bash
python -m pytest -q                      # 80 tests, external APIs mocked
python eval/eval_retrieval.py            # hit@1/3/5 on the sample document
python eval/eval_qa.py --text            # end-to-end, typed questions
python eval/eval_qa.py                   # end-to-end, your recorded questions (eval/audio/qa_XX.wav)
python eval/eval_wer.py                  # WER per backend, with and without biasing (eval/audio/wer_XX.wav)
```

`--synthetic` makes TTS audio, so the scripts can be tested before your recordings exist. Those outputs are prefixed `SYNTHETIC_` and are **not results**. All results go to `eval/results/`.

## Deploy to Hugging Face Spaces (free CPU)

1. Create a free account at huggingface.co, then go to **New Space**. Choose SDK **Gradio**, hardware **CPU basic (free)**, and visibility public or private.
2. Push this folder to the Space repo:
   ```bash
   git init && git add . && git commit -m "VoiceTutor"
   git remote add space https://huggingface.co/spaces/<your-username>/voicetutor
   git push space main            # sign in with an HF access token that has write access
   ```
   `.gitignore` keeps `.env`, `.venv` and your recordings' synthetic copies out of the repo.
3. In the Space, go to **Settings → Variables and secrets → New secret** and add `GROQ_API_KEY`. Leave `LOCAL_WHISPER_MODEL=base.en`, the default; larger models are too slow on 2 shared CPUs.
4. Wait for the build to finish. The first start downloads the Whisper `base.en` model (about 150 MB) and the embedding model (about 130 MB).

Notes:
- **The microphone needs HTTPS.** The `*.hf.space` URL is HTTPS, so this works there. On your laptop, `http://127.0.0.1` is also allowed by browsers.
- **Cold starts are slow.** A free Space sleeps after inactivity and takes about 1 to 2 minutes to wake up and load the models. Open it before a demo.
- **Memory:** the free tier has 16 GB RAM and 2 vCPUs. The app uses about 1 GB with `base.en`. Uploads are capped at 20 MB and 300 pages.
- Browsers may block the spoken welcome until you first click the page. That's normal autoplay policy.

## Assumptions

- **Chunk size is measured in words**: 300 words with a 50-word overlap, which is roughly 400 tokens. Chunks never cross a page, so every citation is exact.
- **DOCX and TXT files have no pages**, so they're split into "pages" of about 400 words, or at form-feed characters.
- **Retrieval confidence is the best dense cosine similarity.** The 0.55 threshold was chosen by inspecting the sample QA set, so the reported refusal accuracy on that set is optimistic.
- **A "section" is found by searching for its name**, then widening to the neighbouring chunks. The app doesn't detect headings.
- **The ASR confidence score** is the duration-weighted mean of `avg_logprob` and `no_speech_prob` across Whisper segments.
- **Groq failures fall back one turn at a time.** After 3 consecutive failures, the app skips Groq for 60 seconds.
- **General-knowledge mode needs Groq.** In offline mode it can't answer outside the notes, and says so.
- **edge-tts uses an unofficial Microsoft endpoint.** It's free but needs internet. gTTS also needs internet. pyttsx3 works offline but is optional.

## What I need to do

- [ ] Get a free Groq API key (console.groq.com → API Keys) and put it in `.env` as `GROQ_API_KEY=...`.
- [ ] Record the 25 WER sentences and 25 QA questions listed in `eval/record_prompts.md`, saved into `eval/audio/`.
- [ ] On the G14, run `scripts/verify.ps1` and send back `eval/results/verify_log.txt`.
- [ ] Open the app, test the microphone in the browser, and capture the screenshots in `docs/SCREENSHOTS.md`.
- [ ] Create a Hugging Face account and deploy the Space using the steps above, adding `GROQ_API_KEY` as a secret.
- [ ] Copy the real numbers from `eval/results/*.md` into the report. `docs/REPORT_NOTES.md` explains the setup.
