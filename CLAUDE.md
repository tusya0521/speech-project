# VoiceTutor: project context for AI assistants

Read this first, along with `CONTEXT.md` (the project glossary), `docs/ARCHITECTURE.md` and `docs/REPORT_NOTES.md`. Keep this file current when decisions, setup or open tasks change.

## What this is
This is a final-year B.Tech CSE Speech Recognition course project, written up as a 20+ page report. VoiceTutor is a voice-first study assistant for visually impaired and dyslexic students. The student uploads notes (PDF, DOCX or TXT), asks by voice, and hears a grounded answer with page citations. English only. Everything is free and open source. The code must stay small enough for the student to explain every module.

## Layout
- `app.py`: Gradio 6 UI. In Gradio 6, `theme` and `css` are passed to `launch()`, not `Blocks()`.
- `voicetutor/`: `config` (env thresholds), `ingest` (load, chunk, TF-IDF keywords), `retrieval` (BM25 + bge-small + RRF, `Index`), `asr` (Groq, then faster-whisper, plus `decide`), `llm` (Groq, retries, `Breaker`, prompts), `dialogue` (`route`, `Session.handle`), `tts` (`speech_prep`, edge-tts, gTTS, pyttsx3), `metrics` (Timer to JSONL).
- `eval/`: sample document builder and PDF, `qa_set.csv`, `wer_set.csv`, `record_prompts.md`, three eval scripts, and `results/`.
- `tests/`: 80 pytest tests. External APIs and the embedding model are mocked in `conftest.py`.
- `scripts/verify.ps1`: one-shot install and verification on the G14, logged to `eval/results/verify_log.txt`.

## Machines
- **Dev box** (where the code was written): Windows, Python 3.10.11, CPU only. Uses a local `.venv`. It has no valid Groq key.
- **G14 (RA402XV, RTX 4060 8GB)**: where the student runs the project. Install into the existing conda env `D:/conda_envs/fine_tune_env`. `verify.ps1` backs it up with `pip freeze`, installs with `--upgrade-strategy only-if-needed`, then runs `pip check`. faster-whisper on CUDA needs the cuBLAS and cuDNN 9 DLLs. The script puts torch's `lib` folder on PATH, or pip-installs `nvidia-*-cu12`. If CUDA fails, the app falls back to CPU automatically. On the G14, set `LOCAL_WHISPER_MODEL=large-v3-turbo` in `.env`.
- **Hugging Face Space**: free CPU tier, Gradio SDK 6.31.0. Front matter is in README. Keep `LOCAL_WHISPER_MODEL=base.en`.

## Design decisions (agreed with the student)
1. **No Groq key means extractive answers.** The answer is the confident retrieved passages read aloud as "From page N: …". Summaries take the first chunk of each page in the range. Rewriting and the LLM side of intent routing are skipped.
2. **Verification split.** The dev box runs mocked tests, the no-key path and the retrieval eval. The G14 runs everything else.
3. **QA eval audio comes from the student's own recordings** (`eval/audio/qa_XX.wav`, `wer_XX.wav`). Synthetic TTS audio is only for script testing, with outputs prefixed `SYNTHETIC_`, and is never reported.
4. **A section is found by retrieval, not by headings.** The section name is used as a search query and widened to neighbouring chunks.
5. **Groq failures are handled per turn**, with 2 retries and backoff. After 3 consecutive failures the breaker opens for 60 seconds. ASR and the LLM share the breaker.

## Status (2026-11-03)
- Phases 1 to 3 are built. 80/80 tests pass on the dev box.
- **Verified on the dev box:** the no-key end-to-end pipeline (edge-tts, faster-whisper `base.en` on CPU, retrieval, extractive answer); the server starts with HTTP 200; the invalid-key path against the real Groq API (401, then offline fallback, then breaker opens); bad file, silence and empty input handling.
- **Real results so far** (dev box, CPU, no key): retrieval hit@1/3/5 = 0.95/0.95/1.00, with 5/5 out-of-scope refused. Text-mode QA: 19/20 citation hits, 5/5 refused. Details are in `docs/REPORT_NOTES.md`.
- **Not yet verified:** the happy path with a valid Groq key, the GPU, the browser microphone, and the HF deployment.

## Open tasks for the student
- [ ] Get a free Groq API key and put it in `.env` as `GROQ_API_KEY`. Never commit it, and never paste it into chat.
- [ ] Record the 50 clips listed in `eval/record_prompts.md` (deferred by the student, 2026-11-03).
- [ ] Run `scripts/verify.ps1` on the G14 and share `eval/results/verify_log.txt`.
- [ ] Fill the PENDING tables in `docs/REPORT_NOTES.md` from `eval/results/*_with-key.md` and `*_no-key.md`.
- [ ] Capture the screenshots in `docs/SCREENSHOTS.md`.
- [ ] Deploy the Space (README steps) and add `GROQ_API_KEY` as a Space secret.

## Rules
- Never invent WER, latency or accuracy numbers. Report tables are filled only from files in `eval/results/`, stating the machine.
- The app must never crash on a bad file, silent audio or an API error. It must say something useful, both aloud and on screen.
- No databases, no auth, no microservices. Keep modules small, with short docstrings.
