# Report notes

These are raw material for the 20+ page report. Every number here is copied from a file in `eval/results/` and names the machine it was measured on. Anything not yet measured is marked **PENDING**.

## 1. Tools and versions

These versions were installed on the dev machine on 2026-11-03.

| Tool | Version | Role | Why chosen |
|---|---|---|---|
| Python | 3.10.11 (dev) / 3.11 (HF Space) | Runtime | Supported by all the libraries below |
| Gradio | 6.31.0 | UI | Built-in mic recording and audio autoplay, free HF Spaces hosting, accessible labelled components |
| groq (SDK) | 1.7.0 | ASR + LLM API | Free tier, very low latency, hosts Whisper and Llama |
| Groq `whisper-large-v3-turbo` | n/a | Online ASR | Fast large Whisper. `verbose_json` gives `avg_logprob` and `no_speech_prob` for confidence |
| Groq `llama-3.3-70b-versatile` | n/a | Answers and summaries | Strong instruction following for grounded, cited answers |
| Groq `llama-3.1-8b-instant` | n/a | Intent fallback, query rewrite | Cheap and fast for short tasks |
| faster-whisper | 1.2.1 (CTranslate2 4.8.5) | Offline ASR fallback | About 4× faster than reference Whisper. int8 on CPU, float16 on GPU. Decodes audio with PyAV, so no system ffmpeg is needed |
| fastembed | 0.9.1 (onnxruntime 1.23.2) | Dense embeddings | `BAAI/bge-small-en-v1.5` (384 dimensions) runs on CPU with ONNX, with no PyTorch, so it suits the free CPU Space |
| rank-bm25 | 0.2.2 | Keyword retrieval | Exact technical terms ("MFCC", "Viterbi") that embeddings can blur |
| PyMuPDF | 1.28.2 | PDF text extraction | Fast, page-accurate, detects password protection |
| python-docx | 1.2.0 | DOCX loading | Simple paragraph access |
| edge-tts | 7.3.1 | Primary TTS | Free, natural neural voices, speaking-rate control |
| gTTS | 2.2.4 | TTS fallback | Free, different service, so it gives redundancy |
| pyttsx3 | optional | Offline TTS fallback | Works with no internet (SAPI5 on Windows, espeak on Linux) |
| jiwer | 4.0.0 | WER | Standard WER implementation |
| pytest | 9.2.0 | Tests | |

## 2. Design decisions

1. **Hybrid retrieval with RRF.** BM25 catches rare exact terms, and dense embeddings catch paraphrases. RRF (k = 60) combines the two rankings without having to calibrate their very different score scales.
2. **Confidence is the dense cosine, not the RRF score.** RRF scores are rank-based and always look similar, while the cosine score measures how similar the question actually is to the notes. The threshold is 0.55.
3. **Page-aware chunking.** Chunks of about 300 words with a 50-word overlap never cross a page, so every chunk cites exactly one page.
4. **Vocabulary biasing.** The top 25 TF-IDF terms of the uploaded document (pages treated as documents, bigrams boosted) go into Whisper's `prompt` / `initial_prompt`, capped at about 120 words because of Groq's 224-token limit. These terms also become the spoken "main topics".
5. **Three-way ASR decision** (accept / confirm / reject), based on duration-weighted `avg_logprob`, `no_speech_prob` and word count. "Confirm" still answers but says "I heard: …", so a blind user can catch a misrecognition without having to read the screen.
6. **Rule-first intent routing.** Regex rules are predictable, free and testable. The LLM classifier is used only when no rule matches and the text doesn't look like a question. Commands only fire for utterances of 6 words or fewer, so "why are longer frames used?" is treated as a question.
7. **Resilience.** Groq calls get 2 retries with exponential backoff. A failure on one turn falls back to faster-whisper and/or an extractive answer. A circuit breaker shared by ASR and the LLM skips Groq for 60 seconds after 3 consecutive failures.
8. **The extractive answer is the no-LLM baseline.** It reads the opening sentences of the confident passages, with their page numbers. It can't hallucinate, but it doesn't synthesise across passages.
9. **speech_prep().** Removes markdown, bullets and tables, expands symbols and simple equations into words, and splits sentences longer than 25 words. This is needed because LLM output is written for the screen, not the ear.
10. **Per-session state.** Each browser tab gets its own `Session` in `gr.State`, holding the document, index, history and cached audio. Nothing is written to disk except the anonymous latency log.
11. **Hyphen normalisation in retrieval.** Whisper writes "tri-phone", which dropped the cosine score below the threshold (measured: 0.50 vs 0.58). The search now also tries the joined spelling.

## 3. Evaluation procedure

- **Data:** an original 10-page document (`eval/sample_doc/build_sample.py`) with one topic per page; 25 QA questions (`eval/qa_set.csv`), 20 in scope with gold pages and 5 out of scope; and 25 WER sentences (`eval/wer_set.csv`), 12 plain and 13 technical, recorded by the author.
- **Retrieval** (`eval_retrieval.py`): hit@k means at least one gold page is among the top-k chunks' pages. Refusal is checked by comparing the confidence with the threshold.
- **WER** (`eval_wer.py`): each available backend (Groq, local) is run with vocabulary biasing on and off. Reference and hypothesis are normalised the same way (lower case, hyphens to spaces, punctuation removed). WER is corpus-level per tag.
- **End-to-end QA** (`eval_qa.py`): each question runs in a fresh session from recorded audio, through ASR, routing, retrieval, the LLM or extractive answer, and TTS. It records the cited pages, citation hit, refusal correctness, and per-stage latency (p50/p95, nearest rank).
- **Honesty rules:** synthetic TTS audio was used only to test the scripts, and those outputs are prefixed `SYNTHETIC_`. The refusal threshold was chosen by looking at this same QA set, so the refusal results are optimistic and should be described that way.

## 4. Results

### 4.1 Retrieval (from `eval/results/retrieval.md`)

Measured on the dev laptop (Windows, Python 3.10.11, CPU) on 2026-11-03, using `BAAI/bge-small-en-v1.5`, BM25 and RRF on the 10-page document (10 chunks).

| Metric | Value | n |
|---|---|---|
| hit@1 | 0.95 | 20 |
| hit@3 | 0.95 | 20 |
| hit@5 | 1.00 | 20 |
| In-scope wrongly refused | 0 | 20 |
| Out-of-scope correctly refused | 5 | 5 |
| Min in-scope confidence | 0.59 | 20 |
| Max out-of-scope confidence | 0.529 | 5 |

The one miss at hit@1 and hit@3 is q19 ("What is vocabulary biasing?"), which ranked page 10 (domain vocabulary challenges) above page 8. The gap between in-scope and out-of-scope confidence is only 0.06.

### 4.2 End-to-end QA, typed input, offline extractive mode (from `eval/results/qa_text.md`)

Measured on the dev laptop (CPU, no Groq key) on 2026-11-03.

| Metric | Value |
|---|---|
| Citation hit | 19/20 |
| In-scope answered | 20/20 |
| Out-of-scope correctly refused | 5/5 |

| Stage | p50 ms | p95 ms |
|---|---|---|
| Retrieval | 11 | 17 |
| TTS (edge-tts, network) | 1305 | 1580 |
| Total | 1320 | 1595 |

### 4.3 PENDING (needs the G14, a Groq key and your recordings)

| Table | Source file after `verify.ps1` |
|---|---|
| WER: Groq vs local, biasing on/off, plain vs technical | `eval/results/wer_with-key.md` |
| End-to-end QA with recorded voice, Groq mode (accuracy + latency) | `eval/results/qa_recorded_with-key.md` |
| End-to-end QA with recorded voice, offline mode, GPU faster-whisper | `eval/results/qa_recorded_no-key.md` |
| Answer quality of Llama vs the extractive baseline | Compare the `answer` column in the two `qa_*_per_question.csv` files (manual reading) |

## 5. Limitations (honest)

- **The refusal threshold is fragile.** It's a single global cosine threshold tuned on 25 questions, with a gap of only 0.06 between classes. Other documents, especially very short or very long chunks, may need a different value.
- **The evaluation document is small and clean.** It has 10 pages, one topic per page, and was written by us. Real lecture notes are messier, with multi-column text, formulas and headers.
- **No OCR.** Scanned pages are detected and announced, but they can't be read.
- **The extractive mode reads passages, it doesn't answer.** It may read the right page without directly answering the question.
- **The LLM can still misquote a passage.** Grounding is enforced by the prompt, not verified sentence by sentence.
- **edge-tts and gTTS need internet.** edge-tts uses an unofficial endpoint that could change. pyttsx3 is the only fully offline option.
- **English only.** Accents not well represented in Whisper's training data may have a higher WER. With one speaker (the author), the results don't generalise across speakers.
- **Browser autoplay rules** may silence the welcome message until the first click.
- **Free-tier limits.** Groq rate limits trigger the fallbacks during heavy use. HF free Spaces sleep and have slow cold starts.
- **No streaming.** The full answer is generated and synthesised before playback starts.
