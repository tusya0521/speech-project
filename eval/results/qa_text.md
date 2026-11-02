# End-to-end QA (text input)

Measured on LAPTOP-BEBIE2T7 (Windows, Python 3.10.11, local ASR device cpu, Groq key not set, local model base.en).

Answer mode: extractive. The first question includes model loading (cold start), so p95 is pessimistic. In extractive mode no LLM is called, so the llm row is empty.

## Accuracy

| metric | value |
|---|---|
| citation hit (gold page among cited pages) | 19/20 |
| in-scope answered (not refused) | 20/20 |
| out-of-scope correctly refused | 5/5 |

## Latency

| stage | p50 ms | p95 ms | n |
|---|---|---|---|
| asr |  |  | 0 |
| rewrite |  |  | 0 |
| retrieval | 11 | 17 | 25 |
| llm | 0 | 0 | 5 |
| tts | 1305 | 1580 | 25 |
| total | 1320 | 1595 | 25 |
