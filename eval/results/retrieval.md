# Retrieval evaluation

Measured on LAPTOP-BEBIE2T7 (Windows, Python 3.10.11, local ASR device cpu, Groq key not set, local model base.en).

Document: `speech_recognition_notes.pdf` (10 pages, 10 chunks). Embedding model `BAAI/bge-small-en-v1.5`, BM25 + dense fused with RRF (k=60). Refusal threshold 0.55 (note: chosen by inspecting this same question set).

| metric | value | n |
|---|---|---|
| hit@1 | 0.95 | 20 |
| hit@3 | 0.95 | 20 |
| hit@5 | 1.00 | 20 |
| in-scope wrongly refused (conf < threshold) | 0 | 20 |
| out-of-scope correctly refused | 5 | 5 |
| min in-scope confidence | 0.59 | 20 |
| max out-of-scope confidence | 0.529 | 5 |

## Per question

| id | in_scope | gold | top5_pages | confidence | refused | hit@1 | hit@3 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| q01 | yes | 1 | 1 4 2 3 8 | 0.798 | False | True | True | True |
| q02 | yes | 1 | 1 10 4 2 8 | 0.788 | False | True | True | True |
| q03 | yes | 2 | 2 3 1 10 4 | 0.771 | False | True | True | True |
| q04 | yes | 2;8 | 2 8 9 6 3 | 0.731 | False | True | True | True |
| q05 | yes | 2 | 2 1 8 10 4 | 0.757 | False | True | True | True |
| q06 | yes | 3 | 3 1 9 2 4 | 0.744 | False | True | True | True |
| q07 | yes | 3 | 3 7 5 2 4 | 0.703 | False | True | True | True |
| q08 | yes | 3 | 3 6 4 5 7 | 0.707 | False | True | True | True |
| q09 | yes | 4 | 4 6 5 1 2 | 0.71 | False | True | True | True |
| q10 | yes | 4;6 | 4 6 1 2 3 | 0.675 | False | True | True | True |
| q11 | yes | 4 | 4 1 2 6 5 | 0.59 | False | True | True | True |
| q12 | yes | 5 | 5 6 1 9 4 | 0.712 | False | True | True | True |
| q13 | yes | 5 | 5 6 2 3 1 | 0.619 | False | True | True | True |
| q14 | yes | 6 | 6 9 5 1 7 | 0.793 | False | True | True | True |
| q15 | yes | 6 | 6 4 2 5 1 | 0.591 | False | True | True | True |
| q16 | yes | 7 | 7 9 8 3 1 | 0.631 | False | True | True | True |
| q17 | yes | 7 | 7 6 1 10 4 | 0.766 | False | True | True | True |
| q18 | yes | 8 | 8 4 7 2 6 | 0.792 | False | True | True | True |
| q19 | yes | 8 | 10 6 1 9 8 | 0.675 | False | False | False | True |
| q20 | yes | 9 | 9 2 10 5 1 | 0.788 | False | True | True | True |
| q21 | no |  | 2 9 5 1 4 | 0.418 | True |  |  |  |
| q22 | no |  | 1 2 6 9 4 | 0.406 | True |  |  |  |
| q23 | no |  | 1 6 5 3 2 | 0.412 | True |  |  |  |
| q24 | no |  | 2 3 9 1 6 | 0.529 | True |  |  |  |
| q25 | no |  | 6 9 3 5 2 | 0.363 | True |  |  |  |
