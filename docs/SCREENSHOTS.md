# Screenshots to capture for the report

Use the G14 with `python app.py` and Chrome at 100% zoom, unless a step says otherwise. Use the sample document `eval/sample_doc/speech_recognition_notes.pdf`. Save the files as `docs/screenshots/NN_name.png`.

| # | Screen | Input / action | What should be visible |
|---|---|---|---|
| 1 | Home page, online mode | Open http://127.0.0.1:7860 with `GROQ_API_KEY` set | Title, "Online mode: Groq …" line, privacy note, high-contrast black and yellow UI |
| 2 | Home page, offline mode | Restart with the key line commented out in `.env` | "Offline mode: no Groq key …" |
| 3 | Upload confirmation | Upload the sample PDF | Document status: "I have read Speech Recognition Fundamentals… 10 pages. Main topics include …", with the audio player filled |
| 4 | Voice question, accepted | Record: "How is word error rate calculated?" | Transcript with `[groq, confidence …]`, answer mentioning page 9, "Sources: page 9", audio playing |
| 5 | Low-confidence confirm | Mumble or speak quickly with background noise | Answer starts "I heard: …" (if you can't trigger it, skip and note this in the report) |
| 6 | Rejected audio | Record 3 seconds of silence | "I did not catch that, please say it again." / "I heard only silence." |
| 7 | Out-of-scope refusal | Ask "What is the capital city of Australia?" | "I could not find that in your notes." with no sources |
| 8 | General-knowledge mode | Settings → tick "Allow general knowledge answers", then repeat #7 | Answer starts "This is not from your notes." |
| 9 | Read page | Say "read page 4" | Page 4 text, "Sources: page 4" |
| 10 | Summarise range | Say "summarise pages 3 to 5" | Summary citing pages 3 to 5 |
| 11 | Section | Say "read the section on beam search" | Text from page 6 |
| 12 | Commands | Say "slower" after an answer, then "repeat" | Same answer text, slower audio; repeat replays without delay |
| 13 | Follow-up | Ask "What is a triphone?" then "Why are they tied?" | The second answer is about decision-tree tying (page 4) |
| 14 | Typed fallback | Type "what is a lattice" and press Enter | Answer citing page 6 |
| 15 | Bad file | Upload any `.exe` or a renamed corrupt `.pdf` | Spoken and shown error, with no crash |
| 16 | Groq failure fallback | Set `GROQ_API_KEY=gsk_invalid` in `.env`, restart, ask a question | Note "Online speech recognition failed…" and "Online answers are unavailable right now…" with an extractive answer |
| 17 | Settings panel | Open Settings | Rate slider, length radio, general-knowledge checkbox with labels |
| 18 | Keyboard focus | Press Tab through the controls | Cyan focus outline visible |
| 19 | Hugging Face Space | The deployed `*.hf.space` URL | Same UI over HTTPS, with the mic permission prompt |
| 20 | Tests | Terminal: `python -m pytest -q` | "80 passed" (or the actual count) |
| 21 | Eval output | Terminal: `python eval/eval_retrieval.py` | hit@k table |
| 22 | GPU check | Terminal output of `verify.ps1` section 3 | "local whisper device: cuda" |
| 23 | Latency log | `logs/latency.jsonl` open in an editor | Per-stage ms per turn |
