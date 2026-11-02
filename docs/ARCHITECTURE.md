# VoiceTutor architecture

All diagrams use a black-and-white theme so they print cleanly in the report. Every module and function name below matches the code.

## 1. System architecture

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#000000','primaryBorderColor':'#000000','lineColor':'#000000','secondaryColor':'#ffffff','tertiaryColor':'#ffffff','clusterBkg':'#ffffff','clusterBorder':'#000000'}}}%%
flowchart LR
    U([Student: browser]) -->|mic audio / typed text / file| UI[app.py<br/>Gradio Blocks]
    UI -->|file| ING[ingest.py<br/>load_pages, chunk_pages, top_keywords]
    ING -->|chunks| IDX[retrieval.py<br/>Index: BM25 + bge-small, RRF]
    UI -->|audio + keywords| ASR[asr.py<br/>transcribe, decide]
    ASR -->|transcript| DLG[dialogue.py<br/>Session.handle, route]
    UI -->|typed text| DLG
    DLG -->|query| IDX
    DLG -->|passages| LLM[llm.py<br/>grounded_answer, summarize,<br/>rewrite_query, classify_intent]
    DLG -->|reply text| UI
    UI -->|speech_prep text| TTS[tts.py<br/>synthesize]
    TTS -->|mp3| UI
    UI -->|stage timings| MET[metrics.py<br/>Timer to logs/latency.jsonl]
    ASR -.->|whisper-large-v3-turbo| GROQ[(Groq API)]
    LLM -.->|llama-3.3-70b / 3.1-8b| GROQ
    ASR -.->|fallback| FW[faster-whisper<br/>local CPU/GPU]
    TTS -.-> EDGE[(edge-tts)] & GT[(gTTS)] & PY[pyttsx3]
    CFG[config.py<br/>.env settings] -.-> ASR & IDX & LLM & DLG
    subgraph SESSION [Per-session state: gr.State, memory only]
        IDX
        DLG
    end
```

## 2. ASR pipeline (`asr.transcribe`)

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#000000','primaryBorderColor':'#000000','lineColor':'#000000'}}}%%
flowchart TD
    A[Audio file] --> B[load_audio<br/>PyAV decode, 16 kHz mono]
    B -->|decode error| R1[reject: could not read recording]
    B --> C{RMS < 0.001?}
    C -->|yes| R2[reject: only silence]
    C -->|no| D{Longer than MAX_AUDIO_SECONDS?}
    D -->|yes| E[Trim, add note]
    D -->|no| F
    E --> F[vocabulary_prompt<br/>top TF-IDF keywords, about 120 words]
    F --> G{GROQ_API_KEY set and<br/>breaker closed?}
    G -->|yes| H[Groq Whisper, verbose_json<br/>call_groq: 2 retries with backoff]
    H -->|failed| I[note: used offline model]
    G -->|no| J
    I --> J[faster-whisper local<br/>initial_prompt, VAD, beam 5]
    H -->|segments| K
    J -->|segments| K[aggregate<br/>duration-weighted avg_logprob, no_speech_prob]
    K --> L{decide}
    L -->|no words, no_speech > 0.6 or logprob < -1.0| M[reject: please say it again]
    L -->|-1.0 to -0.5| N[confirm: answer, prefixed 'I heard ...']
    L -->|>= -0.5| O[accept]
```

## 3. Dialogue and fallback flow (`Session.handle`)

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#000000','primaryBorderColor':'#000000','lineColor':'#000000'}}}%%
flowchart TD
    T[Utterance] --> RT{route: rules first<br/>commands only if 6 words or fewer}
    RT -->|no rule, not question-like,<br/>Groq available| CL[classify_intent LLM]
    CL --> RT2[intent]
    RT -->|help| HP[Help text]
    RT -->|repeat| RP[Replay cached audio<br/>no LLM, no TTS]
    RT -->|slower / faster| SP[Change rate, re-speak last reply]
    RT -->|shorter / longer| LN[Change length, re-run last request]
    RT -->|read| RD{page number?}
    RD -->|yes| RDP[Read page text<br/>warn if unreadable]
    RD -->|no| SEC[Section: search name,<br/>widen to neighbouring chunks]
    RT -->|summarize| SM[Pages / section / whole doc]
    SM --> SMG{Groq OK?}
    SMG -->|yes| SML[llm.summarize]
    SMG -->|no| EX1[Extractive: first sentences per page]
    RT -->|question| FU{Follow-up and Groq OK?}
    FU -->|yes| RW[rewrite_query, last 3 turns]
    FU -->|no| RET
    RW --> RET[Index.search, confidence = max cosine]
    RET --> TH{confidence >= 0.55?}
    TH -->|no| GK{General knowledge allowed<br/>and Groq OK?}
    GK -->|yes| GA["general_answer<br/>prefixed 'This is not from your notes.'"]
    GK -->|no| NF[I could not find that in your notes.]
    TH -->|yes| GQ{Groq OK?}
    GQ -->|yes| GR[grounded_answer]
    GR -->|NOT_FOUND| GK
    GR --> ANS[Answer + cited pages]
    GQ -->|no key / failed / breaker open| EX2[Extractive: confident passages,<br/>'From page N: ...']
```

## 4. Sequence of one voice turn

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#000000','primaryBorderColor':'#000000','lineColor':'#000000','actorBkg':'#ffffff','actorBorder':'#000000','noteBkgColor':'#ffffff','noteBorderColor':'#000000','signalColor':'#000000','signalTextColor':'#000000'}}}%%
sequenceDiagram
    actor S as Student
    participant UI as app.py (Gradio)
    participant A as asr.py
    participant G as Groq API
    participant D as dialogue.Session
    participant R as retrieval.Index
    participant T as tts.py
    participant M as metrics.Timer
    S->>UI: speak, press stop (stop_recording)
    UI->>M: Timer()
    UI->>A: transcribe(path, doc.keywords)
    A->>G: whisper-large-v3-turbo (prompt = keywords)
    G-->>A: segments (avg_logprob, no_speech_prob)
    A-->>UI: ASRResult(text, decision)
    alt decision == reject
        UI->>T: "I did not catch that, please say it again"
    else accept / confirm
        UI->>D: handle(text, timer)
        D->>R: search(query)
        R-->>D: hits, confidence
        D->>G: grounded_answer(passages)
        G-->>D: "On page 9, ..."
        D-->>UI: Reply(text, pages)
        UI->>T: synthesize(speech_prep(text), rate)
        T-->>UI: mp3
    end
    UI->>M: record() appends JSONL
    UI-->>S: transcript, answer + "Sources: page 9", autoplay audio
```
