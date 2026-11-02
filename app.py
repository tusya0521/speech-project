"""VoiceTutor Gradio app. Run: python app.py"""
import gradio as gr

from voicetutor import asr, config, tts
from voicetutor.dialogue import Reply, Session
from voicetutor.ingest import IngestError, ingest
from voicetutor.metrics import Timer
from voicetutor.retrieval import Index

WELCOME = ("Welcome to VoiceTutor. Upload your notes, then press record and ask a question. "
           "Say help at any time to hear what I can do.")
PRIVACY = ("**Privacy:** your document is processed only for this browser session, kept in memory, "
           "and never stored. Closing the tab discards it.")
MODE = (f"Online mode: Groq `{config.GROQ_ASR_MODEL}` + `{config.GROQ_LLM_MODEL}`" if config.GROQ_API_KEY
        else "Offline mode: no Groq key, using local speech recognition and answers read from your notes")

CSS = """
body, .gradio-container { background:#000 !important; color:#fff !important; font-size:20px !important; }
.gradio-container * { font-size:20px; }
h1 { font-size:34px !important; color:#ffde00 !important; }
label, .label-wrap span, span[data-testid="block-info"] { color:#ffde00 !important; font-weight:700; }
textarea, input { background:#111 !important; color:#fff !important; border:2px solid #fff !important; }
button { border:3px solid #ffde00 !important; font-weight:700 !important; }
*:focus-visible { outline:4px solid #00e5ff !important; outline-offset:2px; }
"""


def speak(text: str, rate: int) -> str | None:
    return tts.synthesize(tts.speech_prep(text), rate)[0]


def on_load(state: Session):
    return speak(WELCOME, state.rate), state


def on_upload(file, state: Session):
    state = Session(rate=state.rate, length=state.length, general_ok=state.general_ok)  # fresh index per upload
    if file is None:
        return "No file.", None, state
    try:
        doc = ingest(file if isinstance(file, str) else file.name)
        state.doc, state.index = doc, Index(doc.chunks)
    except IngestError as e:
        msg = str(e)
        return msg, speak(msg, state.rate), state
    except Exception as e:
        msg = f"I could not index that file ({type(e).__name__}). Please try another file."
        return msg, speak(msg, state.rate), state
    phrases = [k for k in doc.keywords if " " in k]  # two-word terms read better as "topics"
    topics = ", ".join((phrases or doc.keywords)[:3])
    msg = f"I have read {doc.title}. It has {doc.num_pages} pages. Main topics include {topics}."
    if doc.unreadable_pages:
        pages = ", ".join(map(str, doc.unreadable_pages[:10]))
        msg += f" Warning: I could not read page {pages}. They may be scanned images."
    msg += " You can now ask a question."
    return msg, speak(msg, state.rate), state


def respond(text: str, state: Session, timer: Timer, prefix: str = ""):
    reply: Reply = state.handle(text, timer)
    audio = state.last_audio if reply.replay and state.last_audio else None
    if audio is None:
        with timer.stage("tts"):
            audio = speak(prefix + reply.speech, state.rate)
        if not reply.replay:
            state.last_audio = audio
    shown = prefix + reply.text
    if reply.pages:
        shown += "\n\nSources: " + ", ".join(f"page {p}" for p in reply.pages)
    timer.record(intent=reply.intent, pages=reply.pages)
    return shown, audio, state


def on_voice(audio_path, state: Session):
    timer = Timer()
    if not audio_path:
        return "", "No recording received.", None, state
    with timer.stage("asr"):
        res = asr.transcribe(audio_path, state.doc.keywords if state.doc else None)
    shown_transcript = f"{res.text or '(nothing recognised)'}   [{res.backend}, confidence {res.avg_logprob:.2f}]"
    note = (res.note + " ") if res.note else ""
    if res.decision == "reject":
        msg = note + "I did not catch that, please say it again."
        timer.record(intent="asr_reject")
        return shown_transcript, msg, speak(msg, state.rate), state
    if res.decision == "confirm":
        note += f"I heard: {res.text}. "
    answer, audio, state = respond(res.text, state, timer, prefix=note)
    return shown_transcript, answer, audio, state


def on_text(text, state: Session):
    if not text or not text.strip():
        return "Please type a question.", None, state
    return respond(text, state, Timer())


def on_settings(rate, length, general_ok, state: Session):
    state.rate, state.length, state.general_ok = int(rate), length, bool(general_ok)
    return state


with gr.Blocks(title="VoiceTutor") as demo:
    state = gr.State(Session())
    gr.Markdown("# VoiceTutor: ask your notes out loud")
    gr.Markdown(f"{MODE}. {PRIVACY}")
    with gr.Row():
        with gr.Column():
            doc_in = gr.File(label="1. Upload your notes (PDF, DOCX or TXT)", file_types=[".pdf", ".docx", ".txt"])
            doc_status = gr.Textbox(label="Document status", interactive=False, lines=3)
            mic = gr.Audio(sources=["microphone", "upload"], type="filepath",
                           label="2. Press record, ask your question, press stop")
            typed = gr.Textbox(label="Or type your question and press Enter", lines=1)
            send = gr.Button("Ask typed question")
        with gr.Column():
            transcript = gr.Textbox(label="What I heard (transcript)", interactive=False)
            answer = gr.Textbox(label="Answer with page citations", interactive=False, lines=10)
            audio_out = gr.Audio(label="Spoken answer (plays automatically)", autoplay=True, interactive=False)
            with gr.Accordion("Settings", open=False):
                rate = gr.Slider(-50, 50, value=0, step=5, label="Speech rate (percent faster or slower)")
                length = gr.Radio(["short", "normal", "detailed"], value="normal", label="Answer length")
                general = gr.Checkbox(value=False, label="Allow general knowledge answers (labelled 'not from your notes')")

    demo.load(on_load, state, [audio_out, state])
    doc_in.upload(on_upload, [doc_in, state], [doc_status, audio_out, state])
    mic.stop_recording(on_voice, [mic, state], [transcript, answer, audio_out, state])
    mic.upload(on_voice, [mic, state], [transcript, answer, audio_out, state])
    typed.submit(on_text, [typed, state], [answer, audio_out, state])
    send.click(on_text, [typed, state], [answer, audio_out, state])
    for c in (rate, length, general):
        c.change(on_settings, [rate, length, general, state], state)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        theme=gr.themes.Base(text_size=gr.themes.sizes.text_lg), css=CSS)
