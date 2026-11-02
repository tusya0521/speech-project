"""End-to-end QA: answer, citations, refusal correctness and per-stage latency (p50/p95).

Each question runs in a fresh session (no history). Input audio per question:
  eval/audio/qa_XX.wav if you recorded it (default; missing files stop the run),
  --synthetic: TTS audio (DEV ONLY, outputs marked SYNTHETIC),
  --text: no audio, ASR stage skipped.
"""
import argparse
import csv
import math
import os
import shutil

from common import out, ROOT, SAMPLE_DOC, SYNTHETIC_BANNER, environment, md_table, write_csv

from voicetutor import asr, config, tts
from voicetutor.dialogue import Session
from voicetutor.ingest import ingest
from voicetutor.metrics import Timer, percentile
from voicetutor.retrieval import Index

STAGES = ["asr", "rewrite", "retrieval", "llm", "tts", "total"]


def audio_for(qid: str, question: str, mode: str) -> str | None:
    if mode == "text":
        return None
    if mode == "synthetic":
        path = os.path.join(ROOT, "eval", "audio_synthetic", f"synthetic_{qid}.mp3")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp, _ = tts.synthesize(question)
            shutil.move(tmp, path)
        return path
    path = os.path.join(ROOT, "eval", "audio", f"{qid.replace('q', 'qa_')}.wav")
    if not os.path.exists(path):
        raise SystemExit(f"Missing recording {path}. See eval/record_prompts.md, or use --synthetic / --text.")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=SAMPLE_DOC)
    ap.add_argument("--qa", default=os.path.join(ROOT, "eval", "qa_set.csv"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--text", action="store_true")
    args = ap.parse_args()
    mode = "synthetic" if args.synthetic else "text" if args.text else "recorded"

    doc = ingest(args.doc)
    index = Index(doc.chunks)
    rows = []
    for r in csv.DictReader(open(args.qa, encoding="utf-8")):
        session = Session(doc=doc, index=index)
        timer = Timer()
        path = audio_for(r["id"], r["question"], mode)
        heard, asr_backend = r["question"], "text"
        if path:
            with timer.stage("asr"):
                res = asr.transcribe(path, doc.keywords)
            heard, asr_backend = res.text, res.backend
        reply = session.handle(heard, timer)
        with timer.stage("tts"):
            tts.synthesize(tts.speech_prep(reply.speech), 0)
        lat = timer.record(eval="qa", id=r["id"])
        gold = {int(p) for p in r["gold_pages"].split(";") if p}
        refused = reply.intent == "not_found"
        rows.append({"id": r["id"], "in_scope": r["in_scope"], "question": r["question"], "heard": heard,
                     "asr": asr_backend, "answer_mode": "groq" if config.GROQ_API_KEY else "extractive",
                     "refused": refused,
                     "refusal_correct": refused == (r["in_scope"] == "no"),
                     "cited_pages": " ".join(map(str, reply.pages)), "gold_pages": r["gold_pages"],
                     "citation_hit": bool(gold & set(reply.pages)) if gold else "",
                     **{f"{s}_ms": lat.get(s, 0.0) for s in STAGES},
                     "answer": reply.text.replace("\n", " ")})
        print(r["id"], refused, rows[-1]["cited_pages"], round(lat["total"]))

    ins = [r for r in rows if r["in_scope"] == "yes"]
    outs = [r for r in rows if r["in_scope"] == "no"]
    acc = [{"metric": "citation hit (gold page among cited pages)",
            "value": f"{sum(r['citation_hit'] is True for r in ins)}/{len(ins)}"},
           {"metric": "in-scope answered (not refused)", "value": f"{sum(not r['refused'] for r in ins)}/{len(ins)}"},
           {"metric": "out-of-scope correctly refused", "value": f"{sum(r['refused'] for r in outs)}/{len(outs)}"}]
    lat_rows = []
    for s in STAGES:
        vals = [r[f"{s}_ms"] for r in rows if r[f"{s}_ms"] > 0]
        fmt = lambda v: "" if math.isnan(v) else f"{v:.0f}"
        lat_rows.append({"stage": s, "p50 ms": fmt(percentile(vals, 50)), "p95 ms": fmt(percentile(vals, 95)),
                         "n": len(vals)})

    prefix = "SYNTHETIC_" if mode == "synthetic" else ""
    write_csv(out(f"{prefix}qa_{mode}_per_question.csv"), rows)
    with open(out(f"{prefix}qa_{mode}.md"), "w", encoding="utf-8") as f:
        f.write(f"# End-to-end QA ({mode} input)\n\n" + (SYNTHETIC_BANNER if mode == "synthetic" else ""))
        f.write(environment() + f"Answer mode: {rows[0]['answer_mode']}. "
                "The first question includes model loading (cold start), so p95 is pessimistic. "
                "In extractive mode no LLM is called, so the llm row is empty.\n\n")
        f.write("## Accuracy\n\n" + md_table(acc) + "\n## Latency\n\n" + md_table(lat_rows))
    print(md_table(acc) + md_table(lat_rows))


if __name__ == "__main__":
    main()
