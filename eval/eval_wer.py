"""Word error rate per ASR backend, with and without vocabulary biasing.

Usage:
  python eval/eval_wer.py                         # uses eval/wer_set.csv (your recordings)
  python eval/eval_wer.py --synthetic             # DEV ONLY: makes TTS audio, outputs marked SYNTHETIC
Biasing keywords come from --doc (default: the sample study PDF).
"""
import argparse
import csv
import os
import re
import shutil
import time

import jiwer
from common import out, ROOT, SAMPLE_DOC, SYNTHETIC_BANNER, environment, md_table, write_csv

from voicetutor import asr, config, tts
from voicetutor.ingest import ingest


def normalize(text: str) -> str:
    """Same normalisation for reference and hypothesis: lower case, hyphens to spaces, no punctuation."""
    t = text.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return " ".join(t.split())


def make_synthetic(rows: list[dict]) -> list[dict]:
    out_dir = os.path.join(ROOT, "eval", "audio_synthetic")
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for i, r in enumerate(rows, start=1):
        path = os.path.join(out_dir, f"synthetic_wer_{i:02d}.mp3")
        if not os.path.exists(path):
            tmp, backend = tts.synthesize(r["reference_text"])
            if not tmp:
                raise SystemExit("TTS failed; cannot make synthetic audio.")
            shutil.move(tmp, path)
        out.append({**r, "audio_path": path})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "eval", "wer_set.csv"))
    ap.add_argument("--doc", default=SAMPLE_DOC)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.synthetic:
        rows = make_synthetic(rows)
    else:
        missing = [r["audio_path"] for r in rows if not os.path.exists(os.path.join(ROOT, r["audio_path"]))]
        if missing:
            raise SystemExit(f"{len(missing)} recordings missing (first: {missing[0]}). "
                             "See eval/record_prompts.md, or use --synthetic to test the script.")
    keywords = ingest(args.doc).keywords
    backends = (["groq"] if config.GROQ_API_KEY else []) + ["local"]

    per_utt, summary = [], []
    for backend in backends:
        for biased in (False, True):
            refs, hyps = {"plain": [], "technical": []}, {"plain": [], "technical": []}
            for r in rows:
                path = r["audio_path"] if os.path.isabs(r["audio_path"]) else os.path.join(ROOT, r["audio_path"])
                t = time.perf_counter()
                res = asr.transcribe(path, keywords if biased else None, backend=backend)
                ms = (time.perf_counter() - t) * 1000
                ref, hyp = normalize(r["reference_text"]), normalize(res.text)
                refs[r["tag"]].append(ref)
                hyps[r["tag"]].append(hyp)
                per_utt.append({"backend": backend, "biased": biased, "tag": r["tag"],
                                "audio": os.path.basename(path), "reference": ref, "hypothesis": hyp,
                                "wer": round(jiwer.wer(ref, hyp), 3), "avg_logprob": round(res.avg_logprob, 3),
                                "decision": res.decision, "used_backend": res.backend, "ms": round(ms)})
            all_r = refs["plain"] + refs["technical"]
            all_h = hyps["plain"] + hyps["technical"]
            summary.append({"backend": backend, "vocab_biasing": "on" if biased else "off",
                            "WER overall": f"{jiwer.wer(all_r, all_h):.3f}",
                            "WER plain": f"{jiwer.wer(refs['plain'], hyps['plain']):.3f}" if refs["plain"] else "",
                            "WER technical": f"{jiwer.wer(refs['technical'], hyps['technical']):.3f}"
                            if refs["technical"] else "",
                            "n": len(all_r)})
            print(summary[-1])

    prefix = "SYNTHETIC_" if args.synthetic else ""
    write_csv(out(f"{prefix}wer_per_utterance.csv"), per_utt)
    write_csv(out(f"{prefix}wer_summary.csv"), summary)
    with open(out(f"{prefix}wer.md"), "w", encoding="utf-8") as f:
        f.write("# ASR word error rate\n\n" + (SYNTHETIC_BANNER if args.synthetic else "") + environment())
        f.write("Normalisation: lower case, hyphens to spaces, punctuation removed. WER is corpus level "
                "(total errors / total reference words). Biasing prompt = top keywords of "
                f"`{os.path.basename(args.doc)}`.\n\n")
        f.write(md_table(summary))


if __name__ == "__main__":
    main()
