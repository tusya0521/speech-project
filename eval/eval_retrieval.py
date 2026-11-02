"""Retrieval quality: hit@1/3/5 against gold pages, plus confidence on out-of-scope questions.

Usage: python eval/eval_retrieval.py [--doc PDF] [--qa CSV]
"""
import argparse
import csv
import os

from common import out, SAMPLE_DOC, environment, md_table, write_csv

from voicetutor import config
from voicetutor.ingest import ingest
from voicetutor.retrieval import Index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=SAMPLE_DOC)
    ap.add_argument("--qa", default=os.path.join(os.path.dirname(__file__), "qa_set.csv"))
    args = ap.parse_args()

    doc = ingest(args.doc)
    index = Index(doc.chunks)
    rows = []
    for r in csv.DictReader(open(args.qa, encoding="utf-8")):
        hits, conf = index.search(r["question"], k=5)
        pages = [h.chunk.page for h in hits]
        gold = {int(p) for p in r["gold_pages"].split(";") if p}
        row = {"id": r["id"], "in_scope": r["in_scope"], "gold": r["gold_pages"],
               "top5_pages": " ".join(map(str, pages)), "confidence": round(conf, 3),
               "refused": conf < config.RETRIEVAL_MIN_CONFIDENCE}
        for k in (1, 3, 5):
            row[f"hit@{k}"] = bool(gold & set(pages[:k])) if gold else ""
        rows.append(row)

    ins = [r for r in rows if r["in_scope"] == "yes"]
    outs = [r for r in rows if r["in_scope"] == "no"]
    summary = [{"metric": f"hit@{k}", "value": f"{sum(r[f'hit@{k}'] for r in ins) / len(ins):.2f}",
                "n": len(ins)} for k in (1, 3, 5)]
    summary += [
        {"metric": "in-scope wrongly refused (conf < threshold)", "value": sum(r["refused"] for r in ins), "n": len(ins)},
        {"metric": "out-of-scope correctly refused", "value": sum(r["refused"] for r in outs), "n": len(outs)},
        {"metric": "min in-scope confidence", "value": min(r["confidence"] for r in ins), "n": len(ins)},
        {"metric": "max out-of-scope confidence", "value": max(r["confidence"] for r in outs), "n": len(outs)},
    ]
    write_csv(out("retrieval_per_question.csv"), rows)
    with open(out("retrieval.md"), "w", encoding="utf-8") as f:
        f.write("# Retrieval evaluation\n\n" + environment())
        f.write(f"Document: `{os.path.basename(args.doc)}` ({doc.num_pages} pages, {len(doc.chunks)} chunks). "
                f"Embedding model `{config.EMBED_MODEL}`, BM25 + dense fused with RRF (k={config.RRF_K}). "
                f"Refusal threshold {config.RETRIEVAL_MIN_CONFIDENCE} "
                "(note: chosen by inspecting this same question set).\n\n")
        f.write(md_table(summary) + "\n## Per question\n\n" + md_table(rows))
    print(md_table(summary))


if __name__ == "__main__":
    main()
