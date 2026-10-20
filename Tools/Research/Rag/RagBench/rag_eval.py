"""Korean retrieval benchmark for the RAG sample.

  bench    — embedder comparison on KorQuAD-dev (the sample corpus), Ko-StrategyQA and AutoRAGRetrieval (MTEB-ko):
             nDCG@10, Recall@1/5/10, MRR@10, CPU batch-1 query latency, size.
  tflite   — the exported e5 .tflite encoders (export_e5.py): recall on the same sets, cosine parity vs HF, latency.
  tokcheck — device tokenisation rule (raw SentencePiece ids, as the AAR's sentencepiece_tokenizer emits them)
             against the HF tokenizer.

Run through Run-RagBench.ps1 (venv, HF token). Results go to Builds/Logs/rag-*.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LOGS = ROOT / "Builds" / "Logs"

# name -> (hf id, query prefix, passage prefix). Prefixes follow each model card.
MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "e5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "e5-large": ("intfloat/multilingual-e5-large", "query: ", "passage: "),
    "e5-large-instruct": ("intfloat/multilingual-e5-large-instruct",
                          "Instruct: Given a question, retrieve passages that answer the question\nQuery: ", ""),
    "embeddinggemma": ("google/embeddinggemma-300m", "task: search result | query: ", "title: none | text: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),
}
SETS = ("korquad", "strategyqa", "autorag")


def names(s: str) -> list[str]:
    """Comma list from the CLI; PowerShell turns `a,b` into an array and hands it over space-joined."""
    return [x for x in s.replace(",", " ").split() if x]


def load_set(name: str) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """(corpus id->text, query id->text, qrels query id->relevant corpus ids)."""
    from datasets import load_dataset

    if name == "korquad":  # the sample corpus: KorQuAD 1.0 dev paragraphs, every question points at its paragraph
        rows = load_dataset("KorQuAD/squad_kor_v1", split="validation")
        corpus, queries, qrels = {}, {}, {}
        for r in rows:
            cid = corpus.setdefault(r["context"], f"c{len(corpus)}")
            queries[r["id"]] = r["question"]
            qrels[r["id"]] = {cid}
        return {v: k for k, v in corpus.items()}, queries, qrels
    repo, split = {"strategyqa": ("mteb/Ko-StrategyQA", "dev"), "autorag": ("mteb/AutoRAGRetrieval", "test")}[name]
    corpus = {str(r["_id"]): f"{r.get('title') or ''} {r['text']}".strip() for r in load_dataset(repo, "corpus", split=split)}
    queries = {str(r["_id"]): r["text"] for r in load_dataset(repo, "queries", split=split)}
    qrels: dict[str, set[str]] = {}
    for r in load_dataset(repo, "qrels", split=split):
        if r["score"] > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    return corpus, {q: t for q, t in queries.items() if q in qrels}, qrels


def score(ranked: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[str, float]:
    m = {"ndcg@10": 0.0, "r@1": 0.0, "r@5": 0.0, "r@10": 0.0, "mrr@10": 0.0}
    for q, docs in ranked.items():
        rel = qrels[q]
        m["ndcg@10"] += sum(1 / math.log2(i + 2) for i, d in enumerate(docs[:10]) if d in rel) / \
            sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
        for k in (1, 5, 10):
            m[f"r@{k}"] += len(set(docs[:k]) & rel) / len(rel)
        m["mrr@10"] += next((1 / (i + 1) for i, d in enumerate(docs[:10]) if d in rel), 0.0)
    return {k: round(v / len(ranked), 4) for k, v in m.items()}


def topk(q_emb, d_emb, qids: list[str], dids: list[str], k: int = 10) -> dict[str, list[str]]:
    import torch

    q, d = torch.as_tensor(q_emb, dtype=torch.float32), torch.as_tensor(d_emb, dtype=torch.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q, d = q.to(dev), d.to(dev)
    out = {}
    for i in range(0, len(qids), 1024):
        idx = (q[i:i + 1024] @ d.T).topk(k, dim=1).indices.cpu().tolist()
        out.update({qids[i + j]: [dids[x] for x in row] for j, row in enumerate(idx)})
    return out


def cmd_bench(args) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    data = {s: load_set(s) for s in names(args.sets)}
    rows = []
    for name in names(args.models):
        hf, qp, pp = MODELS[name]
        model = SentenceTransformer(hf, device="cuda")
        model.max_seq_length = 512
        params = sum(p.numel() for p in model.parameters())
        row = {"model": name, "params_m": round(params / 1e6), "fp32_mb": round(params * 4 / 1e6)}
        for s, (corpus, queries, qrels) in data.items():
            dids, qids = list(corpus), list(queries)
            d = model.encode([pp + corpus[i] for i in dids], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
            q = model.encode([qp + queries[i] for i in qids], batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
            row[s] = score(topk(q, d, qids, dids), qrels)
            print(f"[{name}] {s}: {row[s]}", flush=True)
        cpu = SentenceTransformer(hf, device="cpu")
        torch.set_num_threads(4)
        sample = random.Random(0).sample(list(data[next(iter(data))][1].values()), args.cpu_queries)
        ms = []
        for text in sample:
            t = time.perf_counter()
            cpu.encode([qp + text], normalize_embeddings=True)
            ms.append((time.perf_counter() - t) * 1000)
        row["cpu_query_ms_p50"] = round(statistics.median(ms), 1)
        rows.append(row)
        del model, cpu
        torch.cuda.empty_cache()
    out = Path(args.out or LOGS / "rag-bench.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sets = list(data)
    print("\n| model | params | " + " | ".join(f"{s} nDCG@10 / R@1 / R@10" for s in sets) + " | CPU ms/query |")
    print("|---|---:|" + "---:|" * len(sets) + "---:|")
    for r in rows:
        cells = " | ".join(f"{r[s]['ndcg@10']:.3f} / {r[s]['r@1']:.3f} / {r[s]['r@10']:.3f}" for s in sets)
        print(f"| {r['model']} | {r['params_m']}M | {cells} | {r['cpu_query_ms_p50']} |")


class TfliteEncoder:
    """The exported / fetched encoder graphs of one variant: shortest graph for queries, longest for passages.
    Tokenises with the HF tokenizer (the device rule matches it — see `tokcheck`) and L2-normalises the output."""

    def __init__(self, directory: str, variant: str) -> None:
        from ai_edge_litert.interpreter import Interpreter
        from transformers import AutoTokenizer

        self.dir = ROOT / directory if not Path(directory).is_absolute() else Path(directory)
        meta = json.loads(next(p for p in (self.dir / "encoder.json", self.dir / "e5_tflite.json") if p.exists())
                          .read_text(encoding="utf-8"))
        self.hf = meta["model"]
        _, self.qp, self.pp = next(v for v in MODELS.values() if v[0] == self.hf)
        self.tok = AutoTokenizer.from_pretrained(self.hf)
        self.graphs = {g["length"]: name for name, g in meta["graphs"].items() if name.endswith(f"_{variant}.tflite")}
        self.inputs = {L: meta["graphs"][n]["inputs"] for L, n in self.graphs.items()}
        self.mb = {L: meta["graphs"][n]["mb"] for L, n in self.graphs.items()}
        self.lq, self.lp = min(self.graphs), max(self.graphs)
        self.runners = {L: Interpreter(model_path=str(self.dir / self.graphs[L]), num_threads=4).get_signature_runner()
                        for L in {self.lq, self.lp}}
        self.norms: list[float] = []

    def embed(self, texts: list[str], length: int, ms: list[float] | None = None):
        import numpy as np

        runner, inputs = self.runners[length], self.inputs[length]
        enc = self.tok(texts, padding="max_length", truncation=True, max_length=length, return_tensors="np")
        out = []
        for i in range(len(texts)):
            t = time.perf_counter()
            feed = {inputs[0]: enc["input_ids"][i:i + 1].astype(np.int32)}
            if len(inputs) > 1:  # e5 exports take a mask; the litert-community embeddinggemma graph does not
                feed[inputs[1]] = enc["attention_mask"][i:i + 1].astype(np.int32)
            res = runner(**feed)
            if ms is not None:
                ms.append((time.perf_counter() - t) * 1000)
            out.append(next(iter(res.values()))[0])
        e = np.stack(out)
        self.norms.append(float(np.linalg.norm(e, axis=1).mean()))
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def cmd_tflite(args) -> None:
    from sentence_transformers import SentenceTransformer

    data = {s: load_set(s) for s in names(args.sets)}
    rows = []
    ref = None
    for variant in names(args.variants):
        enc = TfliteEncoder(args.dir, variant)
        if ref is None:
            ref = SentenceTransformer(enc.hf, device="cuda")
        qp, pp, lq, lp = enc.qp, enc.pp, enc.lq, enc.lp
        row = {"variant": variant, "query_len": lq, "passage_len": lp, "mb": enc.mb}
        qms, pms = [], []
        for s, (corpus, queries, qrels) in data.items():
            dids, qids = list(corpus), list(queries)
            de = enc.embed([pp + corpus[i] for i in dids], lp, pms)
            qe = enc.embed([qp + queries[i] for i in qids], lq, qms)
            row[s] = score(topk(qe, de, qids, dids), qrels)
            # parity against the HF/ST encoder on the same truncation the graphs see
            n = min(200, len(qids))
            ref.max_seq_length = lq
            hq = ref.encode([qp + queries[i] for i in qids[:n]], normalize_embeddings=True)
            ref.max_seq_length = lp
            hd = ref.encode([pp + corpus[i] for i in dids[:n]], normalize_embeddings=True)
            row[s]["cos_query_min"] = round(float((hq * qe[:n]).sum(1).min()), 5)
            row[s]["cos_passage_min"] = round(float((hd * de[:n]).sum(1).min()), 5)
            print(f"[tflite {variant}] {s}: {row[s]}", flush=True)
        row["query_ms_p50"] = round(statistics.median(qms), 2)
        row["passage_ms_p50"] = round(statistics.median(pms), 2)
        row["raw_output_norm"] = round(statistics.mean(enc.norms), 4)  # 1.0 = the graph already L2-normalises
        rows.append(row)
        print(f"[tflite {variant}] query {lq} tok p50 {row['query_ms_p50']} ms, passage {lp} tok p50 {row['passage_ms_p50']} ms", flush=True)
    out = Path(args.out or LOGS / "rag-tflite.jsonl")
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_fetch_litert(args) -> None:
    """Pull litert-community's generic (non-SoC) graphs into the layout `tflite` reads: <dir>/encoder.json + files."""
    import shutil
    from ai_edge_litert.interpreter import Interpreter
    from huggingface_hub import hf_hub_download

    d = ROOT / args.dir
    d.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "source": args.repo, "graphs": {}}
    for length in names(args.lengths):
        name = args.pattern.format(length=length)
        dst = d / name
        if not dst.exists():
            shutil.copy(hf_hub_download(args.repo, name), dst)
        sig = Interpreter(model_path=str(dst)).get_signature_list()
        inputs = next(iter(sig.values()))["inputs"]
        meta["graphs"][name] = {"length": int(length), "inputs": inputs, "mb": round(dst.stat().st_size / 1e6, 1)}
        print(f"[fetch] {name} inputs={inputs} {meta['graphs'][name]['mb']} MB")
    (d / "encoder.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


# What the device sample needs per embedder: graphs (query = shortest, passages = longest of the variant), the
# SentencePiece file, and the id rule runTextEncoder applies (mirrors device_ids / the gemma convention).
DEVICE_ENCODERS = {
    "e5-small": {"dir": "External/rag-work/multilingual-e5-small", "variant": "i8",
                 "tokenizer": ("intfloat/multilingual-e5-small", "sentencepiece.bpe.model"),
                 "spec": {"idOffset": 1, "unkId": 3, "bosId": 0, "eosId": 2, "padId": 1, "useMask": True}},
    "embeddinggemma": {"dir": "External/rag-work/embeddinggemma-300m", "variant": "mixed-precision",
                       "tokenizer": ("google/embeddinggemma-300m", "tokenizer.model"),
                       "spec": {"idOffset": 0, "unkId": -1, "bosId": 2, "eosId": 1, "padId": 0, "useMask": False}},
}
INDEX_DIR = ROOT / "External" / "rag-work" / "index"


def cmd_index(args) -> None:
    """Sample index: KorQuAD-dev paragraphs, a fixed question sample, and per-embedder passage embeddings
    (fp16 binary: int32 N, int32 D, then N*D little-endian halves) made with the same passage graph the
    device would use. Also prints the desktop hit@1/@3 the device run should reproduce."""
    import shutil
    import numpy as np
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_dataset("KorQuAD/squad_kor_v1", split="validation")
    pid: dict[str, int] = {}
    for r in rows:
        pid.setdefault(r["context"], len(pid))
    passages = list(pid)
    (INDEX_DIR / "passages.json").write_text(json.dumps({"passages": passages}, ensure_ascii=False), encoding="utf-8")
    sample = random.Random(0).sample(list(rows), args.questions)
    questions = [{"id": r["id"], "question": r["question"], "gold": pid[r["context"]], "answers": r["answers"]["text"]}
                 for r in sample]
    (INDEX_DIR / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions),
                                               encoding="utf-8")
    for name in names(args.embedders):
        cfg = DEVICE_ENCODERS[name]
        enc = TfliteEncoder(cfg["dir"], cfg["variant"])
        pe = enc.embed([enc.pp + p for p in passages], enc.lp).astype(np.float16)
        with (INDEX_DIR / f"{name}.f16").open("wb") as f:
            f.write(np.array(pe.shape, dtype="<i4").tobytes())
            f.write(pe.astype("<f2").tobytes())
        qe = enc.embed([enc.qp + q["question"] for q in questions], enc.lq)
        top = np.argsort(-(qe @ pe.astype(np.float32).T), axis=1)[:, :3]
        hit1 = float(np.mean([top[i, 0] == q["gold"] for i, q in enumerate(questions)]))
        hit3 = float(np.mean([q["gold"] in top[i] for i, q in enumerate(questions)]))
        tok_src = hf_hub_download(*cfg["tokenizer"])
        shutil.copy(tok_src, INDEX_DIR / f"{name}.{cfg['tokenizer'][1]}")
        shutil.copy(enc.dir / enc.graphs[enc.lq], INDEX_DIR / f"{name}.query.tflite")
        meta = {"embedder": name, "queryGraph": f"{name}.query.tflite", "queryLength": enc.lq, "passageLength": enc.lp,
                "tokenizer": f"{name}.{cfg['tokenizer'][1]}", "index": f"{name}.f16", "queryPrefix": enc.qp,
                "dims": int(pe.shape[1]), **cfg["spec"], "desktopHit1": hit1, "desktopHit3": hit3}
        (INDEX_DIR / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[index] {name}: {pe.shape} -> {name}.f16 ({(INDEX_DIR / f'{name}.f16').stat().st_size / 1e6:.2f} MB), "
              f"desktop hit@1 {hit1:.2f} hit@3 {hit3:.2f} on {len(questions)} questions")


def _norm_answer(s: str) -> str:
    """KorQuAD-style: lower-case, drop whitespace and punctuation (Korean answers compare per character)."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"^\s*(답|정답)\s*[:：]\s*", "", s)
    return re.sub(r"[\s\W_]+", "", s)


def answer_scores(pred: str, golds: list[str]) -> tuple[float, float, float]:
    """(exact match, character F1, gold-contained) — best over the gold answers."""
    from collections import Counter
    p = _norm_answer(pred)
    em = f1 = contains = 0.0
    for g in (_norm_answer(x) for x in golds):
        em = max(em, float(p == g))
        contains = max(contains, float(bool(g) and g in p))
        common = sum((Counter(p) & Counter(g)).values())
        if common:
            prec, rec = common / len(p), common / len(g)
            f1 = max(f1, 2 * prec * rec / (prec + rec))
    return em, f1, contains


def cmd_grade_device(args) -> None:
    qs = {q["id"]: q for q in (json.loads(l) for l in (INDEX_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l)}
    print("| run | n | hit@1 | hit@3 | embed ms p50 | retrieve ms p50 | TTFT s p50 | LLM ms p50 | EM | F1 | contains |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for path in args.results:
        rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
        ok = [r for r in rows if not r.get("error")]
        sc = [answer_scores(r["answer"], qs[r["id"]]["answers"]) for r in ok]
        med = lambda k: round(statistics.median(r[k] for r in ok), 1) if ok else float("nan")
        mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else float("nan")
        print(f"| {Path(path).parent.name} | {len(ok)}/{len(rows)} | {mean([r['hit1'] for r in ok])} | {mean([r['hit3'] for r in ok])} "
              f"| {med('embedMs')} | {med('retrieveMs')} | {med('ttftSeconds')} | {med('llmMs')} "
              f"| {mean([s[0] for s in sc])} | {mean([s[1] for s in sc])} | {mean([s[2] for s in sc])} |")


def device_ids(spm, text: str) -> list[int]:
    """What the device must feed: raw SentencePiece ids shifted by XLM-R's fairseq offset (+1; spm <unk> 0 -> 3),
    wrapped in <s>=0 ... </s>=2; pad with <pad>=1, attention_mask 0 on pads."""
    return [0] + [p + 1 if p != 0 else 3 for p in spm.encode(text)] + [2]


def cmd_tokcheck(args) -> None:
    import sentencepiece
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    hf = MODELS[args.model][0]
    spm = sentencepiece.SentencePieceProcessor(model_file=hf_hub_download(hf, "sentencepiece.bpe.model"))
    tok = AutoTokenizer.from_pretrained(hf)
    texts = []
    for s in SETS:
        corpus, queries, _ = load_set(s)
        texts += list(queries.values()) + list(corpus.values())
    texts = random.Random(0).sample(texts, min(args.n, len(texts))) + ["서울역 출구", "3번 버스 30m 앞!", "X-ray 2층", "  공백  여러개 ", "ⓐ①ｆｕｌｌ"]
    bad = [(t, tok(t)["input_ids"], device_ids(spm, t)) for t in texts if tok(t)["input_ids"] != device_ids(spm, t)]
    print(f"tokcheck {hf}: {len(texts) - len(bad)}/{len(texts)} identical")
    for t, a, b in bad[:5]:
        print(f"  MISMATCH {t[:60]!r}\n    hf  {a[:24]}\n    dev {b[:24]}")


def _selfcheck() -> None:
    m = score({"q": ["a", "b", "c"]}, {"q": {"b"}})
    assert m["r@1"] == 0 and m["r@5"] == 1 and m["mrr@10"] == 0.5 and abs(m["ndcg@10"] - 1 / math.log2(3)) < 1e-4, m
    assert answer_scores("답: 세종대왕.", ["세종대왕"]) == (1.0, 1.0, 1.0)
    em, f1, c = answer_scores("훈민정음을 만든 사람은 세종대왕입니다", ["세종대왕"])
    assert em == 0 and c == 1 and 0 < f1 < 1
    assert answer_scores("모름", ["1443년"]) == (0.0, 0.0, 0.0)


def main() -> int:
    _selfcheck()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--models", default=",".join(MODELS))
    b.add_argument("--sets", default=",".join(SETS))
    b.add_argument("--cpu-queries", type=int, default=30)
    b.add_argument("--out", default="")
    t = sub.add_parser("tflite")
    t.add_argument("--dir", required=True)
    t.add_argument("--variants", default="fp32,w8,i8")
    t.add_argument("--sets", default="korquad,autorag")
    t.add_argument("--out", default="")
    k = sub.add_parser("tokcheck")
    k.add_argument("--model", default="e5-small")
    k.add_argument("--n", type=int, default=3000)
    f = sub.add_parser("fetch-litert")
    f.add_argument("--repo", default="litert-community/embeddinggemma-300m")
    f.add_argument("--model", default="google/embeddinggemma-300m")
    f.add_argument("--pattern", default="embeddinggemma-300M_seq{length}_mixed-precision.tflite")
    f.add_argument("--lengths", default="256,512")
    f.add_argument("--dir", default="External/rag-work/embeddinggemma-300m")
    x = sub.add_parser("index")
    x.add_argument("--embedders", default=",".join(DEVICE_ENCODERS))
    x.add_argument("--questions", type=int, default=100)
    g = sub.add_parser("grade-device")
    g.add_argument("results", nargs="+", help="results.jsonl pulled by Run-LiteRtLmAndroidRagSmokeTest.ps1")
    args = ap.parse_args()
    {"bench": cmd_bench, "tflite": cmd_tflite, "tokcheck": cmd_tokcheck, "fetch-litert": cmd_fetch_litert,
     "index": cmd_index, "grade-device": cmd_grade_device}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
