from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_MODEL


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    latencies = {}

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    latencies["chunking_s"] = round(time.time() - t0, 2)
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({latencies['chunking_s']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        latencies["enrichment_s"] = round(time.time() - t0, 2)
        print(f"  ✓ Enriched {len(enriched)} chunks ({latencies['enrichment_s']:.1f}s)", flush=True)
    else:
        latencies["enrichment_s"] = 0.0
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    latencies["indexing_s"] = round(time.time() - t0, 2)
    print(f"  ✓ Indexed ({latencies['indexing_s']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    latencies["reranker_setup_s"] = round(time.time() - t0, 2)
    print(f"  ✓ Reranker ready ({latencies['reranker_setup_s']:.1f}s)", flush=True)

    return search, reranker, latencies


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    from config import OPENAI_API_KEY
    has_valid_key = bool(
        OPENAI_API_KEY
        and not OPENAI_API_KEY.startswith("sk-or-v1-YOUR")
        and not OPENAI_API_KEY.startswith("sk-...")
        and not OPENAI_API_KEY.startswith("your_")
    )
    if has_valid_key and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model=OPENAI_MODEL, messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, pipeline_latencies: dict | None = None):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    t_queries = time.time()
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
    query_latency = round(time.time() - t_queries, 2)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_latency = round(time.time() - t0, 2)
    print(f"  ✓ RAGAS done ({eval_latency:.1f}s)", flush=True)

    latencies = {
        **(pipeline_latencies or {}),
        "queries_generation_s": query_latency,
        "ragas_evaluation_s": eval_latency,
    }
    total_time = sum(latencies.values())
    latencies["total_pipeline_s"] = round(total_time, 2)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    # Latency Breakdown Report (Bonus +2)
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN REPORT (Bonus +2)")
    print("=" * 60)
    print(f"{'Pipeline Stage':<35} {'Duration':>15}")
    print("-" * 52)
    for step_name, dur in latencies.items():
        print(f"{step_name:<35} {dur:>12.2f}s")
    print("=" * 60)

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures, latency_breakdown=latencies)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, latencies = build_pipeline()
    evaluate_pipeline(search, reranker, latencies)
    print(f"\nTotal: {time.time() - start:.1f}s")
