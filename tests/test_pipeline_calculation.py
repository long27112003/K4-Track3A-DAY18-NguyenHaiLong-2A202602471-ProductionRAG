from pathlib import Path

from src.m2_search import SearchResult
from src.pipeline import run_query


def test_advance_late_fee_is_prorated_once():
    policy = (Path(__file__).resolve().parents[1] / "data" / "tam_ung.md").read_text(
        encoding="utf-8"
    )
    result = SearchResult(
        text=policy,
        score=1.0,
        metadata={"source": "tam_ung.md", "parent_id": "tam_ung::parent_0"},
        method="hybrid",
    )

    class Search:
        parent_contexts = {"tam_ung::parent_0": policy}

        def search(self, _query):
            return [result]

    class Reranker:
        def rerank(self, _query, documents, top_k):
            return [result]

    answer, contexts = run_query(
        "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?",
        Search(),
        Reranker(),
    )
    assert "50,000 VNĐ" in answer
    assert "350,000 VNĐ" not in answer
    assert any("[Phép tính" in context for context in contexts)
