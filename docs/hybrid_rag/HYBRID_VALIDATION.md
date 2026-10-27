# Original hybrid RAG validation

**Historical baseline:** these measurements predate the generation revision, reranker and article-lead expansion. See `GROUNDED_GENERATION_VALIDATION.md` for the revised pipeline.

Executed on the existing eight-shard Mac/PostgreSQL collection. No embeddings or HNSW indexes were regenerated.

## Coverage and functional checks

- PostgreSQL subset: 1,250,310 articles and 5,665,798 passages from shards 00000–00007.
- New contentless FTS5 index: 5,665,798 records, integrity check passed, approximately 3.12 GB. PostgreSQL passage IDs are FTS row IDs.
- All 150 question/answer pairs verified against actual PostgreSQL passage text and the corresponding snapshot chunk; 50 articles across 16 editorial domains.
- Notebook schema validated; all code cells executed with evaluation flags off.
- All three retrieval modes exercised against live indexes; cosine HNSW chosen by EXPLAIN ANALYZE.
- RRF combination, safe token quoting, cache-copy isolation, blank input rejection and UI callback checked.
- Both answer-fusion methods returned supported answers for Brazil’s capital, Abraham Lincoln and Adam Smith (six generated answers). These six tests do not establish answer accuracy across the question bank.

## Cross-domain retrieval check

32 questions: first and last question in each editorial domain. Retrieval mode order rotated by question. Application cache bypassed; OS/database caches were uncontrolled and included substantial warming. Values below are descriptive, not a controlled cold/warm comparison.

| Mode | Expected article hit@5 | Expected passage hit@5 | Median retrieval | p95 retrieval | Errors |
|---|---:|---:|---:|---:|---:|
| vector | 29/32 | 26/32 | 12.2 ms | 289.7 ms | 0 |
| bm25 | 25/32 | 21/32 | 1048.6 ms | 2505.5 ms | 1 |
| hybrid | 32/32 | 28/32 | 930.5 ms | 2595.2 ms | 0 |

Hybrid had no degraded-mode warnings in this 32-question run. BM25-only timed out on Q054; that failure counts as a miss. Its runtime is included in the latency summary. The separate boundary tests intentionally force timeouts to verify that degraded hybrid mode is visible.

These are expected-source lookup metrics, not semantic relevance judgments or exact ANN recall. A different passage can contain a correct answer. The positive demo set is not an unbiased sample of Wikipedia and was not held out from development. Full-corpus answer correctness and a 10-ms SLA are not claimed.

## Reproducible outputs

- `hybrid_domain_evaluation.json`: per-query IDs, source hits, ranks/titles, latency, errors and warnings.
- `hybrid_validation.json`: initial six-query smoke tests, six generated answers, query plan and UI verification.
- `hybrid_boundary_tests.json`: timeout, corpus mismatch and diversity checks.
- `hybrid_rag_questions_150.json`: question bank with source evidence and hashes.

In the notebook, set RUN_EVALUATION=True to evaluate all 150 questions across all modes. RUN_ANSWER_ABLATIONS=True compares both fusion methods at k=1/3/5 on a bounded 12-question subset by default.
