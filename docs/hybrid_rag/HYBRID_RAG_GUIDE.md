# Offline hybrid Wikipedia RAG

## What changed

The notebook now queries the existing **eight-shard collection only**: files `train-00000-of-00041.parquet` through `train-00007-of-00041.parquet`, from English Wikipedia `20231101.en`. Coverage is **1,250,310 articles and 5,665,798 passages**. The historical PostgreSQL schema name contains `full`, but the collection is not the complete Wikipedia dataset.

No embeddings were regenerated and no HNSW index was rebuilt. The old pilot and the older full-Wikipedia SQLite BM25 index are not used by this notebook.

| Component | Implementation | Purpose |
|---|---|---|
| Passage store | PostgreSQL `rag_vectors.rag_wiki_bge_full_v1.chunks` | Canonical source text, offsets, URLs, IDs and embeddings |
| Semantic retrieval | BGE-small-en-v1.5, normalized 384-dimensional embeddings; `halfvec(384)` | Match meaning and paraphrases |
| Vector index | Existing `chunks_half_hnsw`, cosine; m=16, ef_construction=128 | Approximate nearest-neighbour search |
| Lexical retrieval | New SQLite FTS5 contentless index, Porter/unicode61 tokenizer, BM25 | Match names, rare terms, identifiers and literal wording |
| Candidate fusion | Equal-weight reciprocal rank fusion, constant 60 | Combine ranks without mixing incompatible score scales |
| Evidence reranking | Local MS MARCO MiniLM-L6-v2 cross-encoder | Rank fused candidates and article introductions by relevance |
| Generation | Local Flan-T5-small | Generate from original passage text, with explicit context budgets and prompt inspection |

The BM25 file is `data/wikipedia_pgvector_8shards_bm25.sqlite` inside the IITB-DL project. It contains **exactly the same 5,665,798 PostgreSQL passage IDs** and is approximately **3.12 GB**. Its creation, optimization and verification took approximately **5 minutes 44 seconds** on this run. It indexes the title and body but does not duplicate source text for display; the application fetches displayed evidence from PostgreSQL.

## Why a separate BM25 sidecar?

The pre-existing full-Wikipedia SQLite index had different chunks and included all 41 shards. Fusing its IDs with the PostgreSQL index would mix document boundaries and corpus coverage. The new sidecar streams only the frozen PostgreSQL subset and uses `chunks.id` as its FTS row ID.

SQLite FTS5 implements BM25. PostgreSQL's built-in `ts_rank` is not BM25, and this implementation does not label it as such. The sidecar avoids installing another database extension and is a pragmatic offline, single-user design. It introduces a synchronization responsibility; it is not a transactionally unified PostgreSQL search service.

## Query flow

1. Validate that PostgreSQL reports a completed eight-shard subset and a valid cosine HNSW index. Compare the lexical file's schema, source/configuration fingerprint, completed flag and document count with PostgreSQL.
2. Encode a query locally with pinned BGE revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. Apply the BGE query prefix, normalize the output and cast it to `halfvec(384)` for cosine search.
3. Fetch up to **50 HNSW candidates**, using `hnsw.ef_search=200`.
4. Concurrently run **BM25** for up to **50 lexical candidates**. Title weight is 2 and body weight 1. Tokenize, remove question stopwords, quote terms and combine them using OR. SQLite's negative BM25 score sorts ascending: a lower number is better.
5. Fuse ranks by `RRF(id) = sum(1 / (60 + rank_in_branch))`. Missing candidates contribute zero. Similarity and RRF scores are not confidence probabilities. Candidate counts and ranks remain visible in the UI.
6. Keep 30 fused candidates and fetch the first stored passage from up to ten matched articles using `chunks_article_start_idx`, a B-tree on `(article_id, char_start)`. This adds definitions and introductory facts without expanding corpus coverage.
7. Split retrieved passages and article leads into source sentences and rerank those exact excerpts using local `cross-encoder/ms-marco-MiniLM-L6-v2`, pinned at `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Select k=1/3/5 distinct passages, cap two per article, and preserve offsets and full original text. This resolves the tested Darwin failure where article-level relevance did not bring the answer sentence into context. Sentence splitting is heuristic and can lose surrounding context.
8. Supply the selected source excerpts to Flan-T5-small using a direct context/question prompt. Numeric source labels were removed after tests found the model returning labels instead of answers. Concatenate packs excerpts into 512 input tokens; omitted/clipped excerpts are reported. Generate from evidence separately processes each excerpt, then synthesizes from original excerpts whose intermediate response was not unknown. Intermediate generated claims are not injected as facts.
9. Generate deterministically with four beams and up to 160 new tokens. There is no shortest-phrase instruction, 16-word rejection or exact copied-answer gate. A citation-label-only output is rejected. Short factual answers remain common. Prose-prompt experiments produced fluent but wrong responses, so verbosity is not forced. The UI exposes actual prompts, source excerpts, full original chunks and stage outputs. Sources identify supplied context, not verified entailment citations.

Retrieval RRF and answer fusion are independent controls. The UI offers Hybrid / Vector only / BM25 only, 1 / 3 / 5 passages, and both answer-fusion methods.

## Run the notebook

1. Open `offline_wikipedia_pgvector_rag.ipynb` in the IITB-DL project. Reload the file and restart the kernel if an older pilot UI is still open.
2. Select the existing Miniforge Python kernel with PyArrow, psycopg, pgvector, sentence-transformers, transformers and ipywidgets installed.
3. Keep `RUN_INGEST=False`, `BUILD_INDEX=False`, `RUN_EVALUATION=False` and `RUN_ANSWER_ABLATIONS=False` for normal use.
4. Run cells in order. Startup must report **shards 00000–00007**, **1,250,310 articles**, and **5,665,798 passages**. It must not report pilot coverage.
5. Use the widget, or call:

```python
result = ask_pg(
    "Who was the sixteenth president of the United States?",
    k=3,
    method="concatenate",
    retrieval_mode="hybrid",
    use_cache=False,
)
print(result["answer"])
print(result["warnings"])
print(result["timings"])
```

`RAG_PG_DSN` can override the connection. The default is `host=localhost dbname=rag_vectors`. Models are loaded locally; notebook execution does not download models or require network inference. Wikipedia links are citations, not runtime data fetches.

## Safeguards and limits

- Both branches search the same frozen eight-shard subset; the lexical constructor refuses a mismatched fingerprint. The builder acquires the ingestion advisory lock, requires the previously audited subset, verifies FTS document count/integrity and publishes its file atomically. Partial builds remain under `.building.sqlite` and can resume.
- PostgreSQL retrieval uses parameterized SQL and read-only transactions. SQLite opens read-only, and raw FTS operators supplied by a user are not executed. HTML is escaped and citation URLs are limited to HTTP(S).
- SQL retrieval has a 10-second statement timeout. BM25 has a 3-second execution budget. A BM25 timeout in Hybrid mode is explicitly shown as degraded vector-only retrieval; BM25-only mode raises a timeout. Other failures are not silently swallowed.
- The query cache is bounded to 128 entries and includes retrieval mode and k. Returned objects are copied. Benchmarks bypass the application cache; database and operating-system caches still affect latency.
- BM25 may be much slower for broad OR queries than HNSW. Parallel branch timing overlaps, so stage times should not simply be added. The notebook reports retrieval time, answer time and end-to-end time separately. There is **no verified sub-10-ms SLA** for this corpus.
- HNSW is approximate; half-precision storage introduces rounding. No new precision reduction or binary quantization was added.
- The reranker is cached locally in `models/ms-marco-MiniLM-L6-v2`. Its manifest records the pinned revision. Sharing the notebook now also requires sharing or separately downloading this model. Runtime uses local files only.
- Flan-T5-small is a substantial quality bottleneck: it can select the wrong entity even with the correct fact in its input. Prompts encourage grounded prose but do not enforce it. Removing the span gate permits legitimate paraphrases and also means unsupported claims can appear. Retrieved text may contain instructions. Abstention is not proof that the corpus lacks an answer.
- Corpus data reflects November 2023, not current Wikipedia. A query can be answered only when useful evidence is present and retrieved. Subject groups in the question bank are editorial labels, not balanced dataset categories.

## Evaluation and 150 questions

The final notebook section contains 150 practice questions over **50 confirmed articles**, spanning **16 subject groups**. `hybrid_rag_questions_150.json` contains each question, expected answer aliases, source article and passage IDs, shard name, evidence quote and the SHA-256 hash of its original passage text. Every question was checked against an actual PostgreSQL passage, whose text matched the pinned source chunk. These are positive development questions, not an independent test set and not 150 guaranteed successful model answers.

`RUN_EVALUATION=True` compares all 150 questions across Vector / BM25 / Hybrid, with application caching disabled. It reports expected-article hit@5, expected-passage hit@5, p50/p95 retrieval latency, timeouts and warnings. A different passage can legitimately answer a question, so expected-passage hit is a strict diagnostic rather than a full relevance judgment. It is not ANN recall against an exact vector-search baseline.

`RUN_ANSWER_ABLATIONS=True` compares k=1/3/5 and both answer-fusion methods on the first 12 questions by default. It records exact match, token F1, citations, abstentions, model calls, token budgets and timing. Answer aliases are limited; lexical scores need human review for valid paraphrases. See `HYBRID_VALIDATION.md` for measured results from this implementation session.

For a reliable production assessment, add independent negative/unanswerable questions, temporal traps, ambiguous queries, rare entities and adversarial source content. Hold these out from tuning and test realistic concurrent load.

## Maintenance and sharing

The lexical build worker is `data/pgvector_full_job/build_hybrid_lexical.py`; its log/status are `lexical_build.log` and `lexical_status.json`. Its configuration and checkpoints protect same-version resumes. Existing final files are validated and reused, not silently overwritten.

Treat the PostgreSQL subset and lexical index as one versioned release. If passage IDs/text/chunking or corpus scope change, invalidate the sidecar and rebuild it under a new file before publishing. This notebook deliberately does not resume ingestion or claim all 41 shards are complete. Maintain backups before a corpus migration.

To share: preserve PostgreSQL passage IDs in the dump, include the matching SQLite sidecar, notebook, question bank and local model/version information. Recipients restore PostgreSQL with a compatible pgvector version and rebuild HNSW as part of logical restoration; they do not regenerate embeddings. Include Wikipedia attribution and the applicable reuse licenses. A multi-user deployment would additionally need authentication, connection pooling, access controls, coordinated corpus publishing, monitoring and recovery testing.

## Primary references

- [SQLite FTS5: BM25, rank and contentless tables](https://www.sqlite.org/fts5.html)
- [pgvector: HNSW and hybrid search](https://github.com/pgvector/pgvector)
- [Elastic: reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [BGE-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)

## Generation revision validation

See `RAG_500_VALIDATION.md` for the latest 500-question test and Darwin regression. `GROUNDED_GENERATION_VALIDATION.md` records the previous revision and is historical. Earlier retrieval measurements in `HYBRID_VALIDATION.md` predate reranking and article-lead expansion; they are not end-to-end performance claims for this revision. The answer key never enters the inference path.

## Reproduce the 500-question test

Run the packaged evaluator with the same Miniforge environment:

```bash
python docs/hybrid_rag/evaluate_rag_500.py \
  --notebook offline_wikipedia_pgvector_rag.ipynb \
  --questions docs/hybrid_rag/rag_test_questions_500.json \
  --output results/rag_500_new_run
```

Use a fresh output directory. `--limit 5` runs a smoke test; omit it for all 500. The evaluator records a manifest, progress, per-query scores, exact prompts and candidate snapshots. It executes setup cells only and does not ingest, embed, or rebuild indexes. Application caches are disabled for measurements; database/OS caches remain uncontrolled. Both vector and BM25 branch rankings are captured from the same hybrid request, not benchmarked as independent end-to-end modes.

The suite contains 150 curated factual development questions plus 175 named-article and 175 masked-description retrieval diagnostics sampled from 350 additional articles. They are different difficulty groups and must be reported separately. All sampled gold passages were matched against both the local Parquet shard and PostgreSQL text. Named-article questions have source targets but no independently adjudicated answer key. Description questions reuse source wording and favour lexical retrieval; they do not establish paraphrase performance.
