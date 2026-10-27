# RAG correction and 500-question evaluation

## Scope and outcome

This test uses the existing PostgreSQL and aligned BM25 collection from Wikipedia shards **00000–00007** only: 1,250,310 articles and 5,665,798 stored passages. No embeddings or HNSW indexes were regenerated.

The Darwin regression is corrected: **Alfred Russel Wallace** is returned for “With whom did Darwin jointly publish his theory of evolution?” in both fusion modes at k=1/3/5. The actual UI callback also passed. This is a specific verified correction, not a promise that every question is answered correctly.

## What changed and why

1. Rerank exact sentence excerpts across 30 fused candidates and introductory passages from up to ten matched articles. Previously the Darwin article could be retrieved while the relevant sentence did not reach the generator. Selection now operates at the evidence-sentence level. The original stored text and offsets remain inspectable.
2. Use a direct context/question prompt for Flan-T5-small and remove numeric source labels. An initial test caught the model returning labels such as “[1]”; that run was stopped, the prompt corrected, and the full final 500-question test restarted. Citation-label-only outputs are also rejected.
3. Keep the explicit 512-token input budget, 160-token output allowance, deterministic decoding and both fusion methods. Context packing exposes omissions and clipping. Intermediate model drafts never replace original evidence.
4. Preserve the complete original chunk beside every selected source excerpt in the UI. Every selected excerpt was checked as an exact substring at its recorded offsets, and every tested model input stayed within budget.

Sentence splitting remains heuristic and may omit needed neighbouring context. The reranker is not an entailment model. A source match does not prove that the answer follows from it.

## Test design

The 500 questions cover **400 distinct articles**. All expected source passages were checked against PostgreSQL and their original local Parquet shard, with source hashes recorded.

- **150 curated factual questions:** the previous development bank, 50 articles and 16 editorial subject groups. These are not held out from tuning.
- **175 named-article questions:** “What is X? Give a short description.” Each has an expected source article/passage but no independently adjudicated answer key. Title lookup is easier than open-domain reasoning.
- **175 description lookups:** identify a subject from a definition excerpt with its title removed. These reuse source wording, so they favour lexical retrieval and are not paraphrase tests.

The 350 new questions were deterministically sampled with seed 20261027 across all eight shards: 44 per shard for 00000–00005 and 43 per shard for 00006–00007. They were not used to adjust the correction. Eligibility filtering excludes lists, duplicate questions and unsuitable lead passages, so this is not an unbiased sample of all Wikipedia topics. Random candidate indices were processed in row order until each shard quota was filled, which also biases selection toward earlier eligible rows within each shard.

The corrected pipeline was frozen before the final run. All 500 queries used Hybrid retrieval, k=3 and Concatenate with the application's caches disabled. Vector, BM25 and RRF rankings were captured from each hybrid request; these are branch diagnostics, not independent end-to-end runs of all three retrieval modes. Both fusion modes and k=1/3/5 were separately exercised on five regression questions (30 combinations); they were not each evaluated on all 500 questions.

## Retrieval results

Counts below are **expected-source hits**, not ANN recall against exact nearest neighbours or human relevance judgments. A different source can also answer a question. Failures and timeouts count against the denominator.

| Group | N | Vector article @5 | BM25 article @5 | RRF article @5 | Focused article @3 | Focused gold passage @3 | Gold article in final context |
|---|---:|---:|---:|---:|---:|---:|---:|
| curated_factual | 150 | 139 | 97 | 133 | 137 | 123 | 137 |
| named_article | 175 | 131 | 163 | 168 | 166 | 160 | 166 |
| description_lookup | 175 | 111 | 122 | 139 | 148 | 148 | 148 |

Across all groups, the expected article reached the final context for **451/500 (90.2%)** queries, and the exact gold passage for **431/500 (86.2%)**. These mixed-difficulty totals do not replace the per-group results. Hybrid retrieval did not dominate vector-only rankings on every group; see the factual-query branch results.

## Generation results and short answers

Expected-answer alias matching ignores case, accents and punctuation and respects word boundaries. It can accept an incorrect sentence that includes the right name and reject a legitimate paraphrase; it is **not verified factual accuracy**. Named-article descriptions have no answer score because they lack an independently reviewed reference answer. The earlier factual keys also have limits: for example, the France northeast-border question is ambiguous and different passages describe borders differently; “So Paulo” versus “São Paulo” is counted as an alias failure even though it may be a spelling error rather than an entity-selection error. Human adjudication is still needed.

| Group | Alias-bearing answers | Expected alias visible in context | One-word answers | Abstentions |
|---|---:|---:|---:|---:|
| curated_factual | 104/150 | 138 | 71 | 2 |
| named_article | Not scored | Not scored | 4 | 0 |
| description_lookup | 93/175 | 91 | 51 | 0 |

For context, the previous full-chunk revision had an alias match in **87/150** curated answers, and the older short-phrase policy had **77/150**. Those earlier runs used the same development questions but are not a controlled causal comparison of individual changes.

Flan-T5-small can return one word even with a large output allowance. A name or capital question legitimately has a short answer. Removing an output cap does not make the model reliably conversational. Explicit prose prompts were tested: one Darwin response mixed Wallace with a claim about Huxley, and other responses answered the wrong question. That style was not enabled in the delivered UI. The notebook keeps concise generation rather than forcing unsupported detail. Six lightweight output-contract tests additionally confirmed that longer model-produced prose is accepted, empty evidence abstains, source offsets are preserved, and citation-label-only output is rejected; there is no hidden one-word or 16-word cap. For dependable conversational explanations, further fine-tuning or a stronger generator would need its own evaluation; Flan-T5-small remains the project generator here.

## Timing and failures

Application caches were disabled; OS/database caches were uncontrolled. Sequential requests on this Mac are not a concurrency or production SLA benchmark. BM25 runs alongside query embedding/vector search, so branch times must not be added as if sequential.

| Stage | Median ms | p95 ms |
|---|---:|---:|
| query_embedding_ms | 10.6 | 14.3 |
| ann_ms | 253.4 | 394.4 |
| bm25_ms | 847.0 | 3001.1 |
| rerank_ms | 605.1 | 959.2 |
| generation_ms | 113.1 | 345.7 |
| end_to_end_ms | 1609.6 | 3817.1 |

Completed without runtime error: **500/500**. Queries with explicit degraded-retrieval warnings: **26**. Runtime errors: **0**. No sub-10-ms end-to-end claim is made.

## Remaining failures to inspect

These examples are diagnostic failures; source-hit and alias metrics alone cannot establish correctness.

- **Q003 — Which country borders metropolitan France to the northeast?** Generated: 'France'. Expected alias: ['Germany']. Alias in input: True.
- **Q006 — What is Brazil's most populous city?** Generated: 'So Paulo'. Expected alias: ['São Paulo']. Alias in input: True.
- **Q007 — What is Australia's official name?** Generated: 'Australia'. Expected alias: ['Commonwealth of Australia']. Alias in input: False.
- **Q008 — Which island state is named in Australia's geographical definition?** Generated: 'New South Wales'. Expected alias: ['Tasmania']. Alias in input: True.
- **Q014 — What is another name for World War I?** Generated: 'World War I'. Expected alias: ['First World War']. Alias in input: True.
- **Q015 — Approximately how many soldiers died in World War I?** Generated: '74,187'. Expected alias: ['9 million']. Alias in input: True.
- **Q017 — In which region of Africa was Ancient Egypt located?** Generated: 'Northeast'. Expected alias: ['Northeast Africa']. Alias in input: True.
- **Q019 — What was the northern side in the American Civil War called?** Generated: 'Northern Department'. Expected alias: ['Union']. Alias in input: True.
- **Q020 — In which year did the American Civil War begin?** Generated: '1862'. Expected alias: ['1861']. Alias in input: True.
- **Q021 — In which year did the American Civil War end?** Generated: '1866'. Expected alias: ['1865']. Alias in input: True.
- **Q030 — Was Aristotle an Ancient Greek philosopher?** Generated: 'yes'. Expected alias: ['Ancient Greek philosopher']. Alias in input: True.
- **Q036 — Which poet was Ada Lovelace's father?** Generated: 'Medora'. Expected alias: ['Lord Byron']. Alias in input: True.

## Files and reproducibility

- `rag_test_questions_500.json` / `.md`: all 500 questions, group, expected source IDs, shard and evidence hashes.
- `rag_500_run/manifest.json`: notebook and question-bank hashes used for the run. `tested_notebook.ipynb` preserves that exact file; `delivery_verification.json` confirms the delivered inference code is identical. Delivery differences are documentation and removal of the failed prose-style UI option.
- `rag_500_run/rag_500_results.json`: per-question retrieval, answer, warnings and timing.
- `rag_500_run/rag_500_generation_traces.jsonl`: actual prompts, outputs and selected source windows.
- `rag_500_run/summary.json`: aggregated metrics.
- `rag_correction_regressions.json`: fusion/style experiments and UI regression evidence. Prose experiments are diagnostic only and are not enabled in the UI.
- `build_rag_500_questions.py`: reproducible source-verification and sampling script.
- `rag_answer_contract_tests.json`: output-contract checks.
- `evaluate_rag_500.py`: reproducible evaluator; see `HYBRID_RAG_GUIDE.md` for invocation. It executes setup cells only, without ingestion or index construction.

Reload `offline_wikipedia_pgvector_rag.ipynb`, restart the kernel and run cells in order. Keep ingestion, index-building and optional evaluation flags false for interactive use. A timestamped backup preserves the previous notebook.
