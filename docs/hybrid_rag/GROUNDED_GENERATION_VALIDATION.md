# Earlier grounded generation revision — 27 October 2026

**Historical result:** superseded by the focused-evidence correction and `RAG_500_VALIDATION.md`.

The notebook still uses Flan-T5-small for every generated answer. Original PostgreSQL chunks, question and source titles enter its prompt. The runtime does not look up answers in the 150-question bank.

## Changes and reasons

- Removed the shortest-phrase prompt, 16-word rejection and verbatim-answer gate. These caused terse responses and rejected valid paraphrases. The prompt requests complete sentences; output allowance is now 160 tokens, with deterministic four-beam decoding. The model can still choose a fragment. We do not force length or pad answers with unsupported facts.
- Added the locally cached MS MARCO MiniLM-L6-v2 cross-encoder to rerank 30 fused candidates plus first passages from up to ten matched articles. The new PostgreSQL B-tree `(article_id, char_start)` makes these lead lookups inexpensive. No embeddings or HNSW indexes were rebuilt; both retrieval branches remain restricted to shards 00000–00007.
- Replaced heuristic lexical sentence windows with original chunks. Concatenate packs whole chunks within 512 input tokens, omitting later chunks that do not fit and explicitly clipping an oversized first chunk. More selected chunks do not imply more visible chunks.
- Generate from evidence sends each selected original chunk separately to Flan. It then synthesizes from original passages whose intermediate answer was not unknown. Intermediate generated claims are not injected as facts. This is a conservative evidence-filter-and-synthesize implementation, not synthesis of unverified model drafts. It can still discard useful evidence or omit it during final packing.
- UI exposes exact prompts, model outputs, reranker scores, stage timings and omitted/clipped passage IDs. Sources identify supplied context, not automated verification of every claim.

## Full 150-question evaluation

Hybrid retrieval, reranking, k=3, Concatenate. The same cached raw candidate pools were reused for the final revision, with live PostgreSQL lead lookups. The baseline used the previous live retrieval and short-answer generation code. These are development-set diagnostics, not an independent benchmark or semantic correctness judgments.

| Diagnostic | Result |
|---|---:|
| Previous answers containing an expected alias | 77 / 150 |
| Revised answers containing an expected alias | 87 / 150 |
| Revised visible context containing an expected alias | 141 / 150 |

Alias matching ignores case, accents and punctuation and checks word boundaries. It can reward a wrong sentence containing the right name and reject a valid paraphrase. These figures must not be described as verified accuracy. The complete per-question record is `grounded_generation_validation.json`.

**This does not resolve all 150 questions.** Many failures remain despite useful evidence being visible. The model also often returns a noun phrase despite a complete-sentence instruction. The revision fixes the generation plumbing and increases evidence visibility, but it does not make Flan-T5-small a reliable general-purpose answerer. Achieving the requested quality needs further model evaluation and potentially task-specific fine-tuning; a stronger generator is another option only if the project permits it.

## Functional verification

- Notebook schema validated and all code cells executed with ingestion/evaluation flags off.
- Real UI callback completed and controls were re-enabled.
- All 150 final generation runs stayed within 512 input tokens; each visible passage was verified as original source text or its exact prefix.
- Both fusion methods at k=1/3/5 returned Brasília for the Brazil-capital smoke test. Generate from evidence made one call per selected passage plus the final synthesis call. This one-question ablation is functional coverage, not accuracy validation of both methods on all 150.
- Smoke-test timings are in `grounded_fusion_smoke_tests.json`. First uncached request took about 2.68 seconds, including about 2.02 seconds for BM25, 0.52 seconds reranking and 0.13 seconds generation. Later requests reused caches; they are not comparable uncached speed measurements. No 10-ms end-to-end claim is made.

## Use

Reload the updated project notebook, restart its kernel, and run cells in order. Keep `RUN_INGEST=False`, `BUILD_INDEX=False`, `RUN_EVALUATION=False`, and `RUN_ANSWER_ABLATIONS=False` for interactive use. Existing databases and embeddings are reused. Open **Exact Flan-T5-small prompts and outputs** below an answer to inspect what the generator actually received.

## Examples still requiring attention

- **Q002: What is the official name of France?** Model: 'France is officially the French Republic.'. Expected alias(es): French Republic.
- **Q003: Which country borders metropolitan France to the northeast?** Model: 'France.'. Expected alias(es): Germany.
- **Q006: What is Brazil's most populous city?** Model: 'Salto Grande.'. Expected alias(es): São Paulo.
- **Q031: With whom did Darwin jointly publish his theory of evolution?** Model: 'John Maynard Smith.'. Expected alias(es): Alfred Russel Wallace.
- **Q056: Which branch of calculus concerns instantaneous rates of change?** Model: 'The former'. Expected alias(es): differential calculus.
