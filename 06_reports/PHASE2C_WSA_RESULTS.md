# Phase 2c-WS-A Results — does CER's advantage survive a real LLM compiler?

**Scope:** real Qwen3 LLM compiles the symbolic spec from retrieved evidence (vLLM, temp=0, generations cached). Retrieval over the purpose-built v2 CER corpus; the LLM emits typed fields, never a PRB number. Offline (no closed-loop DRL).

## Verdict: field-aware CER advantage **SURVIVES** real-LLM compilation

**Deterministic v2 reference (router, not LLM):** ordinary margin-acc 0.15 / under 0.44; field-CER margin-acc 0.88 / under 0.00.

## Per (model, retriever): margin accuracy / unsafe under-reservation / citation validity

| Model | Retr | Arm | Margin acc | Under-rsv | Cite valid | Validity | Recall@5 |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | bge | Ordinary RAG+LLM | 0.42 | 0.42 | 1.00 | 1.00 | 0.34 |
| Qwen3-1.7B | bge | State-aware+LLM | 0.74 | 0.26 | 1.00 | 1.00 | 0.50 |
| Qwen3-1.7B | bge | Field-CER+LLM | 0.91 | 0.00 | 1.00 | 1.00 | 0.64 |
| Qwen3-1.7B | bge | Field-CER+LLM+verifier | 0.91 | 0.00 | 1.00 | 1.00 | 0.64 |
| Qwen3-1.7B | tfidf | Ordinary RAG+LLM | 0.42 | 0.00 | 1.00 | 1.00 | 0.18 |
| Qwen3-1.7B | tfidf | State-aware+LLM | 0.88 | 0.02 | 1.00 | 1.00 | 0.52 |
| Qwen3-1.7B | tfidf | Field-CER+LLM | 0.88 | 0.00 | 1.00 | 1.00 | 0.48 |
| Qwen3-1.7B | tfidf | Field-CER+LLM+verifier | 0.88 | 0.00 | 1.00 | 1.00 | 0.48 |
| Qwen3-4B | bge | Ordinary RAG+LLM | 0.42 | 0.42 | 1.00 | 1.00 | 0.34 |
| Qwen3-4B | bge | State-aware+LLM | 0.71 | 0.21 | 1.00 | 1.00 | 0.50 |
| Qwen3-4B | bge | Field-CER+LLM | 0.78 | 0.02 | 1.00 | 1.00 | 0.64 |
| Qwen3-4B | bge | Field-CER+LLM+verifier | 0.78 | 0.02 | 1.00 | 1.00 | 0.64 |
| Qwen3-4B | tfidf | Ordinary RAG+LLM | 0.42 | 0.00 | 1.00 | 1.00 | 0.18 |
| Qwen3-4B | tfidf | State-aware+LLM | 0.88 | 0.02 | 1.00 | 1.00 | 0.52 |
| Qwen3-4B | tfidf | Field-CER+LLM | 0.88 | 0.00 | 1.00 | 1.00 | 0.48 |
| Qwen3-4B | tfidf | Field-CER+LLM+verifier | 0.88 | 0.00 | 1.00 | 1.00 | 0.48 |
| Qwen3-14B | bge | Ordinary RAG+LLM | 0.42 | 0.42 | 1.00 | 1.00 | 0.34 |
| Qwen3-14B | bge | State-aware+LLM | 0.74 | 0.17 | 1.00 | 1.00 | 0.50 |
| Qwen3-14B | bge | Field-CER+LLM | 0.72 | 0.02 | 1.00 | 1.00 | 0.64 |
| Qwen3-14B | bge | Field-CER+LLM+verifier | 0.72 | 0.02 | 1.00 | 1.00 | 0.64 |
| Qwen3-14B | tfidf | Ordinary RAG+LLM | 0.57 | 0.00 | 1.00 | 1.00 | 0.18 |
| Qwen3-14B | tfidf | State-aware+LLM | 0.88 | 0.02 | 1.00 | 1.00 | 0.52 |
| Qwen3-14B | tfidf | Field-CER+LLM | 0.88 | 0.00 | 1.00 | 1.00 | 0.48 |
| Qwen3-14B | tfidf | Field-CER+LLM+verifier | 0.88 | 0.00 | 1.00 | 1.00 | 0.48 |
| Qwen3-32B | bge | Ordinary RAG+LLM | 0.42 | 0.42 | 1.00 | 1.00 | 0.34 |
| Qwen3-32B | bge | State-aware+LLM | 0.74 | 0.17 | 1.00 | 1.00 | 0.50 |
| Qwen3-32B | bge | Field-CER+LLM | 0.72 | 0.02 | 1.00 | 1.00 | 0.64 |
| Qwen3-32B | bge | Field-CER+LLM+verifier | 0.72 | 0.02 | 1.00 | 1.00 | 0.64 |
| Qwen3-32B | tfidf | Ordinary RAG+LLM | 0.57 | 0.00 | 0.99 | 1.00 | 0.18 |
| Qwen3-32B | tfidf | State-aware+LLM | 0.88 | 0.02 | 1.00 | 1.00 | 0.52 |
| Qwen3-32B | tfidf | Field-CER+LLM | 0.88 | 0.00 | 1.00 | 1.00 | 0.48 |
| Qwen3-32B | tfidf | Field-CER+LLM+verifier | 0.88 | 0.00 | 1.00 | 1.00 | 0.48 |

## Field-CER − Ordinary gap (bootstrap 95% CI)

| Model | Retr | Δ margin-acc [CI] | Δ unsafe-rate [CI] |
|---|---|---|---|
| Qwen3-1.7B | bge | +0.481 [+0.406, +0.562] | -0.425 [-0.506, -0.350] |
| Qwen3-1.7B | tfidf | +0.450 [+0.375, +0.525] | +0.000 [+0.000, +0.000] |
| Qwen3-14B | bge | +0.300 [+0.169, +0.438] | -0.406 [-0.487, -0.325] |
| Qwen3-14B | tfidf | +0.300 [+0.231, +0.369] | +0.000 [+0.000, +0.000] |
| Qwen3-32B | bge | +0.300 [+0.169, +0.438] | -0.406 [-0.487, -0.325] |
| Qwen3-32B | tfidf | +0.306 [+0.231, +0.375] | +0.000 [+0.000, +0.000] |
| Qwen3-4B | bge | +0.356 [+0.237, +0.481] | -0.406 [-0.487, -0.325] |
| Qwen3-4B | tfidf | +0.450 [+0.375, +0.525] | +0.000 [+0.000, +0.000] |

## Interpretation (honest)

- All arms see the state summary, so this is the stringent test: does retrieval help *beyond* the LLM's own reasoning over state? The CER edge is expected to concentrate on **provenance/conflict** (degraded, conflict, noisy) where state alone cannot fix the policy.

- G-WSA gate: advantage_survives = **True**. Per (model,retriever) checks: {"Qwen3-1.7B/bge": {"field_margin_beats_ordinary": true, "field_under_le_ordinary": true, "verifier_safe": true, "verifier_cited": true}, "Qwen3-1.7B/tfidf": {"field_margin_beats_ordinary": true, "field_under_le_ordinary": true, "verifier_safe": true, "verifier_cited": true}, "Qwen3-4B/bge": {"field_margin_beats_ordinary": true, "field_under_le_ordinary": true, "verifier_safe": false, "verifier_cited": true}, "Qwen3-4B/tfidf": {"field_margin_beats_ordinary": true, "field_under_le_ordinary": true, "verifier_safe": true, "verifier_cited": true}, "Qwen3-14B/bge": {"field_margin_beats_ordinary": true, "field_under_le_ordinary": true, "verifier_safe": false, "verifier_cited": true}, "Qwen3-14B/tfidf": {"field_margin_beats_ordinary": true, "field_under_le_ordinary": true, "verifier_safe": true, "verifier_cited": true}, "Qwen3-32B/bge": {"field_margin_beats_ordinary": true, "field_under_le_ordinary": true, "verifier_safe": false, "verifier_cited": true}, "Qwen3-32B/tfidf": {"field_margin_beats_ordinary": true, "field_under_le_ordinary": true, "verifier_safe": true, "verifier_cited": true}}

- The verifier arm rejects any direct-numeric leak and fail-closes (C3), keeping unsafe under-reservation ~0 with grounded citations.

- Verifier edge case, not hidden: schema-valid but semantically wrong margin choices can still under-reserve. Observed verifier unsafe rates: Qwen3-4B/bge: 0.019 (3/160); Qwen3-14B/bge: 0.019 (3/160); Qwen3-32B/bge: 0.019 (3/160).


## Honest boundaries

- Controlled field-labelled corpus (not large-scale real specs). Generations cached at temp=0; `reliability_target` is control-inert under this solver. Closed-loop (M6) still deferred.

## Artifacts

- `05_figures/phase2c_wsa/fig_wsa_margin_acc_vs_size.png`
- `05_figures/phase2c_wsa/fig_wsa_under_rsv_vs_size.png`
- `05_figures/phase2c_wsa/fig_wsa_per_category.png`
- `04_results/phase2c_wsa/summary__*.json`, `generations/*.jsonl`
