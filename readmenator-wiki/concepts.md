# Concepts

Second-brain semantic layer: nouns map atomically to file sets (EXTRACTED); verbs aggregate structural edges (INFERRED).

| Concept | Files | Mentions | Top Files |
|---------|-------|----------|-----------|
| `app` | 2 | 5 | `app.py`, `view.py` |
| `weight` | 2 | 4 | `app.py`, `view.py` |
| `grokking` | 2 | 3 | `app.py`, `view.py` |
| `orbit` | 2 | 3 | `app.py`, `view.py` |
| `predictions` | 2 | 3 | `app.py`, `view.py` |
| `structure` | 2 | 3 | `app.py`, `view.py` |
| `train` | 2 | 3 | `app.py`, `view.py` |
| `training` | 2 | 3 | `app.py`, `view.py` |
| `weights` | 2 | 3 | `app.py`, `view.py` |
| `analyzes` | 2 | 2 | `app.py`, `view.py` |
| `space` | 2 | 2 | `app.py`, `view.py` |

## Verb Edges

| Source | Verb | Target | Strength |
|--------|------|--------|----------|
| `analyzes` | `depends_on` | `app` | 1.00 |
| `analyzes` | `depends_on` | `grokking` | 1.00 |
| `analyzes` | `depends_on` | `orbit` | 1.00 |
| `analyzes` | `depends_on` | `predictions` | 1.00 |
| `analyzes` | `depends_on` | `space` | 1.00 |
| `analyzes` | `depends_on` | `structure` | 1.00 |
| `analyzes` | `depends_on` | `train` | 1.00 |
| `analyzes` | `depends_on` | `training` | 1.00 |
| `analyzes` | `depends_on` | `weight` | 1.00 |
| `analyzes` | `depends_on` | `weights` | 1.00 |
| `app` | `depends_on` | `analyzes` | 1.00 |
| `app` | `depends_on` | `grokking` | 1.00 |
| `app` | `depends_on` | `orbit` | 1.00 |
| `app` | `depends_on` | `predictions` | 1.00 |
| `app` | `depends_on` | `space` | 1.00 |
| `app` | `depends_on` | `structure` | 1.00 |
| `app` | `depends_on` | `train` | 1.00 |
| `app` | `depends_on` | `training` | 1.00 |
| `app` | `depends_on` | `weight` | 1.00 |
| `app` | `depends_on` | `weights` | 1.00 |
| `grokking` | `depends_on` | `analyzes` | 1.00 |
| `grokking` | `depends_on` | `app` | 1.00 |
| `grokking` | `depends_on` | `orbit` | 1.00 |
| `grokking` | `depends_on` | `predictions` | 1.00 |
| `grokking` | `depends_on` | `space` | 1.00 |
| `grokking` | `depends_on` | `structure` | 1.00 |
| `grokking` | `depends_on` | `train` | 1.00 |
| `grokking` | `depends_on` | `training` | 1.00 |
| `grokking` | `depends_on` | `weight` | 1.00 |
| `grokking` | `depends_on` | `weights` | 1.00 |
| `orbit` | `depends_on` | `analyzes` | 1.00 |
| `orbit` | `depends_on` | `app` | 1.00 |
| `orbit` | `depends_on` | `grokking` | 1.00 |
| `orbit` | `depends_on` | `predictions` | 1.00 |
| `orbit` | `depends_on` | `space` | 1.00 |
| `orbit` | `depends_on` | `structure` | 1.00 |
| `orbit` | `depends_on` | `train` | 1.00 |
| `orbit` | `depends_on` | `training` | 1.00 |
| `orbit` | `depends_on` | `weight` | 1.00 |
| `orbit` | `depends_on` | `weights` | 1.00 |
| `predictions` | `depends_on` | `analyzes` | 1.00 |
| `predictions` | `depends_on` | `app` | 1.00 |
| `predictions` | `depends_on` | `grokking` | 1.00 |
| `predictions` | `depends_on` | `orbit` | 1.00 |
| `predictions` | `depends_on` | `space` | 1.00 |
| `predictions` | `depends_on` | `structure` | 1.00 |
| `predictions` | `depends_on` | `train` | 1.00 |
| `predictions` | `depends_on` | `training` | 1.00 |
| `predictions` | `depends_on` | `weight` | 1.00 |
| `predictions` | `depends_on` | `weights` | 1.00 |

## Dialectic Prompts

- Thesis: `analyzes` centralizes 2 files; Antithesis: `app` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `analyzes` centralizes 2 files; Antithesis: `grokking` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `analyzes` centralizes 2 files; Antithesis: `orbit` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `analyzes` centralizes 2 files; Antithesis: `predictions` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `analyzes` centralizes 2 files; Antithesis: `space` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `analyzes` centralizes 2 files; Antithesis: `structure` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `analyzes` centralizes 2 files; Antithesis: `train` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `analyzes` centralizes 2 files; Antithesis: `training` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `analyzes` centralizes 2 files; Antithesis: `weight` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `analyzes` centralizes 2 files; Antithesis: `weights` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
