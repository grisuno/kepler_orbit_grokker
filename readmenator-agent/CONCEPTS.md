# Concepts

Nouns map atomically to file sets (EXTRACTED); verbs aggregate structural edges (INFERRED).

- `app` | files=2 | mentions=5 | `app.py`, `view.py`
- `weight` | files=2 | mentions=4 | `app.py`, `view.py`
- `grokking` | files=2 | mentions=3 | `app.py`, `view.py`
- `orbit` | files=2 | mentions=3 | `app.py`, `view.py`
- `predictions` | files=2 | mentions=3 | `app.py`, `view.py`
- `structure` | files=2 | mentions=3 | `app.py`, `view.py`
- `train` | files=2 | mentions=3 | `app.py`, `view.py`
- `training` | files=2 | mentions=3 | `app.py`, `view.py`
- `weights` | files=2 | mentions=3 | `app.py`, `view.py`
- `analyzes` | files=2 | mentions=2 | `app.py`, `view.py`
- `space` | files=2 | mentions=2 | `app.py`, `view.py`

## Verb Edges

- `analyzes` --depends_on--> `app` (strength 1.00)
- `analyzes` --depends_on--> `grokking` (strength 1.00)
- `analyzes` --depends_on--> `orbit` (strength 1.00)
- `analyzes` --depends_on--> `predictions` (strength 1.00)
- `analyzes` --depends_on--> `space` (strength 1.00)
- `analyzes` --depends_on--> `structure` (strength 1.00)
- `analyzes` --depends_on--> `train` (strength 1.00)
- `analyzes` --depends_on--> `training` (strength 1.00)
- `analyzes` --depends_on--> `weight` (strength 1.00)
- `analyzes` --depends_on--> `weights` (strength 1.00)
- `app` --depends_on--> `analyzes` (strength 1.00)
- `app` --depends_on--> `grokking` (strength 1.00)
- `app` --depends_on--> `orbit` (strength 1.00)
- `app` --depends_on--> `predictions` (strength 1.00)
- `app` --depends_on--> `space` (strength 1.00)
- `app` --depends_on--> `structure` (strength 1.00)
- `app` --depends_on--> `train` (strength 1.00)
- `app` --depends_on--> `training` (strength 1.00)
- `app` --depends_on--> `weight` (strength 1.00)
- `app` --depends_on--> `weights` (strength 1.00)
- `grokking` --depends_on--> `analyzes` (strength 1.00)
- `grokking` --depends_on--> `app` (strength 1.00)
- `grokking` --depends_on--> `orbit` (strength 1.00)
- `grokking` --depends_on--> `predictions` (strength 1.00)
- `grokking` --depends_on--> `space` (strength 1.00)
- `grokking` --depends_on--> `structure` (strength 1.00)
- `grokking` --depends_on--> `train` (strength 1.00)
- `grokking` --depends_on--> `training` (strength 1.00)
- `grokking` --depends_on--> `weight` (strength 1.00)
- `grokking` --depends_on--> `weights` (strength 1.00)
- `orbit` --depends_on--> `analyzes` (strength 1.00)
- `orbit` --depends_on--> `app` (strength 1.00)
- `orbit` --depends_on--> `grokking` (strength 1.00)
- `orbit` --depends_on--> `predictions` (strength 1.00)
- `orbit` --depends_on--> `space` (strength 1.00)
- `orbit` --depends_on--> `structure` (strength 1.00)
- `orbit` --depends_on--> `train` (strength 1.00)
- `orbit` --depends_on--> `training` (strength 1.00)
- `orbit` --depends_on--> `weight` (strength 1.00)
- `orbit` --depends_on--> `weights` (strength 1.00)
- `predictions` --depends_on--> `analyzes` (strength 1.00)
- `predictions` --depends_on--> `app` (strength 1.00)
- `predictions` --depends_on--> `grokking` (strength 1.00)
- `predictions` --depends_on--> `orbit` (strength 1.00)
- `predictions` --depends_on--> `space` (strength 1.00)
- `predictions` --depends_on--> `structure` (strength 1.00)
- `predictions` --depends_on--> `train` (strength 1.00)
- `predictions` --depends_on--> `training` (strength 1.00)
- `predictions` --depends_on--> `weight` (strength 1.00)
- `predictions` --depends_on--> `weights` (strength 1.00)

## Dialectic

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
