# G310 seed13 screen report

**Status:** `RECIPE_SELECTED`
**Selected recipe:** `GR-C`

This is a model-val screen, not final Generator qualification and not held-out.

## Aggregate

### validation_answerable

| arm | groups | answer | coverage | cited | MiniCheck precision | MiniCheck recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 302 | 0.6260 | 0.9100 | 0.4725 | 0.7358 | 0.7207 | 0.0000 |
| GR-F | 302 | 0.5954 | 0.9380 | 0.5490 | 0.8409 | 0.8409 | 0.0000 |
| GR-C | 302 | 0.6188 | 0.9594 | 0.5674 | 0.8499 | 0.8493 | 0.0000 |

### validation_niah

| arm | groups | answer | coverage | cited | MiniCheck precision | MiniCheck recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 207 | 0.6901 | 0.9151 | 0.6197 | 0.8015 | 0.7995 | 0.0000 |
| GR-F | 207 | 0.6135 | 0.9434 | 0.5459 | 0.8099 | 0.8099 | 0.0000 |
| GR-C | 207 | 0.6391 | 0.9620 | 0.5659 | 0.8133 | 0.8133 | 0.0000 |

### validation_2wiki

| arm | groups | answer | coverage | cited | MiniCheck precision | MiniCheck recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 95 | 0.4863 | 0.8989 | 0.1516 | 0.5926 | 0.5491 | 0.0000 |
| GR-F | 95 | 0.5558 | 0.9263 | 0.5558 | 0.9084 | 0.9084 | 0.0000 |
| GR-C | 95 | 0.5747 | 0.9537 | 0.5705 | 0.9295 | 0.9277 | 0.0000 |

### train_unsupported_safety

| arm | groups | answer | coverage | cited | MiniCheck precision | MiniCheck recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 1044 | 0.0000 | 0.3008 | 0.0000 | 0.0000 | 0.0000 | 0.3008 |
| GR-F | 1044 | 0.0000 | 0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0019 |
| GR-C | 1044 | 0.0000 | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |
