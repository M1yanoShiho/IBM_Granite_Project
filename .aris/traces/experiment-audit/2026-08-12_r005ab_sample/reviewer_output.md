# Sample-design reviewer output

**Reviewer task:** `/root/amendment_sample_audit`  
**Mode:** independent read-only specialist audit  
**Project files edited:** no  
**Train-modelval read:** no

## Verdict

No P0 remains after adopting the recommended 2Wiki eligibility and precise isolation semantics. The proposed `64/96/64/128` roles are feasible.

## Recomputed eligibility

| Dataset | Condition | Queries | Components |
|---|---|---:|---:|
| NIAH | fresh after excluding original R005 components | 893 | 801 |
| NIAH | strict clean/cf pair in Top20 | 846 | 766 |
| NIAH | clean and cf both in TopK10 | 778 | 719 |
| 2Wiki | fresh; at least one official support in TopK10 | 2,677 | 2,078 |
| 2Wiki alternative A | every active supporting chunk in TopK10 | 2,498 | 1,942 |

Rule A filters 179 queries and removes 136 components entirely but does not improve actual full-gold-chain completeness. Its retained-query full-chain rate is 41.23%, versus 41.46% under the recommended rule B. The plan must therefore use B: at least one official support in TopK10 for eligibility, then protect every official support that is actually in TopK10.

## Frozen partition protocol

```text
protocol = selector-r005-amendment-v1
seed = 20260812

representative = argmin sha256(
  protocol + "\nrepresentative\n" + dataset + "\n" +
  component_id + "\n" + query_id + "\n" + seed)

component_order = sha256(
  protocol + "\ncomponent-order\n" + dataset + "\n" +
  component_id + "\n" + seed)
```

Components are sorted once and sliced `[0,64)`, `[64,160)`, `[160,224)`, `[224,352)`. The four roles have zero query/component overlap, zero `(dataset,query,evidence)` composite-identity overlap, zero cross-role text-pair-hash overlap and zero active-supervised-content overlap.

Raw document IDs, raw evidence IDs and candidate texts from the shared corpus are not globally disjoint; that is expected and must be reported rather than hidden. Inference and confidence intervals must use query/component as the unit.

## Combined frozen hashes

| Role | Assignment | Top20 | Top10 |
|---|---|---|---|
| R005A-fit | `342d90e4cf324c96a541cee6fce962be770b57ca52ab4c1b93f330c3c8af7491` | `301e910f67b78ec8c898049d89fd8841eb72b9bb8de3c9e3ae8400fb512ebde4` | `b9dcf6a3cabd1c69fdef4d9e42d699cd4ea9ba693b0b4ec3478138e386fcb759` |
| R005A-screen | `a992f1b76abfac727981a79cdaf20d034aa7f8641d8e5d4a7713c9ecfa5bd15c` | `e3ca8cbb3b5842516af7e5c3ca2c7d31858c85a810ad93a8480b7a9ce7e0228c` | `908880c6a31d58c5902e01a20b08eb770aec58d6cbcbe3146ea4494c5cd1ef5e` |
| R005B-fit | `d571a85eb91c3a3a9128124c0c627f6de27d604b5eb4a49510edf0a6fb573111` | `e46276b5e4a700cd313c81bff23470eaade78d5d17e82f866f29ebf39b1b1c55` | `efc358a08b1814337d60fa56d0f7c3757ccb5b9fc4d539708ba97fb9d38052fa` |
| R005B-confirm | `676cdb6cfb51626b98959a53ea7d81774b042cc64be0e4e6a09ba3046016832a` | `bb33ac9268a3405149cc2e428aa552323e0dc7be976a8f3164ead5e8ffc5fb60` | `3259c1bbe613d04f7f6023138451e725c1d5fbe12561e45ef57f4d44abb2a0fa` |

All-assignment SHA: `8b5551e8265ed67c76fd43cd8cf8886892d5a22e143af86411dc98ed64a19fb5`.

Canonical JSONL must use sorted keys, no extra spaces, UTF-8 without BOM or Unicode normalization, LF including the final line, and the exact assignment/candidate fields frozen in the plan. Any input, cardinality or hash mismatch must stop before training.

## Issues

- P0: none after the incorporated revision.
- P1: use 2Wiki rule B; use precise composite isolation; use query/component inference; verify every pinned hash and cardinality before training.
- P2: report true full-chain completeness conditionally, save actual assignment JSONL plus a rebuild verifier, and assert ranks exactly 1–20/1–10.
