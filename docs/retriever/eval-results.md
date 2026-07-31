# Retriever eval results (2026-07-31 21:54, commit 886cc8f)

## SciFact 基础矩阵
```
config                               MRR      R@5     R@10     R@20   Recall   n_scored/n_total
-----------------------------------------------------------------------------------------------
retr_scifact_bm25.toml            0.6078   0.6962   0.7562   0.8112   0.8613   300/300
retr_scifact_strong-bm25.toml     0.6105   0.7071   0.7604   0.8134   0.8624   300/300
retr_scifact_hybrid-rrf.toml      0.7066   0.8126   0.8627   0.9167   0.9466   300/300
retr_scifact_hybrid-convex.toml   0.7233   0.8016   0.8572   0.9142   0.9449   300/300
retr_scifact_granite-dense.toml   0.7179   0.7890   0.8627   0.9067   0.9400   300/300
retr_scifact_query2doc.toml       0.6485   0.7535   0.8241   0.8759   0.9220   300/300
retr_scifact_hyde.toml            0.7001   0.7979   0.8617   0.9193   0.9610   300/300
retr_scifact_decompose.toml       0.5584   0.6503   0.7094   0.7823   0.8432   300/300
```

## NQ 基础矩阵
```
config                          MRR      R@5     R@10     R@20   Recall   n_scored/n_total
------------------------------------------------------------------------------------------
retr_nq_bm25.toml            0.8157   0.6246   0.7598   0.8508   0.9166   2000/2000
retr_nq_strong-bm25.toml     0.8153   0.6159   0.7476   0.8391   0.9098   2000/2000
retr_nq_hybrid-rrf.toml      0.8866   0.6914   0.8220   0.9132   0.9709   2000/2000
retr_nq_hybrid-convex.toml   0.9213   0.7314   0.8489   0.9260   0.9725   2000/2000
retr_nq_granite-dense.toml   0.9327   0.6879   0.7863   0.8587   0.9201   2000/2000
retr_nq_query2doc.toml       0.8416   0.6517   0.7760   0.8673   0.9378   2000/2000
retr_nq_hyde.toml            0.9055   0.6707   0.7734   0.8505   0.9125   2000/2000
retr_nq_decompose.toml       0.7682   0.5655   0.7050   0.8075   0.8920   2000/2000
```

## 2Wiki 基础矩阵
```
config                             MRR      R@5     R@10     R@20   Recall   n_scored/n_total
---------------------------------------------------------------------------------------------
retr_2wiki_bm25.toml            0.9434   0.6653   0.7151   0.7421   0.7621   2000/2000
retr_2wiki_strong-bm25.toml     0.9580   0.6766   0.7222   0.7468   0.7678   2000/2000
retr_2wiki_hybrid-rrf.toml      0.9828   0.7069   0.7460   0.7724   0.7995   2000/2000
retr_2wiki_hybrid-convex.toml   0.9871   0.7090   0.7494   0.7720   0.8004   2000/2000
retr_2wiki_granite-dense.toml   0.9794   0.7040   0.7418   0.7684   0.8057   2000/2000
retr_2wiki_query2doc.toml       0.9356   0.6797   0.7331   0.7625   0.7871   2000/2000
retr_2wiki_hyde.toml            0.9512   0.6913   0.7361   0.7651   0.8084   2000/2000
retr_2wiki_decompose.toml       0.5702   0.4716   0.5491   0.6506   0.7610   2000/2000
```

## SciFact Convex α 曲线
```
config                               MRR      R@5     R@10     R@20   Recall   n_scored/n_total
-----------------------------------------------------------------------------------------------
retr_scifact_strong-bm25.toml     0.6105   0.7071   0.7604   0.8134   0.8624   300/300
retr_scifact_convex-a10.toml      0.6353   0.7221   0.7877   0.8384   0.9359   300/300
retr_scifact_convex-a30.toml      0.6781   0.7812   0.8224   0.8908   0.9466   300/300
retr_scifact_hybrid-convex.toml   0.7233   0.8016   0.8572   0.9142   0.9449   300/300
retr_scifact_convex-a70.toml      0.7309   0.8137   0.8748   0.9201   0.9467   300/300
retr_scifact_convex-a90.toml      0.7208   0.7962   0.8669   0.9101   0.9500   300/300
retr_scifact_granite-dense.toml   0.7179   0.7890   0.8627   0.9067   0.9400   300/300
```

## NQ Convex α 曲线
```
config                          MRR      R@5     R@10     R@20   Recall   n_scored/n_total
------------------------------------------------------------------------------------------
retr_nq_strong-bm25.toml     0.8153   0.6159   0.7476   0.8391   0.9098   2000/2000
retr_nq_convex-a30.toml      0.8680   0.6767   0.8190   0.9195   0.9733   2000/2000
retr_nq_hybrid-convex.toml   0.9213   0.7314   0.8489   0.9260   0.9725   2000/2000
retr_nq_convex-a70.toml      0.9381   0.7210   0.8335   0.9133   0.9720   2000/2000
retr_nq_granite-dense.toml   0.9327   0.6879   0.7863   0.8587   0.9201   2000/2000
```

## SciFact 超参敏感性
```
config                             MRR      R@5     R@10     R@20   Recall   n_scored/n_total
---------------------------------------------------------------------------------------------
retr_scifact_bm25.toml          0.6078   0.6962   0.7562   0.8112   0.8613   300/300
retr_scifact_strong-bm25.toml   0.6105   0.7071   0.7604   0.8134   0.8624   300/300
retr_scifact_bm25-k09b30.toml   0.6119   0.6979   0.7562   0.8127   0.8596   300/300
retr_scifact_bm25-k09b75.toml   0.6099   0.6896   0.7646   0.8094   0.8579   300/300
retr_scifact_bm25-k12b40.toml   0.6095   0.6987   0.7562   0.8105   0.8646   300/300
retr_scifact_bm25-k12b75.toml   0.6054   0.6937   0.7662   0.8172   0.8613   300/300
retr_scifact_bm25-k20b40.toml   0.6089   0.6993   0.7562   0.8180   0.8613   300/300
retr_scifact_bm25-k20b75.toml   0.6067   0.6904   0.7596   0.8053   0.8579   300/300
retr_scifact_rrf-k10.toml       0.6122   0.7029   0.7629   0.8134   0.8641   300/300
retr_scifact_rrf-k30.toml       0.6117   0.7029   0.7596   0.8127   0.8641   300/300
retr_scifact_rrf-k100.toml      0.6117   0.7029   0.7596   0.8127   0.8641   300/300
```

## 显著性检验
```
comparison (ON vs OFF)     metric                              mean_on mean_off    delta    p_value       n
-------------------------------------------------------------------------------------------------------------
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/software/local/languages/miniforge3/envs/python-3.11.15/lib/python3.11/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/software/local/languages/miniforge3/envs/python-3.11.15/lib/python3.11/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/software/local/languages/miniforge3/envs/python-3.11.15/lib/python3.11/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
Exception ignored in: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>
BrokenPipeError: [Errno 32] Broken pipe
comparison (ON vs OFF)     metric                              mean_on mean_off    delta    p_value       n
-------------------------------------------------------------------------------------------------------------
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/software/local/languages/miniforge3/envs/python-3.11.15/lib/python3.11/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/software/local/languages/miniforge3/envs/python-3.11.15/lib/python3.11/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/software/local/languages/miniforge3/envs/python-3.11.15/lib/python3.11/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
Exception ignored in: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>
BrokenPipeError: [Errno 32] Broken pipe
comparison (ON vs OFF)     metric                              mean_on mean_off    delta    p_value       n
-------------------------------------------------------------------------------------------------------------
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/software/local/languages/miniforge3/envs/python-3.11.15/lib/python3.11/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/software/local/languages/miniforge3/envs/python-3.11.15/lib/python3.11/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/software/local/languages/miniforge3/envs/python-3.11.15/lib/python3.11/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
Exception ignored in: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>
BrokenPipeError: [Errno 32] Broken pipe
```

## 显著性检验(修正版)
```
### scifact  (delta=ON-OFF; p>0.05 视为无显著差异)
comparison(ON vs OFF)        metric                              mean_on mean_off    delta   p_value    n
query2doc vs strong-bm25     retriever.core.document_mrr          0.6485   0.6105  +0.0380    0.0265  300
query2doc vs strong-bm25     retriever.core.document_recall_at_10   0.8241   0.7604  +0.0637    0.0005  300
hyde vs granite-dense        retriever.core.document_mrr          0.7001   0.7179  -0.0178    0.1476  300
hyde vs granite-dense        retriever.core.document_recall_at_10   0.8617   0.8627  -0.0010    1.0000  300
decompose vs strong-bm25     retriever.core.document_mrr          0.5584   0.6105  -0.0521    0.0018  300
decompose vs strong-bm25     retriever.core.document_recall_at_10   0.7094   0.7604  -0.0509    0.0132  300
hybrid-rrf vs strong-bm25    retriever.core.document_mrr          0.7066   0.6105  +0.0961    0.0000  300
hybrid-rrf vs strong-bm25    retriever.core.document_recall_at_10   0.8627   0.7604  +0.1023    0.0000  300
hybrid-convex vs granite-dense retriever.core.document_mrr          0.7233   0.7179  +0.0054    0.6889  300
hybrid-convex vs granite-dense retriever.core.document_recall_at_10   0.8572   0.8627  -0.0056    0.7054  300
strong-bm25 vs bm25          retriever.core.document_mrr          0.6105   0.6078  +0.0027    0.7792  300
strong-bm25 vs bm25          retriever.core.document_recall_at_10   0.7604   0.7562  +0.0042    0.6629  300
### nq  (delta=ON-OFF; p>0.05 视为无显著差异)
comparison(ON vs OFF)        metric                              mean_on mean_off    delta   p_value    n
query2doc vs strong-bm25     retriever.core.document_mrr          0.8416   0.8153  +0.0262    0.0001 2000
query2doc vs strong-bm25     retriever.core.document_recall_at_10   0.7760   0.7476  +0.0284    0.0000 2000
hyde vs granite-dense        retriever.core.document_mrr          0.9055   0.9327  -0.0272    0.0000 2000
hyde vs granite-dense        retriever.core.document_recall_at_10   0.7734   0.7863  -0.0128    0.0003 2000
decompose vs strong-bm25     retriever.core.document_mrr          0.7682   0.8153  -0.0472    0.0000 2000
decompose vs strong-bm25     retriever.core.document_recall_at_10   0.7050   0.7476  -0.0426    0.0000 2000
hybrid-rrf vs strong-bm25    retriever.core.document_mrr          0.8866   0.8153  +0.0713    0.0000 2000
hybrid-rrf vs strong-bm25    retriever.core.document_recall_at_10   0.8220   0.7476  +0.0744    0.0000 2000
hybrid-convex vs granite-dense retriever.core.document_mrr          0.9213   0.9327  -0.0114    0.0047 2000
hybrid-convex vs granite-dense retriever.core.document_recall_at_10   0.8489   0.7863  +0.0626    0.0000 2000
strong-bm25 vs bm25          retriever.core.document_mrr          0.8153   0.8157  -0.0004    0.9181 2000
strong-bm25 vs bm25          retriever.core.document_recall_at_10   0.7476   0.7598  -0.0122    0.0024 2000
### 2wiki  (delta=ON-OFF; p>0.05 视为无显著差异)
comparison(ON vs OFF)        metric                              mean_on mean_off    delta   p_value    n
query2doc vs strong-bm25     retriever.core.document_mrr          0.9356   0.9580  -0.0224    0.0000 2000
query2doc vs strong-bm25     retriever.core.document_recall_at_10   0.7331   0.7222  +0.0109    0.0002 2000
hyde vs granite-dense        retriever.core.document_mrr          0.9512   0.9794  -0.0282    0.0000 2000
hyde vs granite-dense        retriever.core.document_recall_at_10   0.7361   0.7418  -0.0056    0.1009 2000
decompose vs strong-bm25     retriever.core.document_mrr          0.5702   0.9580  -0.3878    0.0000 2000
decompose vs strong-bm25     retriever.core.document_recall_at_10   0.5491   0.7222  -0.1731    0.0000 2000
hybrid-rrf vs strong-bm25    retriever.core.document_mrr          0.9828   0.9580  +0.0248    0.0000 2000
hybrid-rrf vs strong-bm25    retriever.core.document_recall_at_10   0.7460   0.7222  +0.0238    0.0000 2000
hybrid-convex vs granite-dense retriever.core.document_mrr          0.9871   0.9794  +0.0076    0.0001 2000
hybrid-convex vs granite-dense retriever.core.document_recall_at_10   0.7494   0.7418  +0.0076    0.0023 2000
strong-bm25 vs bm25          retriever.core.document_mrr          0.9580   0.9434  +0.0146    0.0000 2000
strong-bm25 vs bm25          retriever.core.document_recall_at_10   0.7222   0.7151  +0.0071    0.0015 2000
```
