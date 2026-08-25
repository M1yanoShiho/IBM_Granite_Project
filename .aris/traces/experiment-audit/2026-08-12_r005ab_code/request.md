# R005A/R005B model/loss implementation audit request

Audit the feasibility and minimum correct implementation of the proposed pretrained NLI-path repair and pairwise objective in the current repository and pinned formal snapshot. Verify AutoModel versus AutoModelForSequenceClassification behavior, pooler/classifier parameters, label mapping, exact one-vs-rest initialization, dropout and parameter registration, fingerprint/checkpoint compatibility, pairwise loss normalization, fair batching and the tests required before any result split is read. Read only; do not edit files.
