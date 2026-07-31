"""Frozen hypothesis construction (Graph 2.0 design §2.4).

The template is the primary form because it is deterministic and auditable; a QA2D-style
converter (Chen, Choi & Durrett, Findings of EMNLP 2021) is a pre-registered ablation, not the
default.

Claim-to-claim comparison must use these full sentences, never bare answer strings: an NLI
cross-encoder given "Paul" against "Apostle Paul" is being fed degenerate input, and the shared
question is what makes the pair well formed.
"""

HYPOTHESIS_TEMPLATE = 'The answer to the question "{question}" is {answer}.'


def build_hypothesis(question: str, answer: str) -> str:
    return HYPOTHESIS_TEMPLATE.format(question=question.strip(), answer=answer.strip())
