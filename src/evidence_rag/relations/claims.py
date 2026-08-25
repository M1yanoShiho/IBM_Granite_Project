"""Frozen hypothesis construction (Graph 2.0 design §2.4).

The template is the primary form because it is deterministic and auditable; a QA2D-style
converter (Chen, Choi & Durrett, Findings of EMNLP 2021) is a pre-registered ablation, not the
default.

Claim-to-claim comparison must use these full sentences, never bare answer strings: an NLI
cross-encoder given "Paul" against "Apostle Paul" is being fed degenerate input, and the shared
question is what makes the pair well formed.
"""

from collections.abc import Callable

HypothesisForm = Callable[[str, str], str]

HYPOTHESIS_TEMPLATE = 'The answer to the question "{question}" is {answer}.'
QUESTION_ANSWER_TEMPLATE = "{question}? {answer}."


def build_hypothesis(question: str, answer: str) -> str:
    return HYPOTHESIS_TEMPLATE.format(question=question.strip(), answer=answer.strip())


def build_question_answer(question: str, answer: str) -> str:
    """Ablation rung 1: the frozen template's meta frame removed, question surface kept.

    The frozen form asserts something ABOUT a question; this one puts the question and the
    answer side by side. Gate 0B-2 (R012) showed both arms collapsing on SUPPORTS while the
    answer string was verbatim present in the premise, which points at the hypothesis form
    rather than at the evidence. This rung changes exactly one thing so the collapse can be
    attributed to the meta frame or ruled out.
    """
    return QUESTION_ANSWER_TEMPLATE.format(
        question=question.strip().rstrip("?").strip(), answer=answer.strip()
    )


HYPOTHESIS_FORMS: dict[str, HypothesisForm] = {
    "template": build_hypothesis,
    "question_answer": build_question_answer,
}
