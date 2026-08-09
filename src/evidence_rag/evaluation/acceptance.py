"""Tiered acceptance for a selector arm: what a reading licenses, fixed before it is seen.

A single hard guard turns every result into pass or fail, and the failures then have to be
argued around. E2's -1% recall guard did exactly that: S1 broke it, and from then on the
findings downstream all carried "the guard is broken, but --". The guard was not wrong; a
one-line threshold simply cannot express a method whose value IS a trade, and the arguing
happened after the numbers were known, which is the worst possible moment to decide what counts.

So the trade space is pre-registered instead. Three tiers, thresholds chosen before the run:

  clear-pass  harm credibly down, every recall loss within the clean band
  tradeoff    harm credibly down, some loss in the band between clean and unacceptable --
              real, reportable, and NOT to be announced as success
  fail        harm not credibly down, or any loss past the outer band

"Credibly down" is the CI upper bound below zero, not the point estimate. A negative delta with
a CI straddling zero says the effect was not resolved, and S8 is the standing example of why
that distinction has to be mechanical: two null axes there could have been written up as
"no cost" and were instead reported as a bound.

Every failing condition is listed, not the first. A verdict that stops at one reason makes the
next iteration discover the second one, which is the same round-trip tax `--check-paths-only`
exists to remove.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

# Pre-registered bands, in absolute units (0.03 == 3 percentage points).
CLEAN_LOSS = 0.03
MAX_LOSS = 0.05

Tier = Literal["clear-pass", "tradeoff", "fail"]


@dataclass(frozen=True)
class AcceptanceVerdict:
    tier: Tier
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True for both accepting tiers. Read `tier` when the trade matters -- and it does:
        `tradeoff` is a result to report with its cost attached, never as a clean win."""

        return self.tier != "fail"


def classify(
    *,
    harm_delta: float,
    harm_ci_high: float,
    recall_losses: Mapping[str, float],
    clean_loss: float = CLEAN_LOSS,
    max_loss: float = MAX_LOSS,
) -> AcceptanceVerdict:
    """Judge one arm against the pre-registered bands.

    `harm_delta` and `harm_ci_high` are signed, lower-is-better, so a credible reduction is a CI
    upper bound strictly below zero. `recall_losses` maps a metric name to its LOSS as a positive
    number, because an arm is judged on every recall it could damage, not on one -- E2 watched
    required recall while 2Wiki supporting recall went unwatched, and an arm that trades one for
    the other would have read as clean.
    """

    if clean_loss > max_loss:
        raise ValueError("clean_loss must not exceed max_loss")
    if not recall_losses:
        raise ValueError("at least one recall metric must be judged")

    reasons: list[str] = []
    harm_credible = harm_ci_high < 0
    if not harm_credible:
        reasons.append(
            f"harm not credibly reduced: delta {harm_delta:+.4f}, CI upper {harm_ci_high:+.4f}"
            " does not clear zero"
        )
    # Sorted so the report is stable across runs; every offending metric is named, because
    # fixing the one that happened to be first only surfaces the next one a run later.
    for name, loss in sorted(recall_losses.items()):
        if loss > max_loss:
            reasons.append(f"{name} loss {loss:.4f} exceeds the outer band {max_loss:.4f}")

    if reasons:
        return AcceptanceVerdict(tier="fail", reasons=tuple(reasons))

    traded = sorted(name for name, loss in recall_losses.items() if loss > clean_loss)
    if traded:
        return AcceptanceVerdict(
            tier="tradeoff",
            reasons=tuple(
                f"{name} loss {recall_losses[name]:.4f} is between {clean_loss:.4f}"
                f" and {max_loss:.4f}"
                for name in traded
            ),
        )
    return AcceptanceVerdict(
        tier="clear-pass",
        reasons=(
            f"harm credibly reduced (CI upper {harm_ci_high:+.4f}) and every recall loss within"
            f" {clean_loss:.4f}",
        ),
    )
