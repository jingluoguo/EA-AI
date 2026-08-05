from __future__ import annotations

from dataclasses import dataclass


DIRECT_CONFIDENCE_THRESHOLD = 0.90
CONFIRM_CONFIDENCE_THRESHOLD = 0.50


@dataclass(frozen=True)
class UncertaintyPolicy:
    direct_threshold: float = DIRECT_CONFIDENCE_THRESHOLD
    confirm_threshold: float = CONFIRM_CONFIDENCE_THRESHOLD

    def band(self, score: float) -> str:
        if score >= self.direct_threshold:
            return "direct"
        if score >= self.confirm_threshold:
            return "confirm"
        return "unknown"

    def can_answer_directly(self, score: float) -> bool:
        return self.band(score) == "direct"

    def needs_confirmation(self, score: float) -> bool:
        return self.band(score) == "confirm"

    def needs_guided_learning(self, score: float) -> bool:
        return self.band(score) == "unknown"


DEFAULT_UNCERTAINTY_POLICY = UncertaintyPolicy()


def confidence_band(score: float, policy: UncertaintyPolicy = DEFAULT_UNCERTAINTY_POLICY) -> str:
    return policy.band(score)
