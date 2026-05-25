"""Interaction carrying a full Beta belief instead of a scalar label."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from beta_swarm.belief import BetaBelief


class InteractionType(str, Enum):
    REPLY = "reply"
    VOTE = "vote"
    TRADE = "trade"
    COLLABORATION = "collaboration"


class BetaInteraction(BaseModel):
    """An interaction whose quality label is a :class:`BetaBelief` over ``v in [0,1]``.

    The legacy ``p`` is recovered as ``belief.mean``; the new information is the
    ``belief.concentration``. Payoff inputs (transfer, costs, reputation) carry
    over unchanged from the scalar model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)

    initiator: str = ""
    counterparty: str = ""
    interaction_type: InteractionType = InteractionType.REPLY
    accepted: bool = False

    # The belief over the continuous outcome.
    belief: BetaBelief = Field(default_factory=BetaBelief.uniform)

    # Payoff components (unchanged from the scalar formalism).
    tau: float = 0.0
    c_a: float = 0.0
    c_b: float = 0.0
    r_a: float = 0.0
    r_b: float = 0.0

    # Optional ground-truth continuous outcome in [0, 1], if known.
    ground_truth: Optional[float] = None
    causal_parents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ground_truth")
    @classmethod
    def _gt_in_unit(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"ground_truth must be in [0, 1], got {v}")
        return v

    # Convenience views recovering the legacy scalar interface ------------------
    @property
    def p(self) -> float:
        """Legacy point estimate: the belief mean."""
        return self.belief.mean

    @property
    def concentration(self) -> float:
        return self.belief.concentration

    def to_dict(self) -> dict:
        d = self.model_dump(mode="json", exclude={"belief"})
        d["belief"] = self.belief.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "BetaInteraction":
        data = dict(data)
        belief = data.pop("belief", None)
        kwargs: dict[str, Any] = dict(data)
        if belief is not None:
            kwargs["belief"] = BetaBelief.from_dict(belief)
        return cls(**kwargs)
