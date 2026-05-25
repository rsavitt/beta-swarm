# beta-swarm

**Distributional safety with a full belief over a continuous outcome.**

A generalization of the soft-label formalism in
[`distributional-agi-safety`](https://github.com/swarm-ai-safety/swarm). Where
that framework carries a single scalar `p = P(v = +1)` per interaction, an agent
here carries a parameterized **distribution** over a continuous quality
`v ∈ [0, 1]`:

$$F = \mathrm{Beta}(\alpha, \beta)$$

That belief decomposes into two interpretable quantities:

| Quantity | Formula | Recovers / adds |
|---|---|---|
| **mean** | `α / (α + β)` | the old point estimate `p` |
| **concentration** | `α + β` | *new:* epistemic uncertainty |

This separates two situations the scalar formalism collapses into one number:

- **confident the interaction is mediocre** — mean ≈ 0.5, high concentration
- **uncertain whether it's great or terrible** — mean ≈ 0.5, low concentration

Both have `p = 0.5`. They differ entirely in their downside tail mass.

## What the generalization buys you

| Concept | Scalar `p` | beta-swarm |
|---|---|---|
| Quality label | `p ∈ [0, 1]` | `Beta(α, β)` over `v ∈ [0, 1]` |
| Payoff | `S_soft = p·s₊ − (1−p)·s₋` | `E_F[surplus(v)]` |
| Governance lever | threshold on `p` | tail mass `P(v < τ)`, Value-at-Risk |
| Quality gap | `E[p\|acc] − E[p\|rej]` | Wasserstein-1 / KL between accepted & rejected belief distributions |
| Calibration | Brier score | CRPS (continuous proper score) |

Every distributional quantity **reduces to the scalar one** when the surplus is
linear and the belief is sharp — beta-swarm is a strict superset. The payoff
only departs from the mean-based value when the surplus is **nonlinear**, at
which point the *shape* of the belief (not just its mean) moves the payoff. A
convex downside penalty, for instance, makes a diffuse "great-or-terrible"
belief strictly worse than a sharp "mediocre" one with the same mean.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from beta_swarm import BetaBelief, TailMassGovernor

confident_mediocre = BetaBelief.from_mean_concentration(mean=0.5, concentration=200)
uncertain          = BetaBelief.from_mean_concentration(mean=0.5, concentration=2)

# Same mean — a scalar-p policy cannot tell them apart:
assert confident_mediocre.mean == uncertain.mean == 0.5

# But the downside tail mass is completely different:
confident_mediocre.tail_mass(0.25)   # -> 0.000
uncertain.tail_mass(0.25)            # -> 0.250

# A tail-mass governor rejects the risky one and admits the reliable one:
gov = TailMassGovernor(tau=0.25, max_tail_mass=0.15)
gov.accepts(confident_mediocre)   # True
gov.accepts(uncertain)            # False
```

See [`examples/demo.py`](examples/demo.py) for the full side-by-side.

## Package layout

| Module | Contents |
|---|---|
| `beta_swarm.belief` | `BetaBelief` — mean, concentration, tail mass, quantiles, conjugate updates |
| `beta_swarm.payoff` | `DistributionalPayoffEngine` — `E_F[surplus(v)]`, configurable downside curvature |
| `beta_swarm.governance` | `TailMassGovernor`, `ValueAtRiskGovernor` — risk-aware admission |
| `beta_swarm.metrics` | Wasserstein/KL quality gap, distributional toxicity, CRPS |
| `beta_swarm.proxy` | `BetaProxyComputer` — observables → belief (mean *and* concentration) |
| `beta_swarm.interaction` | `BetaInteraction` — interaction carrying a `BetaBelief` |

## Tests

```bash
python -m pytest -q
```

## License

MIT.
