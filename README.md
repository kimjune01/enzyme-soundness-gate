# Enzyme-JAX rewrite-soundness gate

A value-and-gradient soundness gate, and a proof-by-cases proto-test generator, for
[Enzyme-JAX](https://github.com/EnzymeAD/Enzyme-JAX)'s StableHLO rewrite patterns and AD
derivative rules. This is the process behind issues
[#2570](https://github.com/EnzymeAD/Enzyme-JAX/issues/2570) (LogSimplify domain narrowing)
and [#2571](https://github.com/EnzymeAD/Enzyme-JAX/issues/2571) (cbrt derivative NaN gradient).

## The contract is the oracle

Enzyme's stated contract: an optimized program computes the same VALUES, and
AD-after-optimization produces the same GRADIENT. That contract lets a finding self-adjudicate,
no taste call required:

- value divergence on an ordinary input -> bug, full stop;
- gradient divergence at a smooth point (the gradient is unique there) -> bug, full stop;
- divergence only at non-smooth ties (both subgradients valid), or only at inf/nan/fast-math
  (tacit precision relaxations) -> sanctioned residue, auto-filtered.

The gate hunts the first two and filters the residue. A "gap" that is known or accepted wastes a
maintainer's attention; only self-adjudicating divergences are surfaced.

## The seam

Per-pattern coverage in Enzyme-JAX is syntactic: the ~500 lit tests run
`enzymexlamlir-opt --enzyme-hlo-opt | FileCheck`, verifying a rewrite fires and emits the expected
IR text. They never run the program and never differentiate. The semantic + AD oracle exists
(`test/test_utils.py`: `recursive_check`, the HLOOpt-vs-Jax pipeline, `splatjvp`/`splatvjp`) but is
driven only by a handful of end-to-end models that rarely hit edge cases. So each rewrite is checked
to fire-and-look-right; almost none is checked in isolation to preserve value, and none to preserve
gradient. This fills that seam mechanically.

## Method: two stages

**1. Dynamic gate** (`harness.py`, `adrule_gate.py`). For each rule, reimplement the
(unoptimized, optimized) forms in JAX from the rule's lit test / source, enumerate an edge-input
lattice (sign classes, zero, inf, nan, denormals, magnitudes, ties), compare VALUE and GRADIENT
against the reference (the unoptimized program; autodiff for derivative rules), auto-filter the
residue, and surface self-adjudicating divergences. For value and smooth gradient the JAX
computation is the real-number semantics the rewrite claims to preserve, so a divergence here is a
real bug regardless of how the binary lowers it. Findings are reconciled as accept-sets and recorded
as a replayable hypothesis graph via [`abductor`](https://github.com/kimjune01/abductor).

**2. Static proto-test generator** (`domain_analysis.py`). A finite-class-cover analyzer that reads
a rule's expression structure and PREDICTS the witnessing input class and the verdict, with no
execution. It is mechanical proof by cases over a sign / special-value partition: a rewrite is
domain-unsound when the optimized form is undefined (NaN) somewhere the original is defined, and that
witness falls out of `domain(L) \ domain(R)`. It reproduces both findings from rule structure alone,
including the subtle constant-sign dependence in #2570 (negative constant breaks, positive constant
is sound). A "proto-test" is a conjecture (predicted from a reference model); it earns the name
"test" only after confirmation against the real implementation.

## Findings

- **#2570** `LogSimplify` (`EnzymeHLOOpt.cpp`): `log(a*a) -> 2*log(a)`, `log(pow(x,y)) -> y*log(x)`,
  and the constant `log(a*b)`/`log(a/b)` cases narrow the domain (real for `a != 0`, real only for
  `a > 0`), returning NaN on finite negative inputs.
- **#2571** `CbrtOp` derivative (`HLODerivatives.td`): the rule routes the gradient through
  `pow(x, -2/3)`, which is NaN for negative `x`, while `cbrt` is real and smooth there. Primal
  correct, gradient poisoned. Each ships with a one-line mutation of Enzyme's own test as the repro.

## Reproduce

```bash
pip install jax            # CPU is enough; soundness is a numerical question
python3 domain_analysis.py # static proof-by-cases; no JAX, no build, microseconds
python3 harness.py         # dynamic value+grad gate over the rewrite patterns
python3 adrule_gate.py     # dynamic gradient gate over the AD derivative rules
```

The static analyzer needs no JAX and no Enzyme build. The dynamic gate needs only JAX on CPU. The
gate reconciliation and hypothesis-graph recording use [`abductor`](https://github.com/kimjune01/abductor).

## Scope (stated honestly)

Sound for **domain / value / gradient** soundness over **finitely-partitionable** elementwise and
transcendental ops. Out of scope, by independent characterization rather than convenience:

- FP precision (overflow, underflow, rounding, denormals): sanctioned by fast-math; the analyzer
  models the real field, not the non-ring float behavior, so it is blind here, and the contract
  forgives exactly what it cannot see.
- gradient at ties (non-unique subgradient): no fact of the matter.
- structural / shape / aliasing soundness: a relational question, not a finite class cover.
- fractal / float-boundary regimes (e.g. `sin(1/x)` near 0): the ground truth itself aliases, so
  there is no well-posed soundness question.

Each excluded zone is named so it can be audited. Inside the stated fragment the case analysis is
exhaustive; outside it, the question is ill-posed or relaxed, not hidden.

## License

Dual-licensed, at your option, under
[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) (see `LICENSE`) and
[CC BY-SA-NS](https://june.kim/cc-by-sa-ns) (CC BY-SA 4.0 with a Network Services clause;
see `NOTICE`). Both are copyleft and close the SaaS loophole. Copyright (c) 2026 June Kim.
As sole copyright holder the author reserves the right to dual-license; outside contributions
are accepted only under a CLA that preserves this.
