# Floats Are Not Rings: Proof-by-Cases Proto-Tests for Differentiate-after-Optimize Soundness

**DRAFT. June Kim (independent). 2026-06-22.**

> Status: working draft. Claims are scoped and the boundary is stated explicitly; the empirical
> sweep (Section 8) is in progress. Comments welcome.

## Abstract

An optimizing autodiff compiler makes two promises: an optimization preserves the program's VALUES,
and differentiation after optimization preserves its GRADIENT. We observe that for the elementwise
and transcendental rewrite and derivative rules of such a compiler, checking those promises is not a
search problem but a finite case analysis. A rewrite is a ring (field) identity applied as if floats
were the reals; floats can be modeled as a ring but are not one, and the unsound rules are exactly
those whose identity fails in the real field itself, before floating point enters. We give a static,
finite-class-cover analyzer that reads a rule's expression structure and predicts the witnessing
input class with no execution (mechanical proof by cases), and a dynamic gate that confirms the
prediction against the real implementation. The static analyzer reproduces, from rule structure
alone, two confirmed soundness bugs we reported to Enzyme-JAX, including a subtle constant-sign
dependence. The method is sound and complete on an explicitly stated, finitely-partitionable
fragment; its blind spots (floating-point precision, gradient ties, structural soundness, and
fractal or float-boundary regimes) are characterized one by one and shown to coincide with the cases
where the soundness question is either sanctioned by the contract or ill-posed, never where a
well-posed bug hides.

## 1. The seam

Compilers like Enzyme-JAX (the Reactant/Enzyme stack) lower traced programs to StableHLO, apply a
few hundred linear-algebra rewrite patterns, and differentiate the result. Coverage of those
patterns is overwhelmingly syntactic: each lit test runs `enzymexlamlir-opt --enzyme-hlo-opt |
FileCheck`, verifying that a rewrite fires and emits the expected IR text. It never runs the program
and never differentiates. A semantic and AD oracle exists in the test harness but is driven by a
handful of end-to-end models that rarely reach edge cases. So every rewrite is checked to
fire-and-look-right; almost none is checked in isolation to preserve value, and none to preserve the
gradient. The author writes the input they already believe in, and is blind to the edge of their own
rule.

## 2. The contract is the oracle

The maintainer's stated contract turns most triage from a matter of taste into a matter of
conformance:

- a value divergence on an ordinary input is a bug, full stop;
- a gradient divergence at a smooth point (where the gradient is unique) is a bug, full stop;
- a divergence only at a non-smooth tie (both subgradients valid), or only under inf/nan/fast-math
  (tacit precision relaxations the contract permits), is sanctioned residue.

We hunt the first two and filter the residue. A finding either violates the stated contract or it
does not; nothing rests on our judgment of whether it "should" be a bug.

## 3. The empty cell

Two lines of prior work bracket the problem. Differential compiler testing (CSmith; Equivalence
Modulo Inputs) checks that an optimization preserves VALUE. Autodiff fuzzing (NablaFuzz) checks that
a PRIMITIVE's gradient is correct. Neither checks that a value-preserving optimization preserves the
GRADIENT, which is the precise contract of differentiate-after-optimize. Arranged as a grid, three
cells are occupied and one is empty:

| | unit = primitive | unit = optimization / composition |
|---|---|---|
| checks value | (type checkers) | DL-compiler fuzzers (NNSmith, Tzer, MT-DLComp) |
| checks gradient | AD fuzzers (NablaFuzz) | **this work** |

The empty cell is gradient soundness of the composition optimize-then-AD. Our flagship finding (a
derivative rule whose primal is correct on negatives but whose gradient is NaN there) lives exactly
in it, invisible to a value-only or a primitive-only oracle.

## 4. Method

### 4.1 Dynamic gate

For each rule we reimplement its unoptimized and optimized forms in JAX from the rule's lit test or
source, enumerate an edge-input lattice (sign classes, zero, inf, nan, denormals, magnitudes, and
ties), and compare value and gradient against the reference: the unoptimized program for value, and
autodiff for derivative rules. The residue is auto-filtered and only self-adjudicating divergences
are surfaced. For value and smooth gradient the JAX computation is the real-number semantics the
rewrite claims to preserve, so a divergence is a real bug independent of how the binary lowers it.
Findings are reconciled as accept-sets and recorded as a replayable hypothesis graph.

### 4.2 Static proto-test generator

A *proto-test* is a pair (witnessing input class, predicted verdict) derived from a rule's structure,
not yet a real test because its oracle is model-predicted rather than ground-truth-pinned. It is
promoted to a regression test by confirming the witness against the real implementation and pinning
the expected value. The cbrt repro we filed is a promoted proto-test: predict that negatives break
the rule, confirm in autodiff, write a one-line mutation of the maintainer's own test.

The generator is a finite-class-cover abstract interpreter. A rewrite L to R is domain-unsound when
R is undefined (non-finite) somewhere L is defined; the witness is any input in `domain(L) \
domain(R)`, and that set is computed by evaluating both expressions on one representative per class
of a fixed universal cover. The selection of which inputs to test is therefore not hand-authored per
rule; the same cover is applied to every rule, and only the rule's structure varies. Authoring cost
is amortized over the operator semantics (about thirty ops, modeled once) rather than paid per rule.

## 5. Floats are not rings

The rewrites are ring (field) identities: `log(a*a) = 2 log(a)`, `(a+b)+c = a+(b+c)`, `(a/b)/c =
a/(b*c)` are theorems in the real field, applied as if floats were that field. Floats can be modeled
as a ring but are not one: no associativity, no exact inverses, a finite range, NaN and inf,
rounding. Every rewrite bug is a place where the ring model and the actual floats diverge, and the
classes split exactly along whether the identity is even true in the ring:

- **A self-adjudicating bug is false even in the reals.** `log(a*a) = 2 log(a)` fails in the partial
  real field: the left side is defined for `a != 0` (since `a*a > 0`), the right only for `a > 0`.
  As partial functions they have different domains, so the identity is false before floats enter. A
  real-valued model with domain tracking decides this. Crucially, fast-math does not excuse it:
  fast-math buys "treat floats as a ring," and the identity fails as a ring identity.
- **Sanctioned residue is true in the reals, false because floats are not a ring**: associativity,
  overflow, rounding, denormals. A real-field model is blind to these, because in the ring they hold.

So fast-math is precisely the license to treat floats as a ring, and the analyzer's blind spot equals
the sanctioned class by construction, not by luck: what fast-math forgives is exactly what a ring
model cannot see. This replaces a heuristic residue filter with a decision procedure: a divergence is
a self-adjudicating bug iff the rewrite is invalid in the partial real field, and sanctioned residue
iff it is valid in the reals but invalid in floats-as-non-ring. (The gradient-at-a-tie residue is a
second, analytic source, handled separately by treating ties as a special class.)

## 6. The result form: proof by cases on a bounded fragment

The method is mechanical proof by cases (deduction over a finite partition) replacing the author's
induction by sampling. The non-trivial content is exhaustiveness: proof by cases is a proof only when
the cases cover the domain, that is, when the finite partition lifts to the infinite input space with
uniform behavior per class. The postcondition of enumerating the combinatorial cover is a total
verdict over the abstraction that lifts to a soundness decision over the whole concrete domain, with
a constructive witness per violation, IFF the cover is a proven exhaustive and uniform partition and
the operator model is faithful; otherwise it degrades to agreement on the sampled representatives.

We bound away the ambiguous zones and point at the boundary explicitly, which is legitimate precisely
because each excluded zone is independently characterized as a place where there is no well-posed bug:
ties (no unique subgradient), precision (fast-math sanctioned), fractal or float-boundary regimes
(ground truth itself aliases), and reachability (discharged by confirmation). The complement is the
region where the question is well-posed, and on it the case analysis is unconditional, modulo
faithfulness and the model-to-system step. The integrity condition is that the boundary is fixed by
the structure of ill-posedness and is not allowed to grow to absorb an inconvenient but well-posed
bug; a boundary that expands for convenience is gerrymandering, not scoping. The result is stronger
for naming its boundary loudly than a broader claim that hides one.

The partition exists in three tiers. Tier 1, finitely partitionable (log, sqrt, cbrt, pow, div, exp,
erf): decidable by enumeration. Tier 2, infinite but regular (tan and gamma poles): no finite
partition but a parametric one; needs symbolic reasoning. Tier 3, fractal or chaotic (`sin(1/x)` near
zero): no partition at any level, and float-aliased at the boundary. The method's blind spots all
land where the question is ill-posed, never where a well-posed bug hides. A tensor compiler's
elementwise core is overwhelmingly tier 1, which is what makes it a clean target.

## 7. Findings

- **EnzymeAD/Enzyme-JAX #2570, LogSimplify.** `log(a*a) -> 2*log(a)`, `log(pow(x,y)) -> y*log(x)`,
  and the constant `log(a*b) -> log(a)+log(b)` / `log(a/b) -> log(a)-log(b)` rules narrow the domain
  and return NaN on finite negative inputs. The static analyzer recovers all four from structure,
  including the constant-sign dependence: a negative constant breaks the identity, a positive one is
  sound.
- **EnzymeAD/Enzyme-JAX #2571, CbrtOp derivative.** The rule routes the gradient through
  `pow(x, -2/3)`, which is NaN for negative `x`, while `cbrt` is real and smooth there with a finite
  derivative. The primal is correct; the gradient is poisoned. The repro is a one-line mutation of
  the maintainer's own `cbrt.mlir`: flip the test inputs from positive to negative and the expected
  gradients are unchanged, but the rule yields NaN.

Both findings are self-adjudicating value/gradient divergences on ordinary finite inputs, and both
fall out of `domain(L) \ domain(R)` over the sign cover.

## 8. Empirical sweep (in progress)

A parser over the declarative derivative-rule definitions lifts each rule to the analyzer's
expression language and predicts a verdict against an autodiff reference, with no execution. Built as
a sound over-approximation (tracking the set of possible sign/definedness outcomes and flagging any
combination that diverges), the sweep yields a completeness statement over the rule set: of N rules,
K are flagged domain-narrowing and gate-confirmed, and the remainder are provably valid in the real
field within the stated fragment. The C++ imperative rewrites require a heavier front end and are
left to future work; the declarative derivative table is the clean target.

## 9. Related work

Differential and random compiler testing: CSmith; Equivalence Modulo Inputs and its variants;
metamorphic testing of deep-learning compilers; DL-compiler fuzzers (NNSmith, Tzer). Autodiff
testing: NablaFuzz; finite-difference gradient checking. Rewrite verification: Alive2 and Alive-FP
(the SMT verification pole, of which this is the testing complement for transforms outside the
decidable fragment). Test generation from existing tests: test amplification (DSpot). Spec inference
and oracle-guided methods: Daikon (dynamic invariant detection), oracle-guided component-based
synthesis (the distinguishing-input loop). Two recent results name the problem this work addresses:
a survey of compiler testing lists "test oracles beyond equivalence relations" as an open challenge,
and recent N-version-programming work with coding agents observes that correlated failures trace to
specification ambiguity and poses automated detection and refinement of those ambiguities as future
work. The contribution here is not a new proof technique (the case analysis is elementary) but the
reduction of an open soundness question to a decidable finite case analysis on a characterized
fragment, with confirmed bugs.

## 10. Conclusion

The soundness of an autodiff compiler's elementwise rewrites is, on a precisely stated fragment, a
finite case analysis rather than a search: the rules are ring identities, the bugs are the identities
that fail in the ring itself, and a fixed class cover decides them from structure alone. The method's
limits are stated as a boundary one can audit, and that boundary lands exactly where the soundness
question is sanctioned or ill-posed. A static pass predicts cheaply and exhaustively; a dynamic gate
confirms against the real implementation; and the surviving witnesses are one-line mutations of the
maintainer's own tests.
