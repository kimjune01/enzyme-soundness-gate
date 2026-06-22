"""
AD-rule gradient-soundness gate (orthogonal target: Enzyme-JAX derivative rules).

The lit tests for derivative rules (test/lit_tests/diffrules/...) are SYNTACTIC:
`enzymexlamlir-opt --enzyme-wrap | FileCheck` checks the generated derivative IR
text, never evaluates the gradient. Same blindness as the rewrite library.

This gate compares, per op:
    reference = jax.grad(primal)        (the true gradient via autodiff)
    rule      = the .td HLODerivative expression, reimplemented in JAX
on an edge lattice, in-domain (primal finite), auto-filtering residue. A divergence
at a smooth point is a self-adjudicating gradient bug.

Source: src/enzyme_ad/jax/Implementations/HLODerivatives.td
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from harness import is_residue, agree, finite, SCALAR_LATTICE

# AD-rule registry: name, primal fn, rule-grad fn (from the .td), lit, note
AD_RULES = []
def adrule(**kw): AD_RULES.append(kw)

# --- CbrtOp (HLODerivatives.td:1081): d/dx = (1/3) * pow(x, -2/3) ----------
# cbrt is real+smooth for x<0, but pow(neg, -2/3) is NaN -> gradient NaN. BUG.
adrule(
    name="cbrt",
    primal=lambda x: jnp.cbrt(x),
    rule_grad=lambda x: jnp.power(x, -2.0/3.0) / 3.0,
    lit="diffrules cbrt; HLODerivatives.td:1081",
    note="d/dx cbrt via pow(x,-2/3): NaN for x<0 where true grad is finite real.",
)

# --- controls: rules I verified correct by hand --------------------------------
# ErfInvOp (CHLODerivatives.td:157): sqrt(pi)/2 * exp(erfinv(x)^2). Domain (-1,1).
adrule(
    name="erfinv_ctrl",
    primal=lambda x: jax.scipy.special.erfinv(x),
    rule_grad=lambda x: 0.8862269254527580 * jnp.exp(jax.scipy.special.erfinv(x)**2),
    lit="diffrules/chlo/erfinv.mlir; CHLODerivatives.td:157",
    note="control: rule verified correct.",
)
# ErfcOp (CHLODerivatives.td:150): -2/sqrt(pi) * exp(-x^2).
adrule(
    name="erfc_ctrl",
    primal=lambda x: jax.scipy.special.erfc(x),
    rule_grad=lambda x: -1.1283791670955126 * jnp.exp(-(x*x)),
    lit="diffrules/chlo/erfc.mlir; CHLODerivatives.td:150",
    note="control: rule verified correct.",
)
# SqrtOp (HLODerivatives.td:1316): Select(x==0, 0, 1/(2 sqrt(x))). Both nan for x<0.
adrule(
    name="sqrt_ctrl",
    primal=lambda x: jnp.sqrt(x),
    rule_grad=lambda x: jnp.where(x == 0.0, 0.0, 1.0/(2.0*jnp.sqrt(x))),
    lit="HLODerivatives.td:1316",
    note="control: domain-preserving (both nan for x<0).",
)
# TanhOp (HLODerivatives.td:1322): 1 - tanh^2.
adrule(
    name="tanh_ctrl",
    primal=lambda x: jnp.tanh(x),
    rule_grad=lambda x: 1.0 - jnp.tanh(x)**2,
    lit="HLODerivatives.td:1322",
    note="control: smooth everywhere.",
)


def run_adrule(p, outdir):
    cases = [[v] for v in SCALAR_LATTICE]
    gref = jax.grad(lambda t: p["primal"](t))
    truth, believe, diverged, legend = [], [], [], {}
    for i, xs in enumerate(cases):
        if is_residue(xs):
            continue
        x = jnp.asarray(xs[0], dtype=jnp.float64)
        # gradient only meaningful where the primal is in-domain (finite)
        if not finite(p["primal"](x)):
            continue
        try:
            ref = gref(x)
            rule = p["rule_grad"](x)
        except Exception:
            continue
        # smooth-point condition: only gate where the TRUE gradient is finite and
        # unique. A non-finite reference gradient is a derivative singularity
        # (e.g. d/dx sqrt at 0 = inf); the implementer may pick any convention
        # there -> residue, same class as a min/max tie.
        if not finite(ref):
            continue
        cid = i
        legend[cid] = xs
        truth.append(cid)
        if agree(ref, rule):
            believe.append(cid)
        else:
            diverged.append((cid, xs[0], float(ref), float(rule)))
    base = f"{outdir}/adrule_{p['name']}"
    with open(f"{base}.truth.txt", "w") as f: f.write(" ".join(map(str, truth)) + "\n")
    with open(f"{base}.believe.txt", "w") as f: f.write(" ".join(map(str, believe)) + "\n")
    return len(truth), diverged


if __name__ == "__main__":
    import os, sys
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/enzyme_gate"
    os.makedirs(outdir, exist_ok=True)
    print(f"# AD-rule gradient gate -> {outdir}\n")
    for p in AD_RULES:
        n, div = run_adrule(p, outdir)
        flag = "  <<< GRAD DIVERGENCE" if div else ""
        print(f"adrule_{p['name']:14s}  tested={n:3d}  diverged={len(div):3d}{flag}")
        for cid, x, ref, rule in div[:6]:
            print(f"      c{cid}  x={x}  jax.grad={ref:.6g}  rule={rule}")
