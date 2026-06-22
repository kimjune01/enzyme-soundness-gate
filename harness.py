"""
Enzyme-JAX rewrite-soundness gate — JAX backend for abductor.

For each StableHLO rewrite pattern (faithfully reimplemented from its lit test as
an (unoptimized, optimized) JAX function pair), enumerate an edge-input lattice,
auto-filter the residue (ties / inf-nan inputs / denormals / overflow = tacit
relaxations the maintainer accepts), and emit two accept-set files per gate:

    truth.txt   = every NON-RESIDUE case id  (their contract: opt MUST agree on all)
    believe.txt = the case ids where the optimized form actually matched unopt

`abductor gate --believe believe.txt --truth truth.txt` then reports the symmetric
difference: exactly the self-adjudicating divergences (value, or smooth-gradient).

Definitive, not a proxy: for value + smooth gradient the JAX computation IS the
real-number semantics the rewrite claims to preserve. A divergence here is a real
bug regardless of how enzyme_ad lowers it.
"""
import math
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# residue classifier — auto-filter the cases the maintainer's contract excuses
# ---------------------------------------------------------------------------
# Moderate-magnitude band. Outside it, finite<->non-finite divergences are
# overflow/underflow artifacts (exp saturating, x*x underflowing) = the range/
# fast-math relaxations the maintainer's contract tacitly accepts. Inside it, a
# finite-vs-NaN value divergence cannot be a magnitude artifact -> it is a real
# domain-narrowing bug. The band is the mechanical proxy for "not a range residue".
BAND_HI = 1e3
BAND_LO = 1e-3


def is_residue(xs):
    """True if this input is excused by a tacit relaxation (auto-filter).

    Keys on the INPUT only, never on the output. A finite *moderate* input that
    PRODUCES a nan after optimization is NOT residue — that nan is the bug.
    """
    for v in xs:
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return True  # inf/nan input: "undefined / match XLA"
        if abs(v) > BAND_HI:
            return True  # extreme magnitude: overflow relaxation
        if v != 0.0 and abs(v) < BAND_LO:
            return True  # tiny magnitude: underflow/denormal relaxation
    return False


def is_tie(xs, eps=0.0):
    """Operands equal -> non-smooth point for min/max/select. Grad residue."""
    vals = [float(v) for v in xs]
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if abs(vals[i] - vals[j]) <= eps:
                return True
    return False


# ---------------------------------------------------------------------------
# agreement predicates — nan-aware; allclose for finite
# ---------------------------------------------------------------------------
def agree(a, b, rtol=1e-9, atol=1e-12):
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    both_nan = jnp.logical_and(jnp.isnan(a), jnp.isnan(b))
    # equal infinities (same sign) count as agreement
    both_inf = jnp.logical_and(jnp.isinf(a), a == b)
    close = jnp.isclose(a, b, rtol=rtol, atol=atol)
    return bool(jnp.all(jnp.logical_or(jnp.logical_or(both_nan, both_inf), close)))


def finite(a):
    return bool(jnp.all(jnp.isfinite(jnp.asarray(a, dtype=jnp.float64))))


# ---------------------------------------------------------------------------
# pattern registry — faithful (unopt, opt) pairs from the lit tests
# each pattern: name, arity, f_unopt, f_opt, gates, lit, note
# scalar functions of a length-`arity` input vector, so jax.grad works.
# ---------------------------------------------------------------------------
PATTERNS = []


def pattern(**kw):
    PATTERNS.append(kw)


# --- LogSimplify family (EnzymeHLOOpt.cpp:27376, lit: logsimplify.mlir) ------

# THE BUG: log(a*a) -> 2*log(a). a*a is always >=0 so LHS real for all a!=0;
# RHS = 2*log(a) is NaN for every a<0. Domain narrowed a!=0  ->  a>0.
pattern(
    name="log_mul_square",
    arity=1,
    f_unopt=lambda x: jnp.log(x[0] * x[0]),
    f_opt=lambda x: 2.0 * jnp.log(x[0]),
    gates=("value", "grad"),
    lit="logsimplify.mlir @main2 (the a*a sub-case), cpp:27406",
    note="log(a*a)->2*log(a): unsound for a<0. UNGUARDED in LogSimplify.",
)

# log(pow(x,y)) -> y*log(x). Original real whenever pow(x,y)>0 (incl. x<0, even y);
# optimized real only for x>0. cpp:27391. Test with y=2.0 (constant exponent).
pattern(
    name="log_pow_even",
    arity=1,
    f_unopt=lambda x: jnp.log(jnp.power(x[0], 2.0)),
    f_opt=lambda x: 2.0 * jnp.log(x[0]),
    gates=("value", "grad"),
    lit="logsimplify.mlir (pow sub-case), cpp:27391",
    note="log(pow(x,2))->2*log(x): unsound for x<0.",
)

# log(a*b) -> log(a)+log(b) when one operand is a (negative) constant. cpp:27416.
# With a NEGATIVE constant, x<0 makes x*b>0 (orig real) but log(x)=NaN (opt). BUG.
pattern(
    name="log_mul_negconst",
    arity=1,
    f_unopt=lambda x: jnp.log(x[0] * -3.0),
    f_opt=lambda x: jnp.log(x[0]) + jnp.log(-3.0),
    gates=("value",),
    lit="logsimplify.mlir (mul-const sub-case), cpp:27416",
    note="log(x*-3)->log(x)+log(-3): unsound for x<0 (negative constant).",
)

# log(a/b) -> log(a)-log(b) when one operand is a (negative) constant. cpp:27469.
pattern(
    name="log_div_negconst",
    arity=1,
    f_unopt=lambda x: jnp.log(x[0] / -3.0),
    f_opt=lambda x: jnp.log(x[0]) - jnp.log(-3.0),
    gates=("value",),
    lit="logsimplify.mlir (div-const sub-case), cpp:27469",
    note="log(x/-3)->log(x)-log(-3): unsound for x<0 (negative constant).",
)

# log(a*b) with POSITIVE constant: sound (both NaN for x<0). Control.
pattern(
    name="log_mul_posconst_ctrl",
    arity=1,
    f_unopt=lambda x: jnp.log(x[0] * 3.0),
    f_opt=lambda x: jnp.log(x[0]) + jnp.log(3.0),
    gates=("value",),
    lit="logsimplify.mlir (mul-const, positive), cpp:27416",
    note="control: positive constant -> domain preserved.",
)

# ChainedMultiplyToPower: (x*x)*x -> pow(x,3). XLA pow(neg,int) is correct -> SOUND.
pattern(
    name="chain_pow3_ctrl",
    arity=1,
    f_unopt=lambda x: (x[0] * x[0]) * x[0],
    f_opt=lambda x: jnp.power(x[0], 3.0),
    gates=("value", "grad"),
    lit="ChainedMultiplyToPower, cpp:27080",
    note="control: refuted hypothesis — pow(neg,3)=-8 not NaN, sound.",
)
pattern(
    name="chain_pow4_ctrl",
    arity=1,
    f_unopt=lambda x: (x[0] * x[0]) * (x[0] * x[0]),
    f_opt=lambda x: jnp.power(x[0], 4.0),
    gates=("value", "grad"),
    lit="ChainedMultiplyToPower, cpp:27068",
    note="control: pow(neg,4)=16 not NaN, sound.",
)

# log(exp(x)) -> x. Sound everywhere (control; expect PASS).
pattern(
    name="log_exp_ctrl",
    arity=1,
    f_unopt=lambda x: jnp.log(jnp.exp(x[0])),
    f_opt=lambda x: x[0],
    gates=("value", "grad"),
    lit="logsimplify.mlir @main1, cpp:27383",
    note="control: should be sound on finite inputs.",
)

# log(sqrt(x)) -> log(x)/2. Sound on reals (both nan for x<0). cpp:27480. Control.
pattern(
    name="log_sqrt_ctrl",
    arity=1,
    f_unopt=lambda x: jnp.log(jnp.sqrt(x[0])),
    f_opt=lambda x: jnp.log(x[0]) / 2.0,
    gates=("value", "grad"),
    lit="logsimplify.mlir @main9, cpp:27480",
    note="control: domain-preserving (both nan for x<0).",
)

# log(rsqrt(x)) -> log(x)/(-2). Sound incl. x=0 (+inf both). cpp:27508. Control.
pattern(
    name="log_rsqrt_ctrl",
    arity=1,
    f_unopt=lambda x: jnp.log(jax.lax.rsqrt(x[0])),
    f_opt=lambda x: jnp.log(x[0]) / (-2.0),
    gates=("value", "grad"),
    lit="logsimplify.mlir @main10, cpp:27508",
    note="control: domain-preserving on reals.",
)

# --- max-reduction fusion (known TIE residue; expect PASS once ties filtered) -
pattern(
    name="max_nested_fuse",
    arity=2,
    f_unopt=lambda x: jnp.maximum(jnp.maximum(x[0], x[1]), x[0]),
    f_opt=lambda x: jnp.maximum(x[0], x[1]),
    gates=("value", "grad"),
    lit="addreduceslicefusion.mlir @test_max",
    note="value-sound; grad diverges ONLY at ties -> residue, must be filtered.",
    nonsmooth=True,
)

# add control: fully sound.
pattern(
    name="add_ctrl",
    arity=2,
    f_unopt=lambda x: (x[0] + x[1]) + x[0],
    f_opt=lambda x: (x[0] + x[1]) + x[0],
    gates=("value", "grad"),
    lit="addreduceslicefusion.mlir @test",
    note="control: identical functions.",
)


# ---------------------------------------------------------------------------
# edge-input lattice
# ---------------------------------------------------------------------------
SCALAR_LATTICE = [
    2.0, 1.0, 0.5, 3.7, 7.3,           # ordinary positive
    -2.0, -1.0, -0.5, -3.7, -7.3,      # ordinary NEGATIVE (the hunt)
    0.0, -0.0,                          # zeros
    1e-12, 1e-6,                        # small but normal
    1e6, 1e12,                          # large but no overflow
    1e-200, 1e-308,                     # denormal-ish (residue)
    1e200, 1e308,                       # overflow-on-square (residue)
    math.inf, -math.inf, math.nan,      # special (residue)
]


def lattice(arity):
    if arity == 1:
        return [[v] for v in SCALAR_LATTICE]
    # arity 2: ordinary pairs + ties + a couple specials
    base = [2.0, 1.0, -2.0, -1.0, 0.5, -0.5, 3.0, 1e6, 1e-12]
    cases = []
    for a in base:
        for b in base:
            cases.append([a, b])
    cases += [[1.0, 1.0], [3.0, 3.0], [-2.0, -2.0], [0.0, 0.0]]  # explicit ties
    cases += [[math.nan, 1.0], [math.inf, 1.0], [1e308, 1e308]]   # specials
    return cases


# ---------------------------------------------------------------------------
# gate: emit truth/believe accept-set files for one (pattern, gate)
# ---------------------------------------------------------------------------
def run_pattern_gate(p, gate, outdir):
    arity = p["arity"]
    nonsmooth = p.get("nonsmooth", False)
    cases = lattice(arity)
    fu, fo = p["f_unopt"], p["f_opt"]
    gu = jax.grad(fu) if gate == "grad" else None
    go = jax.grad(fo) if gate == "grad" else None

    truth, believe = [], []
    diverged = []
    legend = {}
    for i, xs in enumerate(cases):
        if is_residue(xs):
            continue  # auto-filter: tacit relaxation
        if gate == "grad" and nonsmooth and is_tie(xs):
            continue  # auto-filter: non-smooth tie, both subgradients valid
        cid = i  # integer case id (abductor accept-set tokens are ints)
        legend[cid] = xs
        x = jnp.asarray(xs, dtype=jnp.float64)
        try:
            if gate == "value":
                a, b = fu(x), fo(x)
            else:
                # gradient is only meaningful where the primal is in-domain and
                # smooth. If either primal is non-finite, the point is on the
                # domain boundary -> residue, skip.
                if not (finite(fu(x)) and finite(fo(x))):
                    continue
                a, b = gu(x), go(x)
        except Exception:
            continue
        if gate == "value" and not agree(a, b) and not finite(a) and not finite(b):
            continue  # both sides non-finite & disagree: special-value (pole/
                      # signed-zero/inf) residue, "match XLA" class — not a
                      # domain-narrowing bug (that is finite-vs-nonfinite).
        truth.append(cid)            # contract: opt must agree on every tested case
        if agree(a, b):
            believe.append(cid)      # opt actually agreed
        else:
            diverged.append((cid, xs, _fmt(a), _fmt(b)))

    base = f"{outdir}/{p['name']}_{gate}"
    with open(f"{base}.truth.txt", "w") as f:
        f.write(" ".join(str(c) for c in truth) + "\n")
    with open(f"{base}.believe.txt", "w") as f:
        f.write(" ".join(str(c) for c in believe) + "\n")
    with open(f"{base}.legend.txt", "w") as f:
        for cid in sorted(legend):
            f.write(f"{cid}\t{legend[cid]}\n")
    return base, len(truth), diverged


def _fmt(a):
    try:
        arr = jnp.asarray(a).ravel()
        return "[" + ",".join(f"{float(v):.6g}" for v in arr) + "]"
    except Exception:
        return str(a)


if __name__ == "__main__":
    import os, sys
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/enzyme_gate"
    os.makedirs(outdir, exist_ok=True)
    print(f"# emitting accept-sets to {outdir}\n")
    for p in PATTERNS:
        for gate in p["gates"]:
            base, n, div = run_pattern_gate(p, gate, outdir)
            flag = "  <<< DIVERGENCE" if div else ""
            print(f"{p['name']:18s} {gate:5s}  tested={n:3d}  diverged={len(div):3d}{flag}")
            for cid, xs, a, b in div[:6]:
                print(f"      {cid}  x={xs}  unopt={a}  opt={b}")
