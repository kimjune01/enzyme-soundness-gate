"""Static finite-class-cover analysis for rewrite / derivative rules.

Research claim (the paper's automation crux + its scope boundary): the edge-input
selection is NOT hand-authored per rule. For any rule over a domain that admits a
FINITE CLASS COVER, evaluate both sides of the rule on one representative per class
and flag the classes where they diverge. A divergence on an ORDINARY class is a
self-adjudicating bug; a divergence only on a SPECIAL class (NaN / inf / a tie /
a domain boundary) is residue. The same machinery covers:

  * numbers: real scalars under sign classes {neg, pos} + special {zero, NaN, +-inf}
  * AD rules: the rule's derivative expression vs the TRUE derivative expression
  * enums: the compare / select / min / max family over the order relation
    {x<y, x==y, x>y} crossed with special-value classes (one operand NaN)

In scope = any finite class cover (numbers + enums). OUT of scope = relational /
structural soundness (tensor shape, index, broadcast, aliasing, or a witness that
is an arithmetic relation between shape variables), which no finite class cover
captures. That boundary is the paper's stated limitation.
"""
import math

NAN, INF = math.nan, math.inf


# expression mini-language:  ('var', name) | ('const', c) | (op, *args)
def ev(e, env):
    t = e[0]
    if t == "var":   return env[e[1]]
    if t == "const": return e[1]
    if t == "neg":   return -ev(e[1], env)
    if t == "mul":   return ev(e[1], env) * ev(e[2], env)
    if t == "add":   return ev(e[1], env) + ev(e[2], env)
    if t == "sub":   return ev(e[1], env) - ev(e[2], env)
    if t == "div":
        u, v = ev(e[1], env), ev(e[2], env)
        if v == 0: return NAN if u == 0 else math.copysign(INF, u) * math.copysign(1, v)
        return u / v
    if t == "log":
        u = ev(e[1], env); return math.log(u) if u > 0 else NAN
    if t == "sqrt":
        u = ev(e[1], env); return math.sqrt(u) if u >= 0 else NAN
    if t == "cbrt":
        u = ev(e[1], env)
        return math.copysign(abs(u) ** (1 / 3), u) if math.isfinite(u) else u
    if t == "exp":
        u = ev(e[1], env)
        try: return math.exp(u)
        except OverflowError: return INF
    if t == "sin":  return math.sin(ev(e[1], env))
    if t == "cos":  return math.cos(ev(e[1], env))
    if t == "tanh": return math.tanh(ev(e[1], env))
    if t == "logistic":
        u = ev(e[1], env)
        try: return 1.0 / (1.0 + math.exp(-u))
        except OverflowError: return 0.0
    if t == "eq":
        u, v = ev(e[1], env), ev(e[2], env)
        return 1.0 if u == v else 0.0
    if t == "pow":
        u, c = ev(e[1], env), e[2]
        if not math.isfinite(u):
            return u if u == INF else NAN
        if float(c).is_integer():
            ci = int(c)
            if u == 0 and ci < 0: return INF
            return float(u) ** ci
        if u > 0: return u ** c
        if u == 0 and c > 0: return 0.0
        return NAN
    if t == "max":
        u, v = ev(e[1], env), ev(e[2], env)
        return NAN if (math.isnan(u) or math.isnan(v)) else max(u, v)
    if t == "min":
        u, v = ev(e[1], env), ev(e[2], env)
        return NAN if (math.isnan(u) or math.isnan(v)) else min(u, v)
    if t in ("ge", "gt", "le", "lt"):
        u, v = ev(e[1], env), ev(e[2], env)
        ok = {"ge": u >= v, "gt": u > v, "le": u <= v, "lt": u < v}[t]  # NaN -> False
        return 1.0 if ok else 0.0
    if t == "select":   # select(cond, t, f): cond true (nonzero, non-NaN) picks t
        c = ev(e[1], env)
        return ev(e[2], env) if (c != 0 and not math.isnan(c)) else ev(e[3], env)
    raise ValueError(e)


def agree(a, b, rtol=1e-9):
    if math.isnan(a) and math.isnan(b): return True
    if math.isinf(a) or math.isinf(b):  return a == b
    return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12)


def analyze(lhs, rhs, cover):
    """Return (all_divergences, ordinary_divergences). A bug iff an ordinary class diverges."""
    div = []
    for label, kind, env in cover:
        a, b = ev(lhs, env), ev(rhs, env)
        if not agree(a, b):
            div.append((label, kind, a, b))
    return div, [d for d in div if d[1] == "ordinary"]


# --- covers (universal, not per-rule) -------------------------------------------
def cover_1v(v="a"):
    return [
        ("neg", "ordinary", {v: -2.0}), ("pos", "ordinary", {v: 2.0}),
        ("zero", "special", {v: 0.0}),
        ("NaN", "special", {v: NAN}), ("+inf", "special", {v: INF}), ("-inf", "special", {v: -INF}),
    ]

COVER_2V = [
    ("x>y", "ordinary", {"x": 2.0, "y": 1.0}),
    ("x<y", "ordinary", {"x": 1.0, "y": 2.0}),
    ("x==y", "special", {"x": 1.0, "y": 1.0}),    # tie: non-smooth, residue
    ("x=NaN", "special", {"x": NAN, "y": 1.0}),
    ("y=NaN", "special", {"x": 1.0, "y": NAN}),
]

A = ("var", "a"); X = ("var", "x"); Y = ("var", "y")
SQ = ("mul", A, A)

VALUE_RULES = {  # (name, L, R, cover)
    "log(a*a) -> 2*log(a)":            (("log", SQ), ("mul", ("const", 2), ("log", A)), cover_1v()),
    "log(pow(a,2)) -> 2*log(a)":       (("log", ("pow", A, 2.0)), ("mul", ("const", 2), ("log", A)), cover_1v()),
    "log(a*-3) -> log(a)+log(-3)  [neg const]": (("log", ("mul", A, ("const", -3.0))),
                                                 ("add", ("log", A), ("log", ("const", -3.0))), cover_1v()),
    "log(a/-3) -> log(a)-log(-3)  [neg const]": (("log", ("div", A, ("const", -3.0))),
                                                 ("sub", ("log", A), ("log", ("const", -3.0))), cover_1v()),
    "log(a*3) -> log(a)+log(3)  [pos const ctrl]": (("log", ("mul", A, ("const", 3.0))),
                                                    ("add", ("log", A), ("log", ("const", 3.0))), cover_1v()),
    "log(a+a) -> log(2)+log(a)  [ctrl]": (("log", ("add", A, A)),
                                          ("add", ("log", ("const", 2)), ("log", A)), cover_1v()),
    "log(exp(a)) -> a  [ctrl]":        (("log", ("exp", A)), A, cover_1v()),
    "log(sqrt(a)) -> log(a)/2  [ctrl]":(("log", ("sqrt", A)), ("mul", ("const", 0.5), ("log", A)), cover_1v()),
}

# AD rules: rule-derivative expression vs the TRUE derivative expression.
AD_RULES = {
    # cbrt' true = 1/(3*cbrt(x)^2) = (1/3)*pow(cbrt(x),-2);  rule = (1/3)*pow(x,-2/3)
    "cbrt' rule pow(x,-2/3)": (
        ("mul", ("const", 1/3), ("pow", ("cbrt", A), -2.0)),        # true
        ("mul", ("const", 1/3), ("pow", A, -2/3)),                  # Enzyme rule
        cover_1v()),
    # sqrt' true = 1/(2*sqrt(x)); rule same  [ctrl]
    "sqrt' rule 1/(2 sqrt x)  [ctrl]": (
        ("mul", ("const", 0.5), ("pow", ("sqrt", A), -1.0)),
        ("mul", ("const", 0.5), ("pow", ("sqrt", A), -1.0)),
        cover_1v()),
}

ENUM_RULES = {  # compare/select family (2-var, order + special classes)
    "select(x>=y,x,y) -> max(x,y)  [commoncompare]": (
        ("select", ("ge", X, Y), X, Y), ("max", X, Y), COVER_2V),
    "select(x>=y,y,x) -> max(x,y)  [synthetic bug: branches swapped]": (
        ("select", ("ge", X, Y), Y, X), ("max", X, Y), COVER_2V),
}


def report(title, rules):
    print(f"=== {title} ===")
    for name, (L, R, cover) in rules.items():
        div, bug = analyze(L, R, cover)
        tag = "BUG  " if bug else "sound"
        wit = [d[0] for d in bug] or "-"
        residue = [d[0] for d in div if d[1] != "ordinary"] or "-"
        print(f"  {tag} {name:50s} bug_class={wit}  residue_class={residue}")


if __name__ == "__main__":
    report("value rewrites (numbers)", VALUE_RULES)
    print()
    report("AD derivative rules (expr vs true-deriv expr)", AD_RULES)
    print()
    report("compare/select family (enums: order x special-value classes)", ENUM_RULES)
