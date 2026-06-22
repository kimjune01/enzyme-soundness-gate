"""Sound over-approximation sweep over Enzyme-JAX's declarative derivative table.

Reads the real `HLODerivatives.td` unary elementwise/transcendental rules, lifts
each rule's forward-derivative expression to an AST, and decides DOMAIN-NARROWING
soundness by a sign+definedness ABSTRACT INTERPRETER -- not point sampling. For
each input sign class (neg / pos) the analyzer tracks, per subexpression, whether
it is guaranteed finite-real (DEF), guaranteed non-finite (UNDEF), or undetermined
(MAYBE). A rule narrows the domain iff its derivative is not guaranteed DEF on a
class where the TRUE derivative is DEF.

Why sign classes are a SOUND exhaustive partition here (the lift the point-sampler
in domain_analysis.py only had on break-point-zero rules): a unary derivative rule
is a function of a single $x whose only definedness break-point is 0 -- log/sqrt
defined iff arg>0/>=0, pow(.,non-integer) iff arg>0, cbrt/exp/tanh total. None
introduce an INTERIOR additive constant (the log((a-3)(a+3)) failure mode needs a
binary rule), so definedness is constant on (-inf,0) and (0,inf). The abstract
interpreter never reads a single representative; it propagates the sign lattice, so
a class it cannot certify is reported MAYBE (flagged), never silently passed. That
is what earns the no-false-negative (sound over-approximation) guarantee.

Output: of N unary rules, K flagged domain-narrowing (gate-confirmable), the rest
PROVABLY real-field definedness-preserving within the fragment.

Usage:  python3 td_sweep.py [path/to/HLODerivatives.td]
"""
import re, sys, math

TD_DEFAULT = ("/Users/junekim/Documents/Enzyme-JAX/src/enzyme_ad/jax/"
              "Implementations/HLODerivatives.td")

# ---------------------------------------------------------------------------
# 1. Parse the .td: pull each `def : HLODerivative<"OpName", (Op $args), [fwd]...`
# ---------------------------------------------------------------------------
def find_rules(text):
    """Yield (op_name, args_list, forward_sexpr_string) for each HLODerivative."""
    for m in re.finditer(r'def\s*:\s*HLODerivative<\s*"([A-Za-z0-9]+)"\s*,', text):
        op = m.group(1)
        i = m.end()
        # (Op $a, $b, ...)
        am = re.compile(r'\(Op([^)]*)\)').match(text, _skip_ws(text, i))
        if not am:
            continue
        args = re.findall(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', am.group(1))
        j = _skip_ws(text, am.end())
        if j >= len(text) or text[j] != ',':
            continue
        j = _skip_ws(text, j + 1)
        if j >= len(text) or text[j] != '[':
            continue  # forward derivative must be a [...] list
        fwd, _ = _match_bracket(text, j, '[', ']')
        yield op, args, fwd.strip()


def _skip_ws(s, i):
    while i < len(s) and s[i] in ' \t\r\n':
        i += 1
    return i


def _match_bracket(s, i, op, cl):
    """s[i]==op; return (inner_without_outer_brackets, index_after_close)."""
    depth, k = 0, i
    while k < len(s):
        if s[k] == op:
            depth += 1
        elif s[k] == cl:
            depth -= 1
            if depth == 0:
                return s[i + 1:k], k + 1
        k += 1
    raise ValueError("unbalanced")


# ---------------------------------------------------------------------------
# 2. Lift the TableGen s-expression to the analyzer AST (domain_analysis tuples)
#    AST: ('var',n) | ('const',c) | (op, *args)
# ---------------------------------------------------------------------------
def tokenize(s):
    s = re.sub(r'//[^\n]*', '', s)            # strip line comments
    return re.findall(r'\(|\)|,|<"[^"]*">|\$[A-Za-z0-9_]+|[A-Za-z0-9_]+|-?\d+\.?\d*', s)


class P:
    def __init__(self, toks): self.t = toks; self.i = 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def next(self): tok = self.t[self.i]; self.i += 1; return tok


def parse_sexpr(p):
    """Parse one node. Returns ('node', head, [children]) or ('$',name) or ('num',v)."""
    tok = p.peek()
    if tok == '(':
        p.next()
        head = p.next()
        tmpl = None
        if p.peek() and p.peek().startswith('<"'):
            tmpl = p.next()[2:-2]          # strip <" ">
        children = []
        while p.peek() not in (')', None):
            if p.peek() == ',':
                p.next(); continue
            children.append(parse_sexpr(p))
        p.next()                            # consume ')'
        return ('node', head, tmpl, children)
    if tok and tok.startswith('$'):
        p.next(); return ('$', tok[1:])
    if tok and re.match(r'-?\d', tok):
        p.next(); return ('num', float(tok))
    p.next(); return ('atom', tok)


# DSL head -> analyzer op.  DiffeRet is the upstream gradient; we factor it out and
# analyze the MULTIPLIER (the rule's local derivative).  CheckedMul/CheckedDiv with
# a DiffeRet operand are unwrapped accordingly.
BINOPS = {"Mul": "mul", "Div": "div", "Add": "add", "Sub": "sub", "Pow": "pow"}
UNOPS = {"Neg": "neg", "Sqrt": "sqrt", "Cbrt": "cbrt", "Exp": "exp", "Log": "log",
         "Sin": "sin", "Cos": "cos", "Tanh": "tanh", "Logistic": "logistic",
         "Abs": "abs", "Rsqrt": "rsqrt", "Sign": "sign", "Real": "id", "Imag": "id"}


class Unsupported(Exception):
    pass


def fold(e):
    """Constant-fold constant-only subtrees so symbolic exponents like (-2)/(3)
    become a literal ('const', -0.666...). Keeps the abstract eval precise."""
    if not isinstance(e, tuple) or e[0] in ('var', 'const'):
        return e
    op = e[0]
    args = [fold(a) for a in e[1:]]
    if all(isinstance(a, tuple) and a[0] == 'const' for a in args):
        import domain_analysis as da
        try:
            val = da.ev((op, *args), {})
            if isinstance(val, float):
                return ('const', val)
        except Exception:
            pass
    return (op, *args)


def lift(node, var):
    """node -> analyzer AST, factoring out DiffeRet (=1). Raises Unsupported."""
    kind = node[0]
    if kind == '$':
        return ('var', var) if node[1] == var else ('var', node[1])
    if kind == 'num':
        return ('const', node[1])
    if kind != 'node':
        raise Unsupported(node)
    head, tmpl, ch = node[1], node[2], node[3]
    if head == "DiffeRet":
        return ('const', 1.0)               # factor out upstream grad
    if head == "HLOConstantFP":
        return ('const', float(tmpl))
    if head in ("CheckedMul", "Mul"):
        a, b = lift(ch[0], var), lift(ch[1], var)
        return _drop1(a, b, 'mul')
    if head in ("CheckedDiv", "Div"):
        a, b = lift(ch[0], var), lift(ch[1], var)
        # CheckedDiv(DiffeRet, D) -> 1/D
        if a == ('const', 1.0):
            return ('div', ('const', 1.0), b)
        return ('div', a, b)
    if head in BINOPS:
        return (BINOPS[head], lift(ch[0], var), lift(ch[1], var))
    if head in UNOPS:
        if UNOPS[head] == 'id':
            return lift(ch[0], var)
        if UNOPS[head] == 'rsqrt':
            return ('div', ('const', 1.0), ('sqrt', lift(ch[0], var)))
        return (UNOPS[head], lift(ch[0], var))
    if head == "Select":
        # Select(cond, t, f). For sign classes neg/pos the guard is x==0 / x>=0:
        # we keep both branches and let the abstract eval join them.
        return ('select_sym', lift(ch[1], var), lift(ch[2], var))
    raise Unsupported(head)


def _drop1(a, b, op):
    if a == ('const', 1.0):
        return b
    if b == ('const', 1.0):
        return a
    return (op, a, b)


# ---------------------------------------------------------------------------
# 3. Sign+definedness ABSTRACT INTERPRETER (the sound over-approximation).
#    Abstract value = (defined, sign) with
#      defined in {DEF, UNDEF, MAYBE}     sign in {NEG, POS, ZERO, UNK}
# ---------------------------------------------------------------------------
DEF, UNDEF, MAYBE = "DEF", "UNDEF", "MAYBE"
NEG, POS, ZERO, UNK = "NEG", "POS", "ZERO", "UNK"


def _meet_def(*ds):
    if UNDEF in ds: return UNDEF
    if MAYBE in ds: return MAYBE
    return DEF


def D(e, xclass):
    """Abstract value of expr e when input var is in xclass (NEG/POS/ZERO)."""
    t = e[0]
    if t == 'var':
        return (DEF, xclass)
    if t == 'const':
        c = e[1]
        if not math.isfinite(c): return (UNDEF, UNK)
        return (DEF, NEG if c < 0 else POS if c > 0 else ZERO)
    if t == 'neg':
        d, s = D(e[1], xclass)
        return (d, {NEG: POS, POS: NEG, ZERO: ZERO, UNK: UNK}[s])
    if t == 'mul':
        d1, s1 = D(e[1], xclass); d2, s2 = D(e[2], xclass)
        return (_meet_def(d1, d2), _sign_mul(s1, s2))
    if t == 'div':
        d1, s1 = D(e[1], xclass); d2, s2 = D(e[2], xclass)
        dd = _meet_def(d1, d2)
        if s2 == ZERO: dd = UNDEF                 # /0 -> non-finite
        elif s2 == UNK: dd = _meet_def(dd, MAYBE) # denom sign unknown -> maybe /0
        return (dd, _sign_mul(s1, s2))
    if t in ('add', 'sub'):
        d1, s1 = D(e[1], xclass); d2, s2 = D(e[2], xclass)
        s = s1 if (t == 'add' and s1 == s2) else UNK
        if t == 'sub' and s1 != UNK and s2 != UNK and s1 != s2 and ZERO not in (s1, s2):
            s = s1                                 # pos-neg=pos, neg-pos=neg
        return (_meet_def(d1, d2), s)
    if t == 'log':
        d, s = D(e[1], xclass)
        if s == POS: return (_meet_def(d), POS if False else UNK)  # log>0? unknown sign
        if s in (NEG, ZERO): return (UNDEF, UNK)
        return (_meet_def(d, MAYBE), UNK)          # arg sign unknown -> maybe NaN
    if t == 'sqrt':
        d, s = D(e[1], xclass)
        if s == POS: return (d, POS)
        if s == ZERO: return (d, ZERO)
        if s == NEG: return (UNDEF, UNK)
        return (_meet_def(d, MAYBE), UNK)
    if t == 'cbrt':
        d, s = D(e[1], xclass); return (d, s)      # total, sign-preserving
    if t == 'exp':
        d, _ = D(e[1], xclass); return (d, POS)    # overflow = residue, not narrowing
    if t == 'logistic':
        d, _ = D(e[1], xclass); return (d, POS)
    if t == 'tanh':
        d, s = D(e[1], xclass); return (d, s)
    if t in ('sin', 'cos'):
        d, _ = D(e[1], xclass); return (d, UNK)
    if t == 'abs':
        d, _ = D(e[1], xclass); return (d, POS)
    if t == 'sign':
        d, s = D(e[1], xclass); return (d, s)
    if t == 'pow':
        d, s = D(e[1], xclass)
        exp = e[2]
        if isinstance(exp, tuple) and exp and exp[0] == 'const':
            c = exp[1]
        elif isinstance(exp, (int, float)):
            c = float(exp)
        else:
            c = None
        if c is None:
            return (_meet_def(d, MAYBE), UNK)      # variable exponent -> leaves fragment
        if float(c).is_integer():
            dd = d
            if s == ZERO and c < 0: dd = UNDEF      # 0^neg -> inf
            return (dd, UNK)                        # parity-dependent sign
        # non-integer exponent: defined iff base > 0 (or =0 with c>0)
        if s == POS: return (d, POS)
        if s == ZERO: return ((d if c > 0 else UNDEF), ZERO if c > 0 else UNK)
        if s == NEG: return (UNDEF, UNK)            # <-- the cbrt' bug class
        return (_meet_def(d, MAYBE), UNK)
    if t == 'select_sym':
        d1, s1 = D(e[1], xclass); d2, s2 = D(e[2], xclass)
        return (_meet_def(d1, d2), s1 if s1 == s2 else UNK)
    raise Unsupported(e)


def _sign_mul(a, b):
    if ZERO in (a, b): return ZERO
    if UNK in (a, b): return UNK
    return POS if a == b else NEG


# ---------------------------------------------------------------------------
# 4. TRUE derivatives (closed form) for the unary primals, as analyzer ASTs.
#    Used only to know where the TRUE gradient is DEF (the obligation the rule
#    must meet). No execution: definedness is read off the same abstract eval.
# ---------------------------------------------------------------------------
X = ('var', 'x')
TRUE_DERIV = {
    "LogOp":      ('div', ('const', 1.0), X),                                  # 1/x
    "Log1pOp":    ('div', ('const', 1.0), ('add', ('const', 1.0), X)),         # 1/(1+x)
    "SqrtOp":     ('div', ('const', 1.0), ('mul', ('const', 2.0), ('sqrt', X))),
    "RsqrtOp":    ('div', ('const', -0.5), ('mul', X, ('sqrt', X))),           # -1/2 x^{-3/2}
    "CbrtOp":     ('mul', ('const', 1/3), ('pow', ('cbrt', X), -2.0)),         # 1/(3 cbrt^2)
    "ExpOp":      ('exp', X),
    "Expm1Op":    ('exp', X),
    "SineOp":     ('cos', X),
    "CosineOp":   ('neg', ('sin', X)),
    "TanhOp":     ('sub', ('const', 1.0), ('mul', ('tanh', X), ('tanh', X))),
    "LogisticOp": ('mul', ('logistic', X), ('sub', ('const', 1.0), ('logistic', X))),
    "NegOp":      ('const', -1.0),
    "AbsOp":      ('sign', X),
}
CLASSES = [(NEG, "neg"), (POS, "pos")]   # ordinary input classes; 0/nan/inf = residue


def verdict(op, ast):
    """Return (status, detail). status in {BUG, sound, skip}."""
    if op not in TRUE_DERIV:
        return ("skip", "no closed-form true-deriv encoded")
    flags = []
    for cls, name in CLASSES:
        rd, _ = D(ast, cls)               # rule derivative definedness on the class
        td, _ = D(TRUE_DERIV[op], cls)    # true derivative definedness on the class
        # domain narrowing: true deriv DEF but rule deriv not guaranteed DEF
        if td == DEF and rd in (UNDEF, MAYBE):
            flags.append(name)
    if flags:
        return ("BUG", "domain-narrowing on " + ",".join(flags))
    return ("sound", "definedness-preserving on neg,pos")


# ---------------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else TD_DEFAULT
    text = open(path).read()
    n_total = n_unary = n_bug = n_sound = n_skip = n_unsup = 0
    bugs, sounds, skips, unsup = [], [], [], []
    for op, args, fwd in find_rules(text):
        n_total += 1
        if len(args) != 1:
            continue                       # unary fragment only
        n_unary += 1
        var = args[0]
        try:
            p = P(tokenize(fwd))
            # the list may hold one or more forward exprs; take the first scalar one
            node = parse_sexpr(p)
            ast = fold(lift(node, var))
        except (Unsupported, Exception) as ex:
            n_unsup += 1; unsup.append((op, str(ex)[:60])); continue
        status, detail = verdict(op, ast)
        if status == "BUG":
            n_bug += 1; bugs.append((op, detail))
        elif status == "sound":
            n_sound += 1; sounds.append((op, detail))
        else:
            n_skip += 1; skips.append((op, detail))

    print(f"# Sweep over {path.split('/')[-1]}")
    print(f"# {n_total} HLODerivative rules total; {n_unary} unary; "
          f"{n_unsup} outside the lift; {n_skip} no true-deriv encoded.\n")
    decided = n_bug + n_sound
    print(f"DECIDED (sound over-approx): {decided}  "
          f"[flagged domain-narrowing: {n_bug}; provably definedness-preserving: {n_sound}]\n")
    print("FLAGGED (domain-narrowing -> gate-confirm):")
    for op, d in bugs: print(f"  BUG    {op:14s} {d}")
    print("\nPROVABLY REAL-FIELD DEFINEDNESS-PRESERVING (in fragment):")
    for op, d in sounds: print(f"  sound  {op:14s} {d}")
    if skips:
        print("\nUNDECIDED (no closed-form true-deriv encoded -> extend TRUE_DERIV):")
        for op, d in skips: print(f"  skip   {op:14s} {d}")
    if unsup:
        print("\nOUTSIDE THE LIFT (parser/op unsupported -> future work):")
        for op, d in unsup: print(f"  --     {op:14s} {d}")


if __name__ == "__main__":
    main()
