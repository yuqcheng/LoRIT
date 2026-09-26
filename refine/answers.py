import re

_CHOICE_RE = re.compile(r"\(([A-J])\)")
_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
_OPTION_LINE_RE = re.compile(r"^\s*\(?([A-J])[\)\.]\s*(.+?)\s*$")

MC_SOURCES = {"AQUA-RAT", "MMLU", "MMLU-PRO", "GPQA"}
NUMERIC_SOURCES = {"GSM8K", "GSM-HARD", "AIME", "MC", ""}
CODE_SOURCES = {"HUMANEVAL"}
GENERATION_SOURCES = {"SRDD", "MULTIAGENTBENCH-CODING", "MULTIAGENTBENCH-RESEARCH",
                      "MULTIAGENTBENCH-BARGAINING"}
GENERATION_CODE_SOURCES = {"SRDD", "MULTIAGENTBENCH-CODING"}

def parse_options(query):
    opts = {}
    for ln in str(query).splitlines():
        m = _OPTION_LINE_RE.match(ln)
        if m:
            opts[m.group(1)] = m.group(2).strip()
    return opts

def _value_key(x):
    s = str(x).lower().strip().replace("√", "sqrt").replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9.\-+/]", "", s)

def answer_to_letter(ans, query):
    if ans is None:
        return None
    s = str(ans).strip()
    m = _CHOICE_RE.search(s)
    if m:
        return m.group(1)
    opts = parse_options(query)
    m2 = re.match(r"^\(?([A-J])\)?\b", s)
    if m2 and (not opts or m2.group(1) in opts):
        return m2.group(1)
    if opts:
        rv = _value_key(s)
        if rv:
            hits = [L for L, t in opts.items() if _value_key(t) == rv]
            if len(hits) == 1:
                return hits[0]
    return None

def normalize_answer(ans, source=None):
    if ans is None:
        return ""
    s = str(ans).strip()
    src = (source or "").upper()
    if src in CODE_SOURCES:
        from .program_tests import normalize_code
        return normalize_code(s)
    if src in MC_SOURCES:
        m = _CHOICE_RE.search(s) or re.search(r"\b([A-J])\b", s)
        return m.group(1) if m else s.lower()
    if src in {"GSM8K", "GSM-HARD", "AIME"} or src == "":
        if "####" in s:
            s = s.split("####")[-1].strip()
        clean = s.replace("$", "").replace(",", "").replace(" ", "")
        fm = re.search(r"(-?\d+)\s*/\s*(\d+)", clean)
        if fm:
            num, den = int(fm.group(1)), int(fm.group(2))
            if den:
                try:
                    v = num / den
                    return str(int(v)) if v == int(v) else f"{v:.6g}"
                except OverflowError:
                    pass
            return f"{num}/{den}"
        nums = _NUM_RE.findall(clean)
        if nums:
            try:
                f = float(nums[-1])
                return str(int(f)) if f == int(f) else f"{f:.6g}"
            except (ValueError, OverflowError):
                return nums[-1]
    return s.lower()

def _to_float(s):
    try:
        return float(str(s).replace(",", "").replace("$", ""))
    except (ValueError, TypeError):
        return None

def is_correct(pred, gt, source, rel_tol=1e-3, query=None):
    src = (source or "").upper()
    if src in GENERATION_SOURCES:
        raise ValueError(f"{source} has no reference answer; use generation_metrics(), not is_correct()")
    if src in CODE_SOURCES:
        from .program_tests import run_test
        return run_test(str(pred or ""), str(gt or ""))[0]
    if src in MC_SOURCES and query is not None:
        gl, pl = answer_to_letter(gt, query), answer_to_letter(pred, query)
        if gl is not None and pl is not None:
            return pl == gl
    p, g = normalize_answer(pred, source), normalize_answer(gt, source)
    if p == g and p != "":
        return True
    if src in NUMERIC_SOURCES:
        pf, gf = _to_float(p), _to_float(g)
        if pf is not None and gf is not None:
            return abs(pf) < 1e-9 if gf == 0 else abs(pf - gf) / (abs(gf) + 1e-12) <= rel_tol
    return False

def generation_metrics(records):
    import ast
    from .program_tests import extract_code
    n = len(records)
    if not n:
        return {"n": 0}
    is_code = (records[0].get("source") or "").upper() in GENERATION_CODE_SOURCES
    ok = empty = errors = 0
    lengths = []
    for r in records:
        if r.get("error"):
            errors += 1
            continue
        answer = extract_code(r.get("response", ""))
        if not answer.strip():
            empty += 1
            continue
        lengths.append(len(answer.splitlines()))
        if is_code:
            try:
                ast.parse(answer)
                ok += 1
            except SyntaxError:
                pass
    scored = n - errors
    lengths.sort()
    out = {"n": n, "n_errors": errors, "n_empty_answers": empty,
           "answered_rate": round((scored - empty) / scored, 4) if scored else 0.0,
           "median_lines": lengths[len(lengths) // 2] if lengths else 0,
           "note": "no reference answer: this is a generation source, so there is no accuracy"}
    if is_code:
        out["syntax_ok"] = ok
        out["syntax_ok_rate"] = round(ok / scored, 4) if scored else 0.0
    return out

def score_run(records):
    n = len(records)
    if not n:
        return {"n": 0, "accuracy": 0.0}
    if (records[0].get("source") or "").upper() in GENERATION_SOURCES:
        m = generation_metrics(records)
        calls = sum(v.get("num_llm_calls", 0) for r in records for v in (r.get("token_stats") or {}).values())
        m["calls_per_item"] = round(calls / n, 2)
        return m
    correct = errors = calls = ptok = ctok = 0
    for r in records:
        if r.get("error"):
            errors += 1
            continue
        if is_correct(r.get("response", ""), r["gt"], r.get("source"), query=r.get("query")):
            correct += 1
        for v in (r.get("token_stats") or {}).values():
            calls += v.get("num_llm_calls", 0)
            ptok += v.get("prompt_tokens", 0)
            ctok += v.get("completion_tokens", 0)
    scored = n - errors
    return {"n": n, "n_scored": scored, "n_errors": errors,
            "accuracy": round(correct / n, 4), "correct": correct,
            "accuracy_scored": round(correct / scored, 4) if scored else 0.0,
            "calls_per_item": round(calls / n, 2),
            "prompt_tokens_per_item": round(ptok / n, 1),
            "completion_tokens_per_item": round(ctok / n, 1),
            "total_calls": calls}
