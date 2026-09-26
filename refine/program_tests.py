import os
import re
import signal
import subprocess
import sys
import tempfile
from functools import lru_cache

DEFAULT_TIMEOUT = 20

_FENCE_RE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)

def extract_code(text):
    s = str(text or "")
    blocks = [b for b in _FENCE_RE.findall(s) if b.strip()]
    if blocks:
        return blocks[-1].strip("\n")
    return s.strip()

def normalize_code(text):
    out = []
    for line in extract_code(text).splitlines():
        line = line.split("#")[0].rstrip() if "#" in line and not _in_string(line) else line.rstrip()
        if line.strip():
            out.append(line)
    return "\n".join(out)

def _in_string(line):
    i = line.find("#")
    quote = None
    for ch in line[:i]:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
    return quote is not None

@lru_cache(maxsize=8192)
def run_test(code, test, timeout=DEFAULT_TIMEOUT):
    program = extract_code(code) + "\n\n" + str(test or "")
    if not program.strip():
        return False, "empty program"
    with tempfile.TemporaryDirectory(prefix="humaneval_") as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            p = subprocess.run([sys.executable, path], cwd=tmp, timeout=timeout,
                               capture_output=True, text=True, start_new_session=True,
                               env={"PATH": os.environ.get("PATH", ""), "HOME": tmp,
                                    "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired as e:
            _kill_group(e)
            return False, f"timeout after {timeout}s"
        except Exception as e:
            return False, f"harness error: {e}"
    if p.returncode == 0:
        return True, "ok"
    tail = (p.stderr or p.stdout or "").strip().splitlines()
    return False, (tail[-1][:200] if tail else f"exit {p.returncode}")

def _kill_group(exc):
    pid = getattr(exc, "pid", None)
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
