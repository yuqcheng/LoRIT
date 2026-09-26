import os
import re
import signal
import subprocess
import sys
import tempfile
from functools import lru_cache

TIMEOUT = 10
AS_BYTES = 2 * 1024 ** 3
CPU_SECONDS = 10
MAX_PROCS = 64
MAX_FILE = 10 * 1024 ** 2
MAX_OUTPUT = 2000

_PRLIMIT = ["prlimit", f"--as={AS_BYTES}", f"--cpu={CPU_SECONDS}",
            f"--nproc={MAX_PROCS}", f"--fsize={MAX_FILE}"]

def _kill_group(proc_exc):
    try:
        os.killpg(os.getpgid(proc_exc.args[0] if isinstance(proc_exc.args, tuple) else 0), signal.SIGKILL)
    except Exception:
        pass

@lru_cache(maxsize=16384)
def run_snippet(code, timeout=TIMEOUT):
    code = str(code or "").strip()
    if not code:
        return {"ok": False, "exit": None, "stdout": "", "stderr": "empty snippet"}
    with tempfile.TemporaryDirectory(prefix="toolcheck_") as tmp:
        path = os.path.join(tmp, "check.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            p = subprocess.run(_PRLIMIT + [sys.executable, path], cwd=tmp, timeout=timeout,
                               capture_output=True, text=True, start_new_session=True,
                               env={"PATH": os.environ.get("PATH", ""), "HOME": tmp,
                                    "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired:
            return {"ok": False, "exit": None, "stdout": "", "stderr": f"timeout after {timeout}s"}
        except Exception as e:
            return {"ok": False, "exit": None, "stdout": "", "stderr": f"harness error: {e}"}
    return {"ok": p.returncode == 0, "exit": p.returncode,
            "stdout": p.stdout[-MAX_OUTPUT:], "stderr": p.stderr[-MAX_OUTPUT:]}

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

def clean_snippet(text):
    s = str(text or "").strip()
    m = _FENCE.search(s)
    return (m.group(1) if m else s).strip()
