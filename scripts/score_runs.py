import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refine.answers import score_run

def load(path):
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out

def topology_stats(records):
    ok = [r for r in records if not r.get("error") and r.get("lorit")]
    if not ok:
        return {}
    n = len(ok[0]["lorit"]["config"]["lenses"]) or 1
    pairs = max(n * (n - 1), 1)

    def mean(f):
        return round(st.mean(f(r) for r in ok), 3)

    return {"num_agents": n,
            "D_density": mean(lambda r: r["lorit"]["final_stats"]["E_D"] / pairs),
            "num_sccs": mean(lambda r: len(r["lorit"]["final_sccs"])),
            "G_edges": mean(lambda r: r["lorit"]["final_stats"]["E_G"]),
            "readout_candidates": mean(lambda r: len(r["lorit"]["readout_candidates"])),
            "rounds_run": mean(lambda r: r["lorit"]["num_rounds_run"])}

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: score_runs.py RUN.jsonl [RUN.jsonl ...]")
    for pattern in sys.argv[1:]:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            records = load(path)
            if not records:
                continue
            m = score_run(records)
            m.update(topology_stats(records))
            m["file"] = os.path.relpath(path)
            print(json.dumps(m, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
