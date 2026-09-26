import json
import os
import sys
import threading
import concurrent.futures
import traceback

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refine import LoRIT
from refine.io_utils import load_backends, load_config, reserve_unprocessed_queries, write_to_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "refine", "configs", "method.yaml")
BACKENDS = os.path.join(ROOT, "configs", "backends.json")

def main():
    if len(sys.argv) != 3:
        sys.exit("usage: run_method.py DATASET BACKEND")
    dataset, backend = sys.argv[1], sys.argv[2]
    cfg = load_config(CONFIG)
    general = {
        "model_api_config": load_backends(BACKENDS, backend),
        "model_name": backend,
        "model_temperature": cfg["temperature"],
        "model_max_tokens": cfg["max_tokens"],
        "model_timeout": cfg["timeout"],
    }
    mas = LoRIT(general)
    with open(os.path.join(ROOT, "data", f"{dataset}.json")) as f:
        data = json.load(f)
    out = os.path.join(ROOT, "runs", dataset, backend, "lorit.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    todo = reserve_unprocessed_queries(out, data)
    lock = threading.Lock()

    def one(sample):
        try:
            rec = dict(sample)
            rec.update(mas.inference(sample))
            rec["token_stats"] = mas.get_token_stats()
        except Exception:
            rec = dict(sample)
            rec["error"] = traceback.format_exc()
        mas.reset_token_stats()
        write_to_jsonl(lock, out, rec)

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["item_workers"]) as ex:
        list(tqdm(ex.map(one, todo), total=len(todo)))

if __name__ == "__main__":
    main()
