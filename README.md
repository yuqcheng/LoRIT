# LoRIT

## Install

    pip install -r requirements.txt

## Serve a local model (optional)

    pip install vllm
    MODEL_DIR=/path/to/model PORT=8000 NOTHINK=1 ./scripts/serve_local.sh

## Configure the backend

    cp configs/backends.example.json configs/backends.json

Edit `model_url` and `model_name`. For a hosted API, put the key in the environment variable named by
`api_key_env`.

## Run

    python scripts/run_method.py demo local-model

The arguments are a dataset (`data/<name>.json`) and a backend name from `configs/backends.json`. Results are
written to `runs/<dataset>/<backend>/lorit.jsonl`; re-running skips finished items. Settings are in
`refine/configs/method.yaml`.

A dataset is a list of records `{"query": ..., "gt": ..., "source": ..., "tag": [...]}`; see `data/demo.json`.

## Score

    python scripts/score_runs.py runs/demo/local-model/lorit.jsonl

## Test

    python -m unittest tests.test_graph
