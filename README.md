# LoRIT

Refinement-induced topology for multi-agent reasoning.

Several agents solve a problem independently. Every ordered pair of their reasoning states is judged for a
refinement relation, and the resulting digraph is condensed over its strongly connected components and
transitively reduced. That topology decides which state revises against which, and the final answer is read
out from the refinement-maximal components. No step votes, clusters answers, or counts agreement.

## Layout

    refine/
      graph.py            D, the SCC condensation C, the transitive reduction G, the execution graph H,
                          and the invariants checked on every build
      orchestrator.py     independent solving, the ordered-pair relation, synchronous revision, the rebuild
                          from final states, the readout
      backend.py          one OpenAI-compatible client for local and hosted models, per-phase token accounting
      sandbox.py          Python snippet runner used by the tool arm
      program_tests.py    pass@1 execution for code benchmarks
      answers.py          answer normalisation and rule-based scoring
      io_utils.py         backend table, resumable output, config loading
      configs/method.yaml every hyper-parameter, including which arm is run
    scripts/
      run_method.py       run the method over one dataset
      score_runs.py       accuracy and topology statistics of a run
      serve_local.sh      serve a local model, non-thinking chat template
    tests/test_graph.py   the invariants, and that the orchestration relies on them

## Configuration

Arms are selected in `refine/configs/method.yaml`, not on the command line: the pool size, the number of
revision rounds, the role set, the wording of the relation, whether the judgments go to a separate backend,
and whether agents may attach sandboxed checks whose passing results form a ledger that a revision may not
drop.

Backends are declared in `configs/backends.json`, modelled on `configs/backends.example.json`. A hosted key
belongs in an environment variable named by `api_key_env`, never in the file. `max_tokens_param` and
`no_temperature` adapt requests for reasoning models that rename the completion budget or reject a
non-default temperature.

## Data

One json file per dataset in `data/`, a list of records:

    {"query": "...", "gt": "...", "source": "GPQA" | "GSM-Hard" | "MMLU-Pro" | "HumanEval" | ...,
     "tag": ["<dataset>", "<subject>"]}

`source` selects the comparison: option letter for multiple choice, relative tolerance for numeric answers,
execution of `gt` against the candidate for code. `data/demo.json` shows the format.

## Records

Each line of a run holds the configuration, all intermediate states, every relation judgment, the graphs, the
readout candidates and per-phase token counts, so the reported tables and topology statistics are recomputable
from a run without querying a model again. Re-running resumes: finished items are skipped.

`sandbox.py` executes code written by the models under test in a fresh subprocess, in a temporary directory,
under address-space, CPU, process and file-size limits with a wall-clock timeout. That is benchmark-harness
hygiene, not a security sandbox, and no network namespace is used.
