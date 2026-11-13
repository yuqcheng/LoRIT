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

The default configuration is the low-cost one:

- `stop_on_unanimity: true`. The run stops once every agent gives the same answer, and the remaining rounds,
  the final topology and the readout are skipped.
- `lazy_relations: true`. Ordered pairs are judged in batches only until the topology is fixed: the SCC
  partition, every intra-component pair, and every condensation edge for the execution graph; the source
  components for the readout. Every pair is a separate call at temperature 0, so this gives the same execution
  graphs and readout candidates as judging every pair. With `direct` lifting all pairs are judged.
- `rel_prompt: concise_verdict`. A short wording of the relation that asks for the verdict only.
- `judge_reasoning_chars: 0`. The judge sees each state's answer, established facts and open obligations,
  without the free-text reasoning.
- `max_state_chars: 400`. At most 400 characters of a state's reasoning, head and tail, are printed into
  another agent's revision prompt.

For the full configuration, set `rel_prompt: strict`, `stop_on_unanimity: false`, `lazy_relations: false`,
`judge_reasoning_chars: null` and `max_state_chars: 6000`.

Backends are declared in `configs/backends.json`, modelled on `configs/backends.example.json`. A hosted key
belongs in an environment variable named by `api_key_env`, never in the file. `max_tokens_param` and
`no_temperature` adapt requests for reasoning models that rename the completion budget or reject a
non-default temperature.

## Serving a local model

    MODEL_DIR=/path/to/model PORT=8000 NOTHINK=1 ./scripts/serve_local.sh

`NOTHINK=1` applies `scripts/nothink.jinja`, which keeps hybrid-reasoning models in their non-thinking mode.
`GPU_UTIL` (default 0.85) is a fraction of the whole card; lower it when the card is shared.

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
