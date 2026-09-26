import concurrent.futures
import hashlib
import json
import random
import re

from .backend import Backend
from . import graph as topo
from .program_tests import extract_code
from .answers import normalize_answer

class LoRIT(Backend):

    EPISTEMIC_ROLES = {
        "direct_solver": "You are a Direct Solver. Solve the problem head-on with the most natural method.",
        "constraint_checker": "You are a Constraint Checker. Identify every explicit and implicit constraint, then solve while verifying each one holds.",
        "counterexample_finder": "You are a Counterexample Finder. Solve the problem, then actively try to break your own answer with a counterexample or boundary case.",
        "alternative_deriver": "You are an Alternative Deriver. Solve by a route structurally different from the obvious one, and say what makes it different.",
        "evidence_grounder": "You are an Evidence Grounder. For each step be explicit about what is established, what is assumed, and what is unsupported.",
        "calibration_agent": "You are a Calibration Agent. Solve the problem and reason explicitly about where you could be wrong.",
    }

    BASELINE_ROLES = {
        "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
        "Programmer": "You are a programmer skilled in software design and debugging. You have experience in designing and developing computer software and hardware.",
        "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
        "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
        "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
        "Historian": "You are a historian. You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.",
        "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
        "Doctor": "You are a medical expert. You recommend evidence-based options and note caveats. You are able to recommend conventional medicines, herbal remedies and other natural alternatives.",
    }

    DOMAIN_ROLES = {
        "General Assistant": "You are a general assistant. You integrate knowledge across domains, break down "
                             "complex problems, and provide clear reasoning while checking assumptions.",
        "Mathematician": "You are a mathematician. You use mathematical logic, derive solutions step by step, "
                         "and verify calculations, boundary cases, and counterexamples.",
        "Economist": "You are an economist. You analyze incentives, constraints, and trade-offs using "
                     "economics, finance, and decision theory, carefully examining causal assumptions.",
        "Psychologist": "You are a psychologist. You analyze cognition, behavior, and social interactions, "
                        "considering biases, contextual factors, and alternative explanations.",
        "Programmer": "You are a programmer. You design algorithms, debug implementations, and check "
                      "correctness, efficiency, and edge cases against task requirements.",
        "Lawyer and Historian": "You are a lawyer and historian. You evaluate claims using legal principles, "
                                "historical context, and evidence, distinguishing facts from interpretations.",
        "Biologist": "You are a biologist. You apply biology, genetics, physiology, and ecology to explain "
                     "biological mechanisms and critically evaluate experimental evidence.",
        "Physicist and Chemist": "You are a physicist and chemist. You combine physical laws and chemical "
                                 "principles to analyze matter, energy, and reactions, checking units, "
                                 "conservation laws, and assumptions.",
    }

    ROLE_SETS = {"epistemic": EPISTEMIC_ROLES, "domain": DOMAIN_ROLES,
                 "domain_prior": BASELINE_ROLES}
    MC_SOURCES = {"AQUA-RAT", "MMLU", "MMLU-PRO", "GPQA"}
    NUMERIC_SOURCES = {"GSM8K", "GSM-HARD", "AIME"}
    CODE_SOURCES = {"HUMANEVAL", "SRDD", "MULTIAGENTBENCH-CODING"}
    FREEFORM_SOURCES = {"MULTIAGENTBENCH-RESEARCH": "a research proposal",
                        "MULTIAGENTBENCH-BARGAINING": "a negotiation strategy"}

    PH_INIT, PH_REL, PH_REL_FINAL, PH_REV, PH_READ = (
        "initial_reasoning", "relation_judgment", "relation_judgment_final", "revision", "final_readout")

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "method" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        mc = dict(self.method_config or {})
        for k in ("role_mode", "num_agents", "max_rounds", "temperature", "lenses",
                  "transitive_reduction", "cross_component_lifting", "max_state_chars",
                  "judge_model", "judge_max_tokens", "rel_prompt", "judge_workers",
                  "tool_checks", "max_checks"):
            v = general_config.get(k)
            if v is not None:
                mc[k] = v

        self.num_agents = int(mc.get("num_agents", 5))
        self.max_rounds = int(mc.get("max_rounds", 2))
        if self.num_agents < 1 or self.max_rounds < 0:
            raise ValueError(f"need num_agents >= 1 and max_rounds >= 0; got {self.num_agents}, {self.max_rounds}")
        self.role_mode = mc.get("role_mode", "domain")
        if self.role_mode not in self.ROLE_SETS:
            raise ValueError(f"role_mode must be one of {sorted(self.ROLE_SETS)}; got {self.role_mode!r}")
        self.LENSES = self.ROLE_SETS[self.role_mode]
        self.lenses = mc.get("lenses") or list(self.LENSES)
        unknown = [x for x in self.lenses if x not in self.LENSES]
        if unknown:
            raise ValueError(f"lenses {unknown} are not in role_mode={self.role_mode!r} "
                             f"(available: {sorted(self.LENSES)})")

        self.transitive_reduction = self._as_bool(mc.get("transitive_reduction", True))
        self.lifting = mc.get("cross_component_lifting", "all_to_all")
        if self.lifting not in topo.LIFTING_MODES:
            raise ValueError(f"cross_component_lifting must be one of {topo.LIFTING_MODES}; got {self.lifting!r}")

        self.temperature = float(mc.get("temperature", 0.7))
        self.judge_temperature = float(mc.get("judge_temperature", 0.0))
        self.judge_max_tokens = int(mc.get("judge_max_tokens", 512))
        self.tool_checks = self._as_bool(mc.get("tool_checks", False))
        self.max_checks = int(mc.get("max_checks", 3))
        self.rel_prompt = mc.get("rel_prompt", "strict")
        variants = {"strict": self._REL_SYS, "balanced": self._REL_SYS_BALANCED,
                    "obligation": self._REL_SYS_OBLIGATION}
        if self.rel_prompt not in variants:
            raise ValueError(f"rel_prompt must be one of {sorted(variants)}; got {self.rel_prompt!r}")
        self._REL_SYS = variants[self.rel_prompt]
        if self.tool_checks:
            self._REL_SYS = self._REL_SYS + self._REL_TOOL_CLAUSE
        self.judge_workers = int(mc.get("judge_workers") or 0) or None
        self.judge_model = mc.get("judge_model") or None
        if self.judge_model:
            if self.judge_model not in self.model_api_config:
                raise ValueError(f"judge_model {self.judge_model!r} is not in the model API config")
            if not all(e.get("api_key") for e in self.model_api_config[self.judge_model]["model_list"]):
                raise RuntimeError(f"judge_model {self.judge_model!r} has no api_key resolved; "
                                   "source ~/.lorit_secrets.env first")
        self.readout_temperature = float(mc.get("readout_temperature", 0.0))
        self.max_state_chars = mc.get("max_state_chars", 6000)
        self.check_invariants = self._as_bool(mc.get("check_invariants", True))
        self.max_workers = int(mc.get("max_workers", 8))
        self.random_seed = mc.get("random_seed", 2025)
        random.seed(self.random_seed)

    @staticmethod
    def _as_bool(v):
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    def inference(self, sample):
        query = sample["query"]
        source = sample.get("source")

        states = self.initialize_agents(query, source)
        log = {
            "problem_id": self._problem_id(sample),
            "config": {"num_agents": self.num_agents, "max_rounds": self.max_rounds,
                       "role_mode": self.role_mode, "judge_model": self.judge_model,
                       "rel_prompt": self.rel_prompt, "tool_checks": self.tool_checks,
                       "lenses": [self.lenses[i % len(self.lenses)] for i in range(self.num_agents)],
                       "transitive_reduction": self.transitive_reduction,
                       "cross_component_lifting": self.lifting},
            "initial_states": states,
            "rounds": [],
        }

        reusable = None
        for t in range(self.max_rounds):
            T = self.build_topology(query, states, phase=self.PH_REL)
            new_states, revised = self.revise_all_agents(query, source, states, T["H"])

            rec = {"round": t, "states_before": states}
            rec.update(self._topology_record(T))
            rec["states_after"] = new_states
            rec["revised_agents"] = revised
            rec["revision_events"] = self._revision_events(t, states, new_states, T)
            log["rounds"].append(rec)

            fixed = [self.state_signature(s) for s in states] == [self.state_signature(s) for s in new_states]
            if fixed:
                rec["stop"] = "fixed_point"
                if not revised:
                    reusable = (t, T)
                states = new_states
                break
            states = new_states

        if reusable is not None:
            final_T, final_src = reusable[1], f"round_{reusable[0]}_unrevised"
        else:
            final_T, final_src = self.final_refinement_topology(query, states), "rebuilt_from_final_states"

        answer, readout = self.refinement_readout(query, source, states, final_T)

        log.update({
            "final_states": states,
            "final_topology_source": final_src,
            "final_refinement_judgments": final_T["judgments"],
            "final_D_edges": topo.edges(final_T["D"]),
            "final_sccs": final_T["sccs"],
            "final_agent_to_component": {str(k): v for k, v in final_T["agent_to_component"].items()},
            "final_C_edges": topo.edges(final_T["C"]),
            "final_G_edges": topo.edges(final_T["G"]),
            "final_stats": final_T["stats"],
            "maximal_components": readout["maximal_components"],
            "readout_candidates": readout["candidates"],
            "readout": {k: v for k, v in readout.items() if k not in ("maximal_components", "candidates")},
            "final_answer": answer,
            "num_rounds_run": len(log["rounds"]),
            "token_usage": self._token_usage(),
        })
        return {"response": answer, "lorit": log}

    def run_checks(self, state, source=None, carried=None):
        from .sandbox import clean_snippet, run_snippet
        snippets = [clean_snippet(x) for x in (state.get("checks") or [])][:self.max_checks]
        preamble = ""
        if self._block_answer(source) and str(state.get("answer") or "").strip():
            preamble = str(state["answer"]).rstrip() + "\n\n"
        results = []
        for code in snippets:
            if not code:
                continue
            r = run_snippet(preamble + code)
            results.append({"code": code[:1200], "ok": bool(r["ok"]), "exit": r["exit"],
                            "stdout": r["stdout"][:400], "stderr": r["stderr"][:400]})
        state["check_results"] = results
        verified = list(carried or [])
        for r in results:
            if r["ok"] and r["code"] not in [v["code"] for v in verified]:
                verified.append({"code": r["code"], "stdout": r["stdout"][:200]})
        state["verified"] = verified
        state["n_failed_checks"] = sum(1 for r in results if not r["ok"])
        return state

    def initialize_agents(self, query, source):
        def one(i):
            lens = self.lenses[i % len(self.lenses)]
            raw = self.call_llm(prompt=self._solve_prompt(query, source), system_prompt=self.LENSES[lens],
                                temperature=self.temperature, phase=self.PH_INIT)
            st = self._parse_state(raw, source, lens, i)
            return self.run_checks(st, source) if self.tool_checks else st

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.num_agents, self.max_workers)) as ex:
            return list(ex.map(one, range(self.num_agents)))

    def infer_refinement(self, query, states, i, j, phase):
        prompt = (f"PROBLEM:\n{query}\n\n"
                  f"[SOURCE STATE]\n{self._fmt(states[i])}\n\n"
                  f"[TARGET STATE]\n{self._fmt(states[j])}\n\n"
                  "Can the SOURCE state concretely refine the TARGET state? Judge only SOURCE -> TARGET. "
                  "Reply with the JSON object only.")
        raw = self.call_llm(prompt=prompt, system_prompt=self._REL_SYS, model_name=self.judge_model,
                            temperature=self.judge_temperature, max_tokens=self.judge_max_tokens, phase=phase)
        refines, witness, issue, ok = self._parse_judgment(raw)
        out = {"source": i, "target": j, "refines": refines, "witness": witness, "target_issue": issue}
        if not ok:
            out["parse_error"] = True
            out["raw"] = str(raw)[:500]
        return out

    def infer_refinement_graph(self, query, states, phase):
        n = len(states)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.judge_workers or self.max_workers) as ex:
            judgments = list(ex.map(lambda p: self.infer_refinement(query, states, p[0], p[1], phase), pairs))
        return topo.refinement_graph_from_judgments(n, judgments), judgments

    def build_topology(self, query, states, phase):
        D, judgments = self.infer_refinement_graph(query, states, phase)
        sccs, a2c, C = topo.compute_scc_quotient(D)
        G = topo.compute_component_topology(C, self.transitive_reduction)
        H = topo.build_execution_graph(D, G, sccs, a2c, self.lifting)
        if self.check_invariants:
            topo.check_invariants(D, sccs, a2c, C, G, H, self.lifting)
        return {"judgments": judgments, "D": D, "sccs": sccs, "agent_to_component": a2c, "C": C, "G": G,
                "H": H, "stats": topo.topology_statistics(D, C, G, H, sccs)}

    def final_refinement_topology(self, query, states):
        return self.build_topology(query, states, phase=self.PH_REL_FINAL)

    def revise_agent(self, query, source, states, j, upstream):
        if not upstream:
            return states[j], False
        raw = self.call_llm(prompt=self._revise_prompt(query, source, states[j], [states[i] for i in upstream]),
                            system_prompt=self.LENSES[states[j]["lens"]], temperature=self.temperature,
                            phase=self.PH_REV)
        new = self._parse_state(raw, source, states[j]["lens"], j)
        if not new["answer"]:
            new = dict(states[j], revision_parse_failed=True)
        elif self.tool_checks:
            new = self.run_checks(new, source, carried=states[j].get("verified"))
        return new, True

    def revise_all_agents(self, query, source, states, H):
        frozen = tuple(states)
        jobs = [(j, sorted(H.predecessors(j))) for j in range(len(frozen))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(frozen))) as ex:
            results = list(ex.map(lambda job: self.revise_agent(query, source, frozen, job[0], job[1]), jobs))
        return [s for s, _ in results], [j for (j, _), (_, called) in zip(jobs, results) if called]

    def refinement_readout(self, query, source, states, T):
        maximal = topo.maximal_components(T["G"])
        cand = sorted(i for c in maximal for i in T["sccs"][c])
        info = {"maximal_components": maximal, "candidates": cand}
        if len(cand) == 1:
            info["mode"] = "single_candidate"
            return states[cand[0]]["answer"], info

        raw = self.call_llm(prompt=self._readout_prompt(query, source, [states[i] for i in cand]),
                            system_prompt=self._READOUT_SYS, temperature=self.readout_temperature,
                            phase=self.PH_READ)
        info["mode"] = "llm_readout"
        info["raw"] = str(raw)
        if self._block_answer(source):
            block = extract_code(str(raw or "")) if "```" in str(raw or "") else ""
            if block.strip():
                return block, info
            info["parse_error"] = True
            return str(raw or "").strip(), info
        obj, _ = self._last_json(str(raw or ""), ("answer",))
        if obj is not None and str(obj.get("answer", "")).strip():
            return str(obj["answer"]).strip(), info
        info["parse_error"] = True
        return str(raw or "").strip(), info

    def state_signature(self, s):
        norm = lambda x: " ".join(str(x).lower().split())
        return (s.get("norm", ""), norm(s.get("reasoning", "")),
                tuple(norm(x) for x in s.get("established", [])),
                tuple(norm(x) for x in s.get("open_obligations", [])))

    def _topology_record(self, T):
        return {
            "refinement_judgments": T["judgments"],
            "D_edges": topo.edges(T["D"]),
            "sccs": T["sccs"],
            "agent_to_component": {str(k): v for k, v in T["agent_to_component"].items()},
            "C_edges": topo.edges(T["C"]),
            "G_edges": topo.edges(T["G"]),
            "H_edges": [[int(i), int(j), T["H"].edges[i, j]["kind"]] for i, j in sorted(T["H"].edges)],
            "stats": T["stats"],
        }

    def _revision_events(self, t, before, after, T):
        ev = []
        for i, j in sorted(T["H"].edges):
            ev.append({"round": t, "source": i, "target": j, "R": T["D"].has_edge(i, j),
                       "h_edge_kind": T["H"].edges[i, j]["kind"],
                       "source_answer_before": before[i]["answer"],
                       "target_answer_before": before[j]["answer"],
                       "target_answer_after": after[j]["answer"],
                       "target_changed": self.state_signature(before[j]) != self.state_signature(after[j])})
        return ev

    def _token_usage(self):
        ps = self.get_phase_token_stats()
        g = lambda ph, k: ps.get(ph, {}).get(k, 0)
        rel = lambda k: g(self.PH_REL, k) + g(self.PH_REL_FINAL, k)
        return {
            "initial_reasoning_prompt_tokens": g(self.PH_INIT, "prompt_tokens"),
            "initial_reasoning_completion_tokens": g(self.PH_INIT, "completion_tokens"),
            "initial_reasoning_tokens": g(self.PH_INIT, "prompt_tokens") + g(self.PH_INIT, "completion_tokens"),
            "relation_judgment_prompt_tokens": rel("prompt_tokens"),
            "relation_judgment_completion_tokens": rel("completion_tokens"),
            "revision_prompt_tokens": g(self.PH_REV, "prompt_tokens"),
            "revision_completion_tokens": g(self.PH_REV, "completion_tokens"),
            "final_readout_prompt_tokens": g(self.PH_READ, "prompt_tokens"),
            "final_readout_completion_tokens": g(self.PH_READ, "completion_tokens"),
            "by_phase": ps,
            "context_clamped_calls": self.context_clamps,
        }

    @staticmethod
    def _problem_id(sample):
        for k in ("problem_id", "id", "idx"):
            if sample.get(k) is not None:
                return str(sample[k])
        return hashlib.sha1(str(sample["query"]).encode("utf-8")).hexdigest()[:16]

    _REL_SYS = (
        "You judge ONE directed relation between two reasoning states for the same problem: can the SOURCE "
        "state concretely refine the TARGET state?\n"
        "SOURCE refines TARGET only if SOURCE contains actionable reasoning content that corrects, completes, "
        "strengthens, or resolves a SPECIFIC weakness in TARGET, for example: correcting a concrete logical "
        "or factual error; supplying a missing derivation; resolving one of TARGET's open obligations; "
        "identifying a constraint TARGET overlooked; providing a counterexample to TARGET; repairing an "
        "invalid intermediate step.\n"
        "NOT a refinement: agreeing with TARGET; merely stating a different or the same final answer; being "
        "longer or more confident; repeating what TARGET already establishes.\n"
        "This is one direction only. Whether TARGET could refine SOURCE is a separate question and must not "
        "affect your verdict; both directions can be true at once.\n"
        "Output exactly one JSON object and nothing else:\n"
        '{"target_issue": "<the specific weakness in TARGET, or empty>", '
        '"witness": "<the concrete content in SOURCE that resolves it, or empty>", '
        '"refines": true or false}'
    )

    _REL_SYS_BALANCED = (
        "You judge ONE directed relation between two reasoning states for the same problem: would the TARGET "
        "state be improved by incorporating content the SOURCE state already has?\n"
        "Answer true if SOURCE has at least one concrete thing TARGET lacks, for example: a correction of an "
        "error in TARGET; a derivation or justification TARGET states without support or skips; a check, case "
        "or constraint TARGET does not handle; evidence that discharges one of TARGET's open obligations; a "
        "reformulation that makes a step TARGET leaves implicit verifiable.\n"
        "Answer false if SOURCE adds nothing TARGET does not already have: it merely agrees, restates the "
        "same answer, is longer or more confident, or differs only in wording.\n"
        "TARGET reaching the same final answer does NOT by itself make the answer false: judge the reasoning "
        "content, not the answer.\n"
        "This is one direction only. Whether TARGET could improve SOURCE is a separate question and must not "
        "affect your verdict; both directions can be true at once.\n"
        "Output exactly one JSON object and nothing else:\n"
        '{"target_issue": "<what TARGET lacks, or empty>", '
        '"witness": "<the content in SOURCE that supplies it, or empty>", '
        '"refines": true or false}'
    )

    _REL_TOOL_CLAUSE = (
        "\nBoth states may carry TOOL CHECKS that were executed in a sandbox. Treat them as the hardest "
        "evidence available: a FAILED check in TARGET is a concrete, named defect, and SOURCE content that "
        "would fix it is a refinement. A PASSED check in SOURCE covering something TARGET only asserts is a "
        "refinement. Passing checks in TARGET that SOURCE lacks count AGAINST refinement. Never treat a "
        "state as refining another merely because it ran more checks."
    )

    _REL_SYS_OBLIGATION = (
        "You judge ONE directed relation between two reasoning states for the same problem: does the SOURCE "
        "state resolve something the TARGET state leaves open or unsupported?\n"
        "Answer true if SOURCE settles a point TARGET leaves unsettled: one of TARGET's open obligations, a "
        "step TARGET asserts without justification, a case or constraint TARGET does not check, or an error "
        "in TARGET. Being more complete on a specific point is enough; SOURCE need not be better overall.\n"
        "Answer false if TARGET already establishes everything SOURCE offers, or SOURCE only agrees, repeats, "
        "or is merely more verbose.\n"
        "This is one direction only; both directions can be true at once.\n"
        "Output exactly one JSON object and nothing else:\n"
        '{"target_issue": "<the open or unsupported point in TARGET, or empty>", '
        '"witness": "<what in SOURCE settles it, or empty>", '
        '"refines": true or false}'
    )

    _READOUT_SYS = (
        "You produce the final answer of a multi-agent reasoning system from its remaining candidate "
        "reasoning states. You judge reasoning validity, never popularity."
    )

    def _is_code(self, source):
        return (source or "").upper() in self.CODE_SOURCES

    def _is_freeform(self, source):
        return (source or "").upper() in self.FREEFORM_SOURCES

    def _block_answer(self, source):
        return self._is_code(source) or self._is_freeform(source)

    def _answer_format(self, source):
        src = (source or "").upper()
        if src in self.MC_SOURCES:
            return 'the option letter in parentheses, e.g. "(B)"'
        if src in self.NUMERIC_SOURCES:
            return 'the final numeric value only, e.g. "18"'
        return "your final answer, as short as possible"

    _CODE_CHECK_NOTE = (
        "\nAlso add a \"checks\" list to that JSON object: at most {n} self-contained Python snippets that "
        "define or import your implementation and then assert properties of it (edge cases, types, small "
        "examples you worked out yourself). Each is run in a sandbox; a failing one is evidence against your "
        "implementation. Do NOT invent or guess the grader's hidden tests -- write your own."
    )

    _CODE_CONTRACT = (
        "Then give your FINAL IMPLEMENTATION as a single ```python code block: the complete function, "
        "including its signature and every import it needs, ready to run as-is. No tests, no example calls, "
        "no explanation inside the block.\n"
        "After that block, output ONE JSON object on the LAST line, no code fences:\n"
        '{"established": ["<a step or property you have actually established>"],\n'
        ' "open_obligations": ["<something you have NOT verified: an unhandled case, an assumption, an '
        'untested branch>"]}\n'
        "Be honest in open_obligations. Naming a genuine gap is more useful than claiming none."
    )

    _FREEFORM_CONTRACT = (
        "Then give your FINAL {what} as a single fenced block:\n```\n<the complete {what}, self-contained, "
        "ready to hand over>\n```\n"
        "After that block, output ONE JSON object on the LAST line, no code fences:\n"
        '{{"established": ["<a point you have actually settled>"],\n'
        ' "open_obligations": ["<something you have NOT settled: an open question, an assumption, a gap>"]}}\n'
        "Be honest in open_obligations. Naming a genuine gap is more useful than claiming none."
    )

    _CHECK_CLAUSE = (
        '\n "checks": ["<a self-contained Python snippet that VERIFIES one of your claims: it must exit '
        'non-zero (e.g. raise AssertionError) if the claim is false, and print nothing sensitive. No network, '
        'no file access, no input()>"]'
    )

    def _checks_note(self, n):
        return (f"\nAdd at most {n} entries to \"checks\". Each one is run in a sandbox and its result is "
                "attached to your state: a passing check marks the claim it tests as verified, a failing check "
                "is evidence against your own reasoning and you should say so in open_obligations. Write checks "
                "that could actually fail -- a check that trivially passes verifies nothing.")

    def _contract(self, source):
        if self._is_code(source) and self.tool_checks:
            return (self._CODE_CONTRACT.replace(
                '{"established": ["<a step or property you have actually established>"],\n',
                '{"established": ["<a step or property you have actually established>"],\n')
                + self._CODE_CHECK_NOTE.format(n=self.max_checks))
        if self._is_freeform(source):
            return self._FREEFORM_CONTRACT.format(what=self.FREEFORM_SOURCES[(source or "").upper()])
        if self._is_code(source):
            return self._CODE_CONTRACT
        base = ("Then output your reasoning state as ONE JSON object on the LAST line, no code fences:\n"
                '{"answer": "<' + self._answer_format(source) + '>",\n'
                ' "established": ["<a step or fact you have actually established>"],\n'
                ' "open_obligations": ["<something you have NOT verified: an unchecked case, an assumption, '
                'an incomplete step>"]'
                + (self._CHECK_CLAUSE if self.tool_checks else "") + '}\n'
                "Be honest in open_obligations. Naming a genuine gap is more useful than claiming none.")
        return base + (self._checks_note(self.max_checks) if self.tool_checks else "")

    def _solve_prompt(self, query, source):
        if self.role_mode == "domain":
            return ("You will independently attempt the user's task first. Let's think step by step.\n"
                    "Be precise and complete.\n"
                    f"\n[Task]\n{query}\n\nWrite your reasoning first. " + self._contract(source))
        return ("Solve the following problem independently. Think step by step and write out your reasoning, "
                f"then commit to an answer.\n\n{query}\n\n" + self._contract(source))

    def _revise_prompt(self, query, source, own, upstream):
        blocks = "\n\n".join(f"[Upstream state {k + 1}]\n{self._fmt(s)}" for k, s in enumerate(upstream))
        return (
            "You are revising your own reasoning state for the problem below.\n\n"
            f"PROBLEM:\n{query}\n\n"
            f"=== CURRENT STATE (yours) ===\n{self._fmt(own)}\n\n"
            "=== UPSTREAM REFINEMENT STATES ===\n"
            "These states are upstream of yours in the refinement structure: they may contain corrections or "
            "information your state is missing. They are not guaranteed to be correct.\n\n"
            f"{blocks}\n\n"
            "=== INSTRUCTIONS ===\n"
            + ("- Your state carries TOOL CHECKS that passed. Those claims are verified: keep them, and keep "
               "any implementation detail they depend on. You may add to them, you may not drop or contradict "
               "them without a check of your own that fails.\n"
               "- If one of YOUR checks failed, treat that as a concrete defect and fix it first.\n"
               if self.tool_checks else "") +
            "- Preserve the parts of your current reasoning that are already valid.\n"
            "- Inspect each upstream state for concrete corrections, missing derivations, overlooked "
            "constraints, counterexamples, or content that resolves one of your open obligations.\n"
            "- Revise only where you can verify from the problem itself that the change is justified.\n"
            "- Resolve your open obligations when possible; keep those that remain unresolved.\n"
            "- Do not blindly follow an upstream state. Agreement among several states is not evidence: do "
            "not count how many states hold an answer.\n"
            "- Produce a COMPLETE updated reasoning state, not a list of changes.\n\n"
            "Write your full updated reasoning. " + self._contract(source)
        )

    def _readout_prompt(self, query, source, cands):
        blocks = "\n\n".join(f"[Candidate state {k + 1}]\n{self._fmt(s)}" for k, s in enumerate(cands))
        return (
            f"PROBLEM:\n{query}\n\n"
            "The reasoning states below are the refinement-maximal candidates: no other state in the system was "
            "judged able to refine them. They may disagree.\n\n"
            f"{blocks}\n\n"
            "=== INSTRUCTIONS ===\n"
            "- Do not count votes. Do not prefer an answer because more candidate states support it.\n"
            "- Inspect the logical and factual support each state gives for its answer.\n"
            "- Where candidates conflict, find the step where they diverge and resolve it using the reasoning "
            "content.\n"
            "- Synthesize ONE final answer. Do not rank the candidates.\n\n"
            + ("Briefly compare the candidates, then give the final implementation as a single ```python code "
               "block: the complete function with its imports, ready to run as-is, no tests and no commentary "
               "inside the block."
               if self._is_code(source) else
               f"Briefly compare the candidates, then give the final "
               f"{self.FREEFORM_SOURCES.get((source or '').upper(), 'answer')} as a single fenced block."
               if self._is_freeform(source) else
               "Briefly compare the reasoning, then output ONE JSON object on the LAST line, no code fences:\n"
               '{"answer": "<' + self._answer_format(source) + '>"}')
        )

    def _checks_block(self, s):
        if not self.tool_checks:
            return ""
        res = s.get("check_results") or []
        ver = s.get("verified") or []
        if not res and not ver:
            return "\nTOOL CHECKS: none run\n"
        lines = []
        for r in res:
            head = "PASSED" if r["ok"] else "FAILED"
            detail = (r["stdout"] or "").strip().replace("\n", " ")[:120] if r["ok"] else \
                     (r["stderr"] or "").strip().splitlines()[-1][:160] if (r["stderr"] or "").strip() else ""
            lines.append(f"  [{head}] {r['code'].splitlines()[0][:90]} -> {detail}")
        extra = len(ver) - sum(1 for r in res if r["ok"])
        if extra > 0:
            lines.append(f"  [VERIFIED EARLIER] {extra} check(s) from a previous round still hold")
        return "\nTOOL CHECKS (run in a sandbox, not written by the grader):\n" + "\n".join(lines) + "\n"

    def _fmt(self, s):
        if "\n" in str(s["answer"]):
            return (f"IMPLEMENTATION:\n```python\n{s['answer']}\n```\n"
                    f"REASONING:\n{self._clip(s.get('reasoning', '')) or '(none)'}\n"
                    "ESTABLISHED:\n" + ("\n".join(f"  - {x}" for x in s["established"]) or "  (none)")
                    + "\nOPEN OBLIGATIONS:\n" + ("\n".join(f"  - {x}" for x in s["open_obligations"]) or "  (none)")
                    + self._checks_block(s))
        return (f"ANSWER: {s['answer']}\n"
                f"REASONING:\n{self._clip(s.get('reasoning', '')) or '(none)'}\n"
                "ESTABLISHED:\n" + ("\n".join(f"  - {x}" for x in s["established"]) or "  (none)")
                + "\nOPEN OBLIGATIONS:\n" + ("\n".join(f"  - {x}" for x in s["open_obligations"]) or "  (none)")
                + self._checks_block(s))

    def _clip(self, text):
        text = str(text or "").strip()
        m = self.max_state_chars
        if not m or len(text) <= m:
            return text
        half = int(m) // 2
        return text[:half] + "\n[... middle of reasoning truncated ...]\n" + text[-half:]

    _BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')

    def _last_json(self, text, keys):
        dec = json.JSONDecoder()
        found = (None, None)
        for m in re.finditer(r"\{", text):
            span = text[m.start():]
            obj = None
            for candidate in (span, self._BAD_ESCAPE.sub(r"\\\\", span)):
                try:
                    obj, _ = dec.raw_decode(candidate)
                    break
                except ValueError:
                    continue
            if isinstance(obj, dict) and any(k in obj for k in keys):
                found = (obj, m.start())
        return found

    _FIELD_ANSWER = re.compile(r'\{\s*"(?:answer|claim)"\s*:\s*"((?:[^"\\]|\\.)*)"')

    def _salvage_state(self, text):
        hits = list(self._FIELD_ANSWER.finditer(text))
        if not hits:
            return None, None
        m = hits[-1]
        tail = text[m.start():]
        obj = {"answer": m.group(1).replace('\\"', '"')}
        for key in ("established", "open_obligations"):
            lm = re.search(r'"' + key + r'"\s*:\s*\[(.*?)\]\s*[\]},]', tail, re.S)
            obj[key] = re.findall(r'"((?:[^"\\]|\\.)*)"', lm.group(1)) if lm else []
        return obj, m.start()

    def _parse_state(self, raw, source, lens, agent):
        if self._block_answer(source):
            return self._parse_code_state(raw, source, lens, agent)
        text = str(raw or "")
        obj, start = self._last_json(text, ("answer", "claim"))
        status = None
        if obj is None:
            obj, start = self._salvage_state(text)
            status = "state_parse_salvaged"
        if obj is None:
            lines = [x.strip() for x in text.strip().splitlines() if x.strip() and not x.strip().startswith("```")]
            obj, reasoning, status = {"answer": lines[-1] if lines else ""}, text.strip(), "state_parse_failed"
        else:
            reasoning = re.sub(r"```(?:json)?\s*$", "", text[:start]).strip()
        answer = str(obj.get("answer", obj.get("claim", ""))).strip()
        state = {"agent": agent, "lens": lens, "answer": answer, "norm": normalize_answer(answer, source),
                 "reasoning": reasoning,
                 "established": self._strs(obj.get("established")),
                 "open_obligations": self._strs(obj.get("open_obligations")),
                 "checks": self._snippets(obj.get("checks"), limit=self.max_checks)}
        if status:
            state[status] = True
        return state

    def _parse_code_state(self, raw, source, lens, agent):
        text = str(raw or "")
        code = extract_code(text) if "```" in text else ""
        obj, start = self._last_json(text, ("established", "open_obligations", "answer"))
        if not code and obj is not None and str(obj.get("answer", "")).strip():
            code = extract_code(str(obj["answer"]))
        reasoning = text[:text.find("```")].strip() if "```" in text else text.strip()
        state = {"agent": agent, "lens": lens, "answer": code, "norm": normalize_answer(code, source),
                 "reasoning": reasoning,
                 "established": self._strs((obj or {}).get("established")),
                 "open_obligations": self._strs((obj or {}).get("open_obligations")),
                 "checks": self._snippets((obj or {}).get("checks"), limit=self.max_checks)}
        if not code:
            state["state_parse_failed"] = True
        return state

    def _parse_judgment(self, raw):
        text = str(raw or "")
        obj, _ = self._last_json(text, ("refines",))
        if obj is not None:
            v = obj.get("refines")
            lv = v.strip().lower() if isinstance(v, str) else None
            refines = v is True or lv in ("true", "yes")
            ok = isinstance(v, bool) or lv in ("true", "false", "yes", "no")
            return refines, str(obj.get("witness", "") or ""), str(obj.get("target_issue", "") or ""), ok
        m = re.search(r'"?refines"?\s*:\s*(true|false)', text, re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true", "", "", True
        return False, "", "", False

    def _snippets(self, x, limit=3):
        if x is None:
            return []
        items = [x] if isinstance(x, str) else (list(x) if isinstance(x, (list, tuple)) else [x])
        out = []
        for it in items:
            t = str(it).strip()[:2000]
            if t and t not in out:
                out.append(t)
            if len(out) >= limit:
                break
        return out

    def _strs(self, x, limit=8):
        if x is None:
            return []
        items = [x] if isinstance(x, str) else (list(x) if isinstance(x, (list, tuple)) else [x])
        out = []
        for it in items:
            s = " ".join(str(it).split())[:300]
            if s and s not in out:
                out.append(s)
            if len(out) >= limit:
                break
        return out
