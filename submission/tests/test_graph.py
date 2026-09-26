import itertools
import os
import random
import re
import sys
import threading
import unittest

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from refine import graph as topo
from refine.orchestrator import LoRIT as LoRIT

def judgments_from_edges(n, edge_list):
    E = set(edge_list)
    return [{"source": i, "target": j, "refines": (i, j) in E, "witness": "", "target_issue": ""}
            for i in range(n) for j in range(n) if i != j]

def pipeline(n, edge_list, transitive_reduction=True, lifting="all_to_all"):
    D = topo.refinement_graph_from_judgments(n, judgments_from_edges(n, edge_list))
    sccs, a2c, C = topo.compute_scc_quotient(D)
    G = topo.compute_component_topology(C, transitive_reduction)
    H = topo.build_execution_graph(D, G, sccs, a2c, lifting)
    return D, sccs, a2c, C, G, H

def random_digraph_edges(rng, n, p):
    return [(i, j) for i in range(n) for j in range(n) if i != j and rng.random() < p]

class TestRequiredInvariants(unittest.TestCase):

    def setUp(self):
        self.rng = random.Random(0)
        self.cases = [(n, random_digraph_edges(self.rng, n, p))
                      for n in (1, 2, 3, 5, 7) for p in (0.0, 0.1, 0.3, 0.6, 1.0) for _ in range(12)]

    def test_1_condensation_is_acyclic(self):
        for n, E in self.cases:
            _, _, _, C, _, _ = pipeline(n, E)
            self.assertTrue(nx.is_directed_acyclic_graph(C))

    def test_2_G_is_acyclic(self):
        for n, E in self.cases:
            for tr in (True, False):
                _, _, _, _, G, _ = pipeline(n, E, tr)
                self.assertTrue(nx.is_directed_acyclic_graph(G))

    def test_3_G_preserves_condensation_reachability(self):
        for n, E in self.cases:
            for tr in (True, False):
                _, _, _, C, G, _ = pipeline(n, E, tr)
                for u, v in itertools.product(C.nodes, C.nodes):
                    self.assertEqual(nx.has_path(C, u, v), nx.has_path(G, u, v))

    def test_4_intra_scc_execution_edges(self):
        for n, E in self.cases:
            for lifting in topo.LIFTING_MODES:
                D, _, a2c, _, _, H = pipeline(n, E, lifting=lifting)
                for i, j in itertools.permutations(D.nodes, 2):
                    if a2c[i] == a2c[j]:
                        self.assertEqual(H.has_edge(i, j), D.has_edge(i, j))

    def test_5_hasse_lifting(self):
        for n, E in self.cases:
            for tr in (True, False):
                _, sccs, _, _, G, H = pipeline(n, E, tr, "all_to_all")
                for a, b in G.edges:
                    for i in sccs[a]:
                        for j in sccs[b]:
                            self.assertTrue(H.has_edge(i, j))

    def test_6_agent_level_reachability(self):
        for n, E in self.cases:
            for tr in (True, False):
                D, _, _, _, _, H = pipeline(n, E, tr, "all_to_all")
                for i, j in itertools.product(D.nodes, D.nodes):
                    if i != j:
                        self.assertEqual(nx.has_path(D, i, j), nx.has_path(H, i, j))

    def test_runtime_checker_accepts_all_cases(self):
        for n, E in self.cases:
            for tr, lifting in itertools.product((True, False), topo.LIFTING_MODES):
                D, sccs, a2c, C, G, H = pipeline(n, E, tr, lifting)
                topo.check_invariants(D, sccs, a2c, C, G, H, lifting)

    def test_runtime_checker_catches_a_broken_H(self):
        D, sccs, a2c, C, G, H = pipeline(3, [(0, 1), (1, 2)])
        H.remove_edge(1, 2)
        with self.assertRaises(AssertionError):
            topo.check_invariants(D, sccs, a2c, C, G, H)

class TestToyGraphs(unittest.TestCase):
    A, B, C_ = 0, 1, 2

    def test_7_mutual_refinement_forms_one_scc(self):
        D, sccs, a2c, C, G, H = pipeline(2, [(self.A, self.B), (self.B, self.A)])
        self.assertEqual(a2c[self.A], a2c[self.B])
        self.assertEqual(sccs, [[0, 1]])
        self.assertEqual(set(H.edges), {(0, 1), (1, 0)})

    def test_8_transitive_reduction(self):
        E = [(self.A, self.B), (self.B, self.C_), (self.A, self.C_)]
        D, sccs, a2c, C, G, H = pipeline(3, E)
        g = {(sccs[u][0], sccs[v][0]) for u, v in G.edges}
        self.assertIn((self.A, self.B), g)
        self.assertIn((self.B, self.C_), g)
        self.assertNotIn((self.A, self.C_), g)
        _, sccs2, _, _, G2, _ = pipeline(3, E, transitive_reduction=False)
        self.assertIn((self.A, self.C_), {(sccs2[u][0], sccs2[v][0]) for u, v in G2.edges})

    def test_9_readout_orientation(self):
        D, sccs, a2c, C, G, H = pipeline(3, [(self.A, self.B), (self.B, self.C_)])
        maximal = topo.maximal_components(G)
        self.assertEqual([m for c in maximal for m in sccs[c]], [self.A])

    def test_independent_directions_only_one_way(self):
        D, *_ = pipeline(2, [(0, 1)])
        self.assertTrue(D.has_edge(0, 1))
        self.assertFalse(D.has_edge(1, 0))

    def test_lifting_across_scc(self):
        E = [(0, 1), (1, 0), (2, 3), (3, 2), (1, 2)]
        D, sccs, a2c, C, G, H = pipeline(4, E, lifting="all_to_all")
        self.assertEqual(sccs, [[0, 1], [2, 3]])
        self.assertEqual({(i, j) for i, j in H.edges if a2c[i] != a2c[j]}, {(0, 2), (0, 3), (1, 2), (1, 3)})
        self.assertEqual({H.edges[e]["kind"] for e in [(0, 2), (1, 3)]}, {topo.H_LIFTED})
        D, sccs, a2c, C, G, H = pipeline(4, E, lifting="direct")
        self.assertEqual({(i, j) for i, j in H.edges if a2c[i] != a2c[j]}, {(1, 2)})

    def test_statistics(self):
        _, sccs, _, C, G, H = pipeline(3, [(0, 1), (1, 2), (0, 2)])
        D = topo.refinement_graph_from_judgments(3, judgments_from_edges(3, [(0, 1), (1, 2), (0, 2)]))
        st = topo.topology_statistics(D, C, G, H, sccs)
        self.assertEqual((st["E_D"], st["E_C"], st["E_G"], st["E_H"]), (3, 3, 2, 2))
        self.assertEqual(st["num_sccs"], 3)
        self.assertEqual(st["scc_sizes"], [1, 1, 1])
        self.assertEqual(st["num_maximal_components"], 1)
        self.assertAlmostEqual(st["transitive_reduction_ratio"], 1 / 3)
        _, sccs, _, C, G, H = pipeline(2, [])
        D = topo.refinement_graph_from_judgments(2, [])
        self.assertIsNone(topo.topology_statistics(D, C, G, H, sccs)["transitive_reduction_ratio"])

LENS_ANSWERS = {}

class FakeLoRIT(LoRIT):

    REFINES = set()

    def __init__(self, **overrides):
        cfg = {"model_api_config": {"fake": {"model_list": [], "max_workers": 1}}, "model_name": "fake",
               "model_temperature": 0.7, "model_max_tokens": 100, "model_timeout": 1}
        cfg.update(overrides)
        super().__init__(cfg)
        self.prompts = {"judge": [], "revise": [], "readout": []}
        self._lock = threading.Lock()

    def call_llm(self, prompt=None, system_prompt=None, temperature=None, max_tokens=None, phase=None, **kw):
        ans = lambda block: re.search(r"ANSWER: (.*)", block).group(1).strip()
        with self._lock:
            self.phase_token_stats.setdefault(phase, {"num_llm_calls": 0, "prompt_tokens": 0,
                                                      "completion_tokens": 0})["num_llm_calls"] += 1
        if phase == self.PH_INIT:
            lens = next(k for k, v in self.LENSES.items() if v == system_prompt)
            return f"I reason carefully.\n{{\"answer\": \"{LENS_ANSWERS[lens]}\", \"established\": [\"x\"], \"open_obligations\": []}}"
        if phase in (self.PH_REL, self.PH_REL_FINAL):
            with self._lock:
                self.prompts["judge"].append(prompt)
            src = ans(prompt.split("[SOURCE STATE]")[1].split("[TARGET STATE]")[0])
            tgt = ans(prompt.split("[TARGET STATE]")[1])
            r = (src, tgt) in self.REFINES
            return f'{{"target_issue": "t", "witness": "w", "refines": {str(r).lower()}}}'
        if phase == self.PH_REV:
            with self._lock:
                self.prompts["revise"].append(prompt)
            up = prompt.split("=== UPSTREAM REFINEMENT STATES ===")[1]
            new = ans(up.split("[Upstream state 1]")[1])
            return f"Revised after checking.\n{{\"answer\": \"{new}\", \"established\": [\"y\"], \"open_obligations\": []}}"
        if phase == self.PH_READ:
            with self._lock:
                self.prompts["readout"].append(prompt)
            first = ans(prompt.split("[Candidate state 1]")[1])
            return f'Compared.\n{{"answer": "{first}"}}'
        raise AssertionError(f"unexpected phase {phase}")

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.lenses = list(LoRIT.DOMAIN_ROLES)
        LENS_ANSWERS.clear()

    def test_chain_readout_picks_source_without_llm_or_votes(self):
        LENS_ANSWERS.update({self.lenses[0]: "7", self.lenses[1]: "5", self.lenses[2]: "3"})
        FakeLoRIT.REFINES = {("7", "5"), ("5", "3")}
        m = FakeLoRIT(num_agents=3, max_rounds=0)
        out = m.inference({"query": "q", "source": "GSM8K"})
        t = out["lorit"]
        self.assertEqual(out["response"], "7")
        self.assertEqual(t["final_D_edges"], [[0, 1], [1, 2]])
        self.assertEqual(t["readout_candidates"], [0])
        self.assertEqual(t["readout"]["mode"], "single_candidate")
        self.assertEqual(m.prompts["readout"], [])
        self.assertEqual(len(m.prompts["judge"]), 6)

    def test_minority_source_beats_majority(self):
        LENS_ANSWERS.update({self.lenses[0]: "7", **{l: "5" for l in self.lenses[1:5]}})
        FakeLoRIT.REFINES = {("7", "5")}
        m = FakeLoRIT(num_agents=5, max_rounds=2)
        out = m.inference({"query": "q", "source": "GSM8K"})
        t = out["lorit"]
        r0 = t["rounds"][0]
        self.assertEqual(r0["D_edges"], [[0, 1], [0, 2], [0, 3], [0, 4]])
        self.assertEqual(r0["revised_agents"], [1, 2, 3, 4])
        self.assertEqual(t["rounds"][1]["stop"], "fixed_point")
        self.assertEqual(t["final_topology_source"], "round_1_unrevised")
        self.assertEqual(t["readout_candidates"], [0, 1, 2, 3, 4])
        self.assertEqual(t["readout"]["mode"], "llm_readout")
        self.assertIn("Do not count votes", m.prompts["readout"][0])
        self.assertEqual(out["response"], "7")

    def test_revision_is_synchronous(self):
        LENS_ANSWERS.update({self.lenses[0]: "9", self.lenses[1]: "8", self.lenses[2]: "6"})
        FakeLoRIT.REFINES = {("9", "8"), ("8", "6")}
        m = FakeLoRIT(num_agents=3, max_rounds=1)
        out = m.inference({"query": "q", "source": "GSM8K"})
        r0 = out["lorit"]["rounds"][0]
        self.assertEqual([s["answer"] for s in r0["states_after"]], ["9", "9", "8"])
        prompt_for_2 = next(p for p in m.prompts["revise"] if "=== CURRENT STATE (yours) ===\nANSWER: 6" in p)
        self.assertIn("ANSWER: 8", prompt_for_2.split("=== UPSTREAM REFINEMENT STATES ===")[1])
        self.assertNotIn("ANSWER: 9", prompt_for_2.split("=== UPSTREAM REFINEMENT STATES ===")[1])

    def test_final_topology_rebuilt_after_last_revision(self):
        LENS_ANSWERS.update({self.lenses[0]: "9", self.lenses[1]: "8", self.lenses[2]: "6"})
        FakeLoRIT.REFINES = {("9", "8"), ("8", "6")}
        m = FakeLoRIT(num_agents=3, max_rounds=1)
        t = m.inference({"query": "q", "source": "GSM8K"})["lorit"]
        self.assertEqual(t["final_topology_source"], "rebuilt_from_final_states")
        self.assertEqual(t["rounds"][0]["D_edges"], [[0, 1], [1, 2]])
        self.assertEqual(t["final_D_edges"], [[0, 2], [1, 2]])
        self.assertEqual(len(m.prompts["judge"]), 12)
        self.assertEqual(t["token_usage"]["by_phase"]["relation_judgment_final"]["num_llm_calls"], 6)
        self.assertEqual(t["readout_candidates"], [0, 1])

    def test_mutual_refinement_with_both_directions_true(self):
        LENS_ANSWERS.update({self.lenses[0]: "1", self.lenses[1]: "2"})
        FakeLoRIT.REFINES = {("1", "2"), ("2", "1")}
        m = FakeLoRIT(num_agents=2, max_rounds=0)
        t = m.inference({"query": "q", "source": "GSM8K"})["lorit"]
        self.assertEqual(t["final_sccs"], [[0, 1]])
        self.assertEqual(t["final_D_edges"], [[0, 1], [1, 0]])
        self.assertEqual(t["readout_candidates"], [0, 1])

    def test_cli_overrides_and_validation(self):
        self.assertEqual(FakeLoRIT().role_mode, "domain")
        self.assertEqual(FakeLoRIT(role_mode="epistemic").role_mode, "epistemic")
        m = FakeLoRIT(transitive_reduction=0, cross_component_lifting="direct", num_agents=3)
        self.assertFalse(m.transitive_reduction)
        self.assertEqual((m.lifting, m.num_agents), ("direct", 3))
        with self.assertRaises(ValueError):
            FakeLoRIT(cross_component_lifting="mesh")

class TestParsing(unittest.TestCase):
    def setUp(self):
        self.m = FakeLoRIT()

    def test_judgment_parse(self):
        self.assertEqual(self.m._parse_judgment('{"target_issue": "a", "witness": "b", "refines": true}'),
                         (True, "b", "a", True))
        self.assertEqual(self.m._parse_judgment('Sure.\n{\n "refines": false\n}')[::3], (False, True))
        self.assertEqual(self.m._parse_judgment("I think yes")[::3], (False, False))

    def test_state_parse_multiline_json_and_reasoning(self):
        raw = "Step 1: 16-3-4=9.\nStep 2: 9*2=18.\n{\n  \"answer\": \"18\",\n  \"established\": [\"9 eggs\"],\n  \"open_obligations\": []\n}"
        s = self.m._parse_state(raw, "GSM8K", "direct_solver", 0)
        self.assertEqual((s["answer"], s["norm"]), ("18", "18"))
        self.assertTrue(s["reasoning"].startswith("Step 1"))
        self.assertNotIn("{", s["reasoning"])
        self.assertNotIn("state_parse_failed", s)

    def test_state_parse_latex_escapes_in_fence(self):
        raw = ('So the answer is (D).\n\n```json\n{"answer": "(D)",\n "established": ["$ S_3 / A_3 \\cong '
               '\\mathbb{Z}_2 $"],\n "open_obligations": []}\n```')
        s = self.m._parse_state(raw, "MMLU", "direct_solver", 0)
        self.assertEqual(s["answer"], "(D)")
        self.assertEqual(s["reasoning"], "So the answer is (D).")
        self.assertIn("mathbb", s["established"][0])
        self.assertFalse(any(k.startswith("state_parse") for k in s))

    def test_state_parse_salvages_bad_bracket(self):
        raw = 'Work.\n{"answer": "70000",\n "established": ["a", "b"],\n "open_obligations": ["c"]]'
        s = self.m._parse_state(raw, "GSM8K", "direct_solver", 0)
        self.assertEqual((s["answer"], s["established"], s["open_obligations"]), ("70000", ["a", "b"], ["c"]))
        self.assertTrue(s.get("state_parse_salvaged"))

    def test_signature_depends_on_more_than_answer(self):
        a = {"norm": "18", "reasoning": "x", "established": ["e"], "open_obligations": []}
        b = dict(a, open_obligations=["check units"])
        self.assertNotEqual(self.m.state_signature(a), self.m.state_signature(b))

if __name__ == "__main__":
    unittest.main()

class TestRobustness(unittest.TestCase):

    def test_huge_number_does_not_crash_normalization(self):
        from refine.answers import is_correct, normalize_answer
        huge = "9" * 400
        self.assertEqual(normalize_answer(huge, "GSM-Hard"), huge)
        self.assertFalse(is_correct(huge, "-9867630.0", "GSM-Hard"))
        self.assertEqual(normalize_answer("1" + "0" * 400 + "/3", "GSM-Hard"), "1" + "0" * 400 + "/3")
        self.assertEqual(normalize_answer("-9867630.0", "GSM-Hard"), "-9867630")

    def test_context_overflow_clamps_max_tokens_once(self):
        import httpx
        import openai
        from unittest import mock
        from refine.backend import Backend as MAS
        seen = []

        class FakeCompletions:
            def create(self, **req):
                seen.append(req["max_tokens"])
                if req["max_tokens"] + 24823 > 32768:
                    raise openai.BadRequestError(
                        "'max_tokens' or 'max_completion_tokens' is too large: 8192. This model's maximum context "
                        "length is 32768 tokens and your request has 24823 input tokens (8192 > 32768 - 24823).",
                        response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)
                msg = mock.Mock(content="ok")
                return mock.Mock(choices=[mock.Mock(message=msg)], usage=mock.Mock(prompt_tokens=24823, completion_tokens=5))

        class FakeClient:
            def __init__(self, **kw): self.chat = mock.Mock(completions=FakeCompletions())
            def close(self): pass

        cfg = {"model_api_config": {"m": {"model_list": [{"model_name": "m", "model_url": "u", "api_key": "k"}]}},
               "model_name": "m", "model_temperature": 0.7, "model_max_tokens": 8192, "model_timeout": 1}
        with mock.patch.object(openai, "OpenAI", FakeClient):
            m = MAS(cfg)
            self.assertEqual(m.call_llm(prompt="p"), "ok")
        self.assertEqual(seen, [8192, 32768 - 24823 - 16])
        self.assertEqual(m.context_clamps, 1)

class TestCodeTasks(unittest.TestCase):

    TEST = ("def check(candidate):\n    assert candidate('') == 0\n    assert candidate('abc') == 3\n\n"
            "def test_check():\n    check(strlen)\n\ntest_check()\n")

    def setUp(self):
        self.m = FakeLoRIT()

    def test_executor_verdicts(self):
        from utils.code_exec import run_test
        self.assertTrue(run_test("def strlen(s):\n    return len(s)\n", self.TEST)[0])
        self.assertFalse(run_test("def strlen(s):\n    return 0\n", self.TEST)[0])
        self.assertFalse(run_test("def strlen(s):\n    raise RuntimeError\n", self.TEST)[0])
        ok, detail = run_test("def strlen(s):\n    while True:\n        pass\n", self.TEST, timeout=3)
        self.assertFalse(ok)
        self.assertIn("timeout", detail)

    def test_is_correct_runs_the_test(self):
        from utils.scoring import is_correct
        self.assertTrue(is_correct("```python\ndef strlen(s):\n    return len(s)\n```", self.TEST, "HumanEval"))
        self.assertFalse(is_correct("def strlen(s):\n    return 1\n", self.TEST, "HumanEval"))

    def test_state_parsing_takes_the_last_code_block(self):
        raw = ("First attempt, wrong:\n```python\ndef strlen(s):\n    return 0\n```\n"
               "On reflection:\n```python\ndef strlen(s):\n    return len(s)\n```\n"
               '{"established": ["handles empty string"], "open_obligations": ["unicode not checked"]}')
        s = self.m._parse_state(raw, "HumanEval", "Programmer", 0)
        self.assertEqual(s["answer"], "def strlen(s):\n    return len(s)")
        self.assertEqual(s["established"], ["handles empty string"])
        self.assertEqual(s["open_obligations"], ["unicode not checked"])
        self.assertTrue(s["reasoning"].startswith("First attempt"))
        self.assertNotIn("state_parse_failed", s)

    def test_state_parsing_flags_a_missing_code_block(self):
        s = self.m._parse_state("I would use len(s).", "HumanEval", "Programmer", 0)
        self.assertEqual(s["answer"], "")
        self.assertTrue(s["state_parse_failed"])

    def test_equal_programs_are_one_hypothesis_despite_comments(self):
        a = self.m._parse_state("```python\ndef f(x):\n    # fast path\n    return x + 1\n```", "HumanEval", "Programmer", 0)
        b = self.m._parse_state("```python\ndef f(x):\n    return x + 1\n```", "HumanEval", "Programmer", 1)
        c = self.m._parse_state("```python\ndef f(x):\n    return x + 2\n```", "HumanEval", "Programmer", 2)
        self.assertEqual(a["norm"], b["norm"])
        self.assertNotEqual(a["norm"], c["norm"])

    def test_prompts_ask_for_a_code_block(self):
        solve = self.m._solve_prompt("q", "HumanEval")
        self.assertIn("```python code block", solve)
        self.assertNotIn('{"answer"', solve)
        self.assertIn('{"answer"', self.m._solve_prompt("q", "GSM8K"))
        st = {"answer": "def f():\n    return 1", "reasoning": "r", "established": [], "open_obligations": []}
        self.assertIn("IMPLEMENTATION:\n```python", self.m._fmt(st))
        self.assertIn("```python code block", self.m._readout_prompt("q", "HumanEval", [st, st]))

class TestGenerationSource(unittest.TestCase):

    def test_is_correct_refuses_a_generation_source(self):
        from utils.scoring import is_correct
        with self.assertRaises(ValueError):
            is_correct("print(1)", "", "SRDD")

    def test_generation_metrics(self):
        from utils.scoring import generation_metrics, score_run
        recs = [{"source": "SRDD", "response": "```python\nimport sys\n\ndef main():\n    print('hi')\n```",
                 "token_stats": {"m": {"num_llm_calls": 76}}},
                {"source": "SRDD", "response": "```python\ndef broken(:\n```", "token_stats": {}},
                {"source": "SRDD", "response": "I would build a GUI with tkinter.", "token_stats": {}},
                {"source": "SRDD", "response": "", "token_stats": {}},
                {"source": "SRDD", "error": "boom", "token_stats": {}}]
        m = generation_metrics(recs)
        self.assertEqual((m["n"], m["n_errors"], m["n_empty_answers"]), (5, 1, 1))
        self.assertEqual(m["syntax_ok"], 1)
        self.assertEqual(m["syntax_ok_rate"], 0.25)
        self.assertNotIn("accuracy", score_run(recs))
        self.assertEqual(score_run(recs)["calls_per_item"], 15.2)

    def test_srdd_answers_are_code_blocks(self):
        m = FakeLoRIT()
        self.assertIn("```python code block", m._solve_prompt("build a thing", "SRDD"))
        s = m._parse_state("Plan.\n```python\nprint('x')\n```", "SRDD", "Programmer", 0)
        self.assertEqual(s["answer"], "print('x')")

class TestFreeformSource(unittest.TestCase):

    def setUp(self):
        self.m = FakeLoRIT()

    def test_contract_asks_for_a_plain_block(self):
        p = self.m._solve_prompt("negotiate", "MultiAgentBench-bargaining")
        self.assertIn("negotiation strategy", p)
        self.assertNotIn("```python", p)
        self.assertIn("```", p)
        self.assertIn("research proposal", self.m._solve_prompt("q", "MultiAgentBench-research"))
        self.assertIn("```python code block", self.m._solve_prompt("q", "MultiAgentBench-coding"))

    def test_parsing_takes_the_last_block(self):
        raw = "Draft:\n```\nopen high\n```\nBetter:\n```\nanchor at 40, concede to 55\n```\n{\"established\": [\"budget known\"]}"
        s = self.m._parse_state(raw, "MultiAgentBench-bargaining", "Economist", 0)
        self.assertEqual(s["answer"], "anchor at 40, concede to 55")
        self.assertEqual(s["established"], ["budget known"])

    def test_generation_metrics_skip_syntax_for_prose(self):
        from utils.scoring import generation_metrics
        recs = [{"source": "MultiAgentBench-research", "response": "```\nWe propose a study of X.\n```"},
                {"source": "MultiAgentBench-research", "response": ""}]
        m = generation_metrics(recs)
        self.assertNotIn("syntax_ok_rate", m)
        self.assertEqual((m["n_empty_answers"], m["answered_rate"]), (1, 0.5))
        code = generation_metrics([{"source": "MultiAgentBench-coding", "response": "```python\nx = 1\n```"}])
        self.assertEqual(code["syntax_ok_rate"], 1.0)

class TestBlockReadout(unittest.TestCase):

    def test_readout_extracts_the_block(self):
        LENS_ANSWERS.clear()
        lenses = list(LoRIT.DOMAIN_ROLES)

        class R(FakeLoRIT):
            def call_llm(self, prompt=None, system_prompt=None, temperature=None, max_tokens=None,
                         phase=None, **kw):
                if phase == self.PH_READ:
                    return ("### Comparison of candidates\nBoth are similar, but the second is safer.\n"
                            "```python\ndef f(x):\n    return x + 1\n```")
                if phase == self.PH_INIT:
                    lens = next(k for k, v in self.LENSES.items() if v == system_prompt)
                    return f"reasoning\n```python\ndef f(x):\n    return {LENS_ANSWERS[lens]}\n```\n" + '{"established": []}'
                if phase in (self.PH_REL, self.PH_REL_FINAL):
                    return '{"refines": false}'
                raise AssertionError(phase)

        LENS_ANSWERS.update({lenses[0]: "x + 1", lenses[1]: "x + 2"})
        m = R(num_agents=2, max_rounds=0)
        out = m.inference({"query": "add one", "source": "HumanEval"})
        self.assertEqual(out["response"], "def f(x):\n    return x + 1")
        self.assertEqual(out["lorit"]["readout"]["mode"], "llm_readout")
        self.assertNotIn("parse_error", out["lorit"]["readout"])

class TestSRDDMetrics(unittest.TestCase):

    def test_completeness(self):
        from utils.srdd_metrics import completeness
        full = "def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n"
        self.assertEqual(completeness(full)[0], True)
        for stub in ("def f():\n    pass\n", "def f():\n    ...\n",
                     "def f():\n    raise NotImplementedError\n", "def f():\n    return 1  # TODO: handle zero\n",
                     "class A:\n    def m(self):\n        pass\n"):
            self.assertFalse(completeness(stub)[0], stub)
        self.assertTrue(completeness('def f():\n    """doc"""\n    return 1\n')[0])
        self.assertFalse(completeness("def broken(:\n")[0])

    def test_executability(self):
        from utils.srdd_metrics import executability
        self.assertTrue(executability("print('hello')\n")[0])
        ok, detail = executability("while True:\n    pass\n", timeout=3)
        self.assertTrue(ok)
        self.assertIn("still running", detail)
        self.assertFalse(executability("raise ValueError('boom')\n")[0])
        self.assertFalse(executability("import nonexistent_module_xyz\n")[0])
        self.assertFalse(executability("x = input()\n")[0])

    def test_consistency_and_quality(self):
        from analyze_srdd import MINILM
        from utils.srdd_metrics import evaluate_softwares
        if not os.path.isdir(MINILM):
            self.skipTest("embedding model not available")
        recs = [{"query": "A calculator that adds two numbers and prints the result.",
                 "response": "```python\ndef add(a, b):\n    return a + b\n\nprint(add(2, 3))\n```"},
                {"query": "A calculator that adds two numbers and prints the result.",
                 "response": "```python\ndef add(a, b):\n    pass\n```"}]
        per, summ = evaluate_softwares(recs, MINILM, timeout=10)
        self.assertTrue(per[0]["complete"] and per[0]["executable"])
        self.assertFalse(per[1]["complete"])
        self.assertGreater(per[0]["consistency"], per[1]["consistency"] - 1)
        self.assertEqual(per[1]["quality"], 0.0)
        self.assertEqual(summ["completeness"], 0.5)
