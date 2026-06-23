# Copyright Sierra

"""Offline analyzer for tau-bench result JSON files.

Parses one or more result files produced by `run.py` and reports the main
benchmark metrics (Pass^1 / Pass^k) plus the two-agent-handoff analysis metrics
(handoff trigger rate, compression ratio, token usage, agent step counts).

This script does NOT call any LLM and has no external dependencies beyond the
Python standard library, so it is safe and cheap to run on full sweeps.

Examples:
    python analyze_results.py --results-path results/two-agent-handoff-*.json
    python analyze_results.py --results-path results/run.json --output analysis_metrics.json
"""

import os
import glob
import json
import argparse
from math import comb
from collections import defaultdict
from typing import Any, Dict, List, Optional


def is_successful(reward: float) -> bool:
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def load_results(paths: List[str]) -> List[Dict[str, Any]]:
    """Load and concatenate all result entries from the given files/globs."""
    expanded: List[str] = []
    for p in paths:
        matches = sorted(glob.glob(p))
        if matches:
            expanded.extend(matches)
        elif os.path.exists(p):
            expanded.append(p)
        else:
            print(f"[warn] no file matched: {p}")
    results: List[Dict[str, Any]] = []
    for path in expanded:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"[warn] skipping {path}: not a list of results")
            continue
        results.extend(data)
        print(f"[ok] loaded {len(data)} entries from {path}")
    return results


def compute_pass_hat_k(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass^k following the tau-bench definition (arxiv 2406.12045)."""
    trials = sorted(set(r.get("trial", 0) for r in results))
    num_trials = len(trials)

    # success count per task id
    c_per_task: Dict[int, int] = defaultdict(int)
    seen_tasks = set()
    for r in results:
        tid = r["task_id"]
        seen_tasks.add(tid)
        if is_successful(r.get("reward", 0.0)):
            c_per_task[tid] += 1
    # ensure tasks with zero successes are represented
    for tid in seen_tasks:
        c_per_task.setdefault(tid, 0)

    pass_hat_ks: Dict[int, float] = {}
    for k in range(1, num_trials + 1):
        total = 0.0
        for tid in seen_tasks:
            c = c_per_task[tid]
            # only meaningful when c >= k for at least some tasks
            total += comb(c, k) / comb(num_trials, k) if num_trials >= k else 0.0
        pass_hat_ks[k] = total / len(seen_tasks) if seen_tasks else 0.0

    return {
        "num_trials": num_trials,
        "num_unique_tasks": len(seen_tasks),
        "pass_hat_k": {str(k): v for k, v in pass_hat_ks.items()},
    }


def compute_handoff_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    infos = [r.get("info", {}) or {} for r in results]
    n = len(infos)

    error_entries = sum(1 for i in infos if "error" in i)
    handoff_triggered = sum(1 for i in infos if i.get("handoff_triggered"))
    # entries that actually produced a package (handoff path)
    pkg_infos = [i for i in infos if i.get("handoff_package") is not None]

    ratios = [i["compression_ratio"] for i in infos if i.get("compression_ratio") is not None]
    token_full = [i["token_full"] for i in infos if i.get("token_full")]
    token_count = [i["token_count"] for i in infos if i.get("token_count")]
    a_steps = [i["agent_a_steps"] for i in infos if i.get("agent_a_steps") is not None]
    b_steps = [i["agent_b_steps"] for i in infos if i.get("agent_b_steps") is not None]

    return {
        "total_entries": n,
        "error_entries": error_entries,
        "handoff_triggered": handoff_triggered,
        "handoff_triggered_rate": round(handoff_triggered / n, 4) if n else None,
        "produced_package": len(pkg_infos),
        "finished_in_agent_a": n - len(pkg_infos) - error_entries,
        "avg_compression_ratio": round(_mean(ratios), 4) if ratios else None,
        "avg_token_full": round(_mean(token_full), 1) if token_full else None,
        "avg_token_count": round(_mean(token_count), 1) if token_count else None,
        "avg_token_saving_rate": round(1 - _mean(ratios), 4) if ratios else None,
        "avg_agent_a_steps": round(_mean(a_steps), 2) if a_steps else None,
        "avg_agent_b_steps": round(_mean(b_steps), 2) if b_steps else None,
    }


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict, str)):
        return len(value) > 0
    return True


def compute_handoff_subset_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Focused analysis over the subset of entries that produced a handoff package.

    Reports success of the handoff path, structured-package field completeness
    (a precursor to slot-accuracy / constraint-recall), confirmation status
    distribution, and success rate bucketed by compression ratio.
    """
    subset = [
        r for r in results
        if (r.get("info", {}) or {}).get("handoff_package") is not None
    ]
    n = len(subset)
    if n == 0:
        return {"count": 0}

    succ = sum(1 for r in subset if is_successful(r.get("reward", 0.0)))

    # split active handoff (Agent A called handoff tool) vs budget-exhausted
    active = [r for r in subset if (r.get("info", {}) or {}).get("handoff_triggered")]
    passive = [r for r in subset if not (r.get("info", {}) or {}).get("handoff_triggered")]

    def _rate(rs: List[Dict[str, Any]]) -> Optional[float]:
        if not rs:
            return None
        return round(sum(1 for r in rs if is_successful(r.get("reward", 0.0))) / len(rs), 4)

    # package field completeness (fraction of packages with a non-empty field)
    pkg_fields = [
        "task_goal", "completed_subtasks", "remaining_subtasks",
        "tool_trace_summary", "intermediate_state", "semantic_frame",
        "kept_constraints", "execution_boundary", "refs", "recover_hint",
    ]
    field_present = {f: 0 for f in pkg_fields}
    completed_counts, remaining_counts, trace_counts = [], [], []
    state_key_counts, constraint_counts, slot_counts = [], [], []
    intent_present = 0
    confirmation_dist: Dict[str, int] = defaultdict(int)

    for r in subset:
        pkg = (r.get("info", {}) or {}).get("handoff_package", {}) or {}
        for f in pkg_fields:
            if _non_empty(pkg.get(f)):
                field_present[f] += 1
        completed_counts.append(len(pkg.get("completed_subtasks", []) or []))
        remaining_counts.append(len(pkg.get("remaining_subtasks", []) or []))
        trace_counts.append(len(pkg.get("tool_trace_summary", []) or []))
        state_key_counts.append(len((pkg.get("intermediate_state", {}) or {})))
        constraint_counts.append(len(pkg.get("kept_constraints", []) or []))
        sf = pkg.get("semantic_frame", {}) or {}
        if _non_empty(sf.get("intent")):
            intent_present += 1
        slot_counts.append(len((sf.get("slots", {}) or {})))
        eb = pkg.get("execution_boundary", {}) or {}
        confirmation_dist[str(eb.get("confirmation_status", "missing"))] += 1

    field_completeness = {
        f: round(field_present[f] / n, 4) for f in pkg_fields
    }

    # success rate bucketed by compression ratio
    buckets = {"<=0.15": [], "0.15-0.25": [], "0.25-0.40": [], ">0.40": []}
    for r in subset:
        cr = (r.get("info", {}) or {}).get("compression_ratio")
        if cr is None:
            continue
        if cr <= 0.15:
            key = "<=0.15"
        elif cr <= 0.25:
            key = "0.15-0.25"
        elif cr <= 0.40:
            key = "0.25-0.40"
        else:
            key = ">0.40"
        buckets[key].append(r)
    success_by_compression = {
        k: {"count": len(v), "success_rate": _rate(v)} for k, v in buckets.items()
    }

    return {
        "count": n,
        "success_rate": round(succ / n, 4),
        "active_handoff": {"count": len(active), "success_rate": _rate(active)},
        "passive_handoff_budget_exhausted": {
            "count": len(passive), "success_rate": _rate(passive)
        },
        "avg_completed_subtasks": round(_mean(completed_counts), 2),
        "avg_remaining_subtasks": round(_mean(remaining_counts), 2),
        "avg_tool_trace_entries": round(_mean(trace_counts), 2),
        "avg_intermediate_state_keys": round(_mean(state_key_counts), 2),
        "avg_kept_constraints": round(_mean(constraint_counts), 2),
        "avg_semantic_slots": round(_mean(slot_counts), 2),
        "semantic_intent_present_rate": round(intent_present / n, 4),
        "confirmation_status_distribution": dict(confirmation_dist),
        "field_completeness": field_completeness,
        "success_by_compression_bucket": success_by_compression,
    }


def compute_per_task(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards_by_task: Dict[int, List[float]] = defaultdict(list)
    for r in results:
        rewards_by_task[r["task_id"]].append(r.get("reward", 0.0))

    per_task = {}
    always_fail, always_pass, flaky = [], [], []
    for tid, rewards in sorted(rewards_by_task.items()):
        succ = sum(1 for x in rewards if is_successful(x))
        total = len(rewards)
        per_task[tid] = {"success": succ, "trials": total, "rate": round(succ / total, 4)}
        if succ == 0:
            always_fail.append(tid)
        elif succ == total:
            always_pass.append(tid)
        else:
            flaky.append(tid)

    return {
        "per_task": per_task,
        "always_pass_tasks": always_pass,
        "always_fail_tasks": always_fail,
        "flaky_tasks": flaky,
    }


def analyze(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    rewards = [r.get("reward", 0.0) for r in results]
    summary = {
        "total_entries": n,
        "avg_reward": round(sum(rewards) / n, 4) if n else None,
        "num_successful_entries": sum(1 for x in rewards if is_successful(x)),
    }
    return {
        "summary": summary,
        "pass_k": compute_pass_hat_k(results),
        "handoff": compute_handoff_metrics(results),
        "handoff_subset": compute_handoff_subset_metrics(results),
        "tasks": compute_per_task(results),
    }


def print_report(report: Dict[str, Any]) -> None:
    s = report["summary"]
    pk = report["pass_k"]
    h = report["handoff"]
    t = report["tasks"]

    print("\n" + "=" * 60)
    print("  TAU-BENCH RESULT ANALYSIS")
    print("=" * 60)

    print("\n[Summary]")
    print(f"  total entries        : {s['total_entries']}")
    print(f"  successful entries   : {s['num_successful_entries']}")
    print(f"  average reward (Pass^1): {s['avg_reward']}")

    print("\n[Pass^k]")
    print(f"  num trials           : {pk['num_trials']}")
    print(f"  unique tasks         : {pk['num_unique_tasks']}")
    for k, v in pk["pass_hat_k"].items():
        print(f"  Pass^{k}               : {round(v, 4)}")

    print("\n[Handoff metrics]")
    print(f"  handoff triggered    : {h['handoff_triggered']}/{h['total_entries']} "
          f"({h['handoff_triggered_rate']})")
    print(f"  produced package     : {h['produced_package']}")
    print(f"  finished in agent A  : {h['finished_in_agent_a']}")
    print(f"  error entries        : {h['error_entries']}")
    print(f"  avg compression ratio: {h['avg_compression_ratio']}")
    print(f"  avg token saving rate: {h['avg_token_saving_rate']}")
    print(f"  avg token full/count : {h['avg_token_full']} / {h['avg_token_count']}")
    print(f"  avg agent A/B steps  : {h['avg_agent_a_steps']} / {h['avg_agent_b_steps']}")

    hs = report.get("handoff_subset", {})
    if hs.get("count"):
        print("\n[Handoff-only subset]  (entries that produced a package)")
        print(f"  packages analyzed    : {hs['count']}")
        print(f"  success rate         : {hs['success_rate']}")
        print(f"  active handoff       : {hs['active_handoff']['count']} "
              f"(success {hs['active_handoff']['success_rate']})")
        print(f"  passive (budget out) : {hs['passive_handoff_budget_exhausted']['count']} "
              f"(success {hs['passive_handoff_budget_exhausted']['success_rate']})")
        print(f"  avg completed/remain : {hs['avg_completed_subtasks']} / {hs['avg_remaining_subtasks']}")
        print(f"  avg tool-trace items : {hs['avg_tool_trace_entries']}")
        print(f"  avg state keys       : {hs['avg_intermediate_state_keys']}")
        print(f"  avg kept constraints : {hs['avg_kept_constraints']}")
        print(f"  avg semantic slots   : {hs['avg_semantic_slots']}")
        print(f"  intent present rate  : {hs['semantic_intent_present_rate']}")
        print(f"  confirmation status  : {hs['confirmation_status_distribution']}")
        print("  field completeness   :")
        for f, v in hs["field_completeness"].items():
            print(f"      {f:<22}: {v}")
        print("  success by compression ratio bucket:")
        for k, v in hs["success_by_compression_bucket"].items():
            print(f"      {k:<10} n={v['count']:<4} success={v['success_rate']}")

    print("\n[Per-task]")
    print(f"  always pass          : {len(t['always_pass_tasks'])}")
    print(f"  always fail          : {len(t['always_fail_tasks'])}")
    print(f"  flaky (mixed)        : {len(t['flaky_tasks'])}")
    if t["always_fail_tasks"]:
        preview = t["always_fail_tasks"][:30]
        more = "..." if len(t["always_fail_tasks"]) > 30 else ""
        print(f"  always-fail task ids : {preview}{more}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-path",
        type=str,
        nargs="+",
        required=True,
        help="One or more result JSON files or globs (e.g. results/two-agent-handoff-*.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the analysis as JSON",
    )
    args = parser.parse_args()

    results = load_results(args.results_path)
    if not results:
        print("No results loaded. Nothing to analyze.")
        return

    report = analyze(results)
    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Analysis written to {args.output}")


if __name__ == "__main__":
    main()
