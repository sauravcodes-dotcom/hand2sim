"""Run the full experiment grid and write results/ (JSONL per run, summary, report, GIFs).

    python scripts/run_experiments.py            # everything
    python scripts/run_experiments.py --only replay_poor_k5
    python scripts/run_experiments.py --gallery  # re-film selected episodes only
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio
import numpy as np

from hand2sim.harness import (FAILURE_MODES, config_hash, episodes_to_jsonl, evaluate,
                              run_episode, summarize)
from hand2sim.policies import oracle_policy, replay_policy
from hand2sim.sim import make_env

NOISE = {  # hand-tracking noise levels: per-landmark jitter (m), depth x3, dropout rate
    "clean": dict(jitter=0.002, dropout=0.0),
    "phone": dict(jitter=0.01, dropout=0.10),
    "poor": dict(jitter=0.02, dropout=0.15),
    "bad": dict(jitter=0.03, dropout=0.20),
    "very_bad": dict(jitter=0.045, dropout=0.30),
}
EVAL_SEEDS = list(range(10_000, 10_100))  # 100 held-out episodes per run, same for every run
MAX_STEPS = 100


def experiments():
    exps = {"oracle": dict(policy="oracle")}
    for lvl in NOISE:
        for k in (1, 5, 9):
            exps[f"replay_{lvl}_k{k}"] = dict(policy="replay", noise=lvl, smooth_k=k, reanchor=True)
    exps["naive_phone_k5"] = dict(policy="replay", noise="phone", smooth_k=5, reanchor=False)
    return exps


def build(cfg):
    if cfg["policy"] == "oracle":
        return oracle_policy()
    return replay_policy(NOISE[cfg["noise"]], base_seed=0, reanchor=cfg["reanchor"],
                         smooth_k=cfg["smooth_k"])


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def run(name, cfg, out):
    t0 = time.time()
    eps, _ = evaluate(build(cfg), EVAL_SEEDS, MAX_STEPS)
    h = config_hash(cfg)
    os.makedirs(f"{out}/runs", exist_ok=True)
    episodes_to_jsonl(eps, f"{out}/runs/{name}.jsonl", {"run": name, "config_hash": h})
    s = summarize(eps)
    s.update(name=name, config=cfg, config_hash=h, seconds=round(time.time() - t0, 1))
    print(f"{name:24s} {s['rate']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]  "
          f"{ {k: v for k, v in s['outcomes'].items() if v} }")
    return s


def gallery(out, exps, per_mode=1):
    """Film one episode per (run, outcome) for a few illustrative runs."""
    os.makedirs(f"{out}/gallery", exist_ok=True)
    env = make_env(MAX_STEPS, render=True)
    shown = []
    for name in ["replay_phone_k5", "replay_bad_k1", "replay_very_bad_k5", "naive_phone_k5"]:
        rows = [json.loads(l) for l in open(f"{out}/runs/{name}.jsonl")]
        for mode in FAILURE_MODES:
            picks = [r["seed"] for r in rows if r["outcome"] == mode][:per_mode]
            for sd in picks:
                ep, frames, _ = run_episode(env, build(exps[name]), sd, record=True)
                assert ep.outcome == mode, (name, sd, ep.outcome, mode)  # reproducibility check
                if not frames:
                    continue
                path = f"gallery/{name}__{mode}__seed{sd}.gif"
                imageio.mimsave(f"{out}/{path}", frames, duration=0.08, loop=0)
                shown.append((name, mode, sd, path))
    env.close()
    return shown


def report(out, summaries, shown):
    by = {s["name"]: s for s in summaries}
    L = ["# Results", "",
         f"100 held-out episodes per run (seeds {EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}), "
         f"Wilson 95% intervals. Commit `{git_rev()}`, Python {platform.python_version()}.", "",
         "## Success rate vs. hand-tracking noise and smoothing window", "",
         "| noise (jitter, dropout) | no smoothing (k=1) | k=5 | k=9 |", "|---|---|---|---|"]
    for lvl, n in NOISE.items():
        cells = []
        for k in (1, 5, 9):
            s = by.get(f"replay_{lvl}_k{k}")
            cells.append(f"{s['rate']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]" if s else "-")
        L.append(f"| {lvl} ({n['jitter'] * 100:.1f} cm, {n['dropout']:.0%}) | " + " | ".join(cells) + " |")
    L += ["", "## Baselines", "", "| run | success | 95% CI |", "|---|---|---|"]
    for name in ["oracle", "replay_phone_k5", "naive_phone_k5"]:
        s = by.get(name)
        if s:
            L.append(f"| {name} | {s['rate']:.2f} | [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}] |")
    L += ["", "## Failure modes (counts out of 100)", "",
          "| run | " + " | ".join(FAILURE_MODES) + " |", "|---" * (len(FAILURE_MODES) + 1) + "|"]
    for s in summaries:
        L.append(f"| {s['name']} | " + " | ".join(str(s["outcomes"][m]) for m in FAILURE_MODES) + " |")
    if shown:
        L += ["", "## Gallery", ""]
        for name, mode, sd, path in shown:
            L.append(f"**{name}**, {mode} (seed {sd})  \n![]({path})\n")
    open(f"{out}/REPORT.md", "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--gallery", action="store_true", help="only re-film gallery + rewrite report")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    exps = experiments()
    path = f"{a.out}/summary.json"
    summaries = json.load(open(path)) if os.path.exists(path) else []
    if not a.gallery:
        names = a.only or list(exps)
        summaries = [s for s in summaries if s["name"] not in names]
        for n in names:
            summaries.append(run(n, exps[n], a.out))
            json.dump(sorted(summaries, key=lambda s: list(exps).index(s["name"])),
                      open(path, "w"), indent=1)
    summaries = sorted(summaries, key=lambda s: list(exps).index(s["name"]))
    shown = gallery(a.out, exps) if (a.gallery or not a.only) else []
    report(a.out, summaries, shown)


if __name__ == "__main__":
    main()
