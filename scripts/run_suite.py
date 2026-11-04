"""Run the failure-mode suite and write outputs/suite_<mode>.md (+ .json)."""
import argparse
import json
import os
import time

from atl.suite import tasks

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="exact", choices=["exact", "linear"])
ap.add_argument("--tasks", default="f1,f2,f3,f4")
ap.add_argument("--out", default="outputs")
a = ap.parse_args()


def table(rows):
    if not rows:
        return ""
    keys = list(rows[0].keys())
    fmt = lambda v: f"{v:+.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


os.makedirs(a.out, exist_ok=True)
res, md = {}, [f"# Failure-mode suite (mode={a.mode})\n"]
for t in a.tasks.split(","):
    t0 = time.time()
    fn = {"f1": tasks.f1_collapse, "f2": tasks.f2_reference_bias, "f3": tasks.f3_lambda, "f4": tasks.f4_metrics}[t]
    rows = fn(mode=a.mode)
    res[t] = rows
    md += [f"## {t}: {fn.__name__}  ({time.time() - t0:.0f}s)\n", table(rows), ""]
    print("\n".join(md[-3:]), flush=True)
json.dump(res, open(f"{a.out}/suite_{a.mode}.json", "w"), indent=1)
open(f"{a.out}/suite_{a.mode}.md", "w").write("\n".join(md))
print("written", f"{a.out}/suite_{a.mode}.md")
