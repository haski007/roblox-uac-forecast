#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 for the 72 hours after the history ends.

Method (standard library only, deterministic):
  * Learn, from every pair of past reads of the chart, how a game's place moves with the time lag between the
    two reads: for a game at place r at the first read, where is it at the second read?  Counts are pooled by
    place bin and by lag, and smoothed along the places (wider smoothing further down the chart).
  * One forecast must serve all 72 hourly reads, so for each place we average the lag-specific distributions over
    the horizons of the window (time from the last read to each hourly slot).
  * A game's place probability is blended with a small uniform floor; every game on the last chart is named at every
    place, and what is left over is the share of games that are not on the chart yet (new entrants).
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

PLACES = 10
MAX_ROWS = 100

PARAMS = {
    "s0": 1.8,      # smoothing width in places at place 1
    "s1": 0.1,      # extra smoothing width per place
    "lag_sigma": 0.35,  # width, in log-hours, of the lag kernel
    "min_lag": 0.5,  # ignore pairs closer than this (hours)
    "max_lag": 150.0,
    "floor": 0.0,  # floor probability of any listed game at any place
    "blend": 0.0,
    "reserve_min": 0.1,  # never give out more than 1 - reserve_min of a place
    "horizon_step": 3,    # evaluate every n-th slot of the window
    "beta": 0.45,         # momentum: how far a game's recent trend shifts its starting place (log scale)
    "lookback": 24.0,    # hours back for the trend
    "new_prev": 130.0,   # place assumed for a game that was not on the chart at the lookback read
    "tau_cap": 2.0,
}

# place bins for the starting place r (inclusive upper bounds)
RBINS = [1, 2, 3, 4, 5, 6, 8, 10, 13, 18, 26, 40, 60, 10 ** 6]


def rbin(r):
    for i, ub in enumerate(RBINS):
        if r <= ub:
            return i
    return len(RBINS) - 1


def parse_time(s):
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc).timestamp() / 3600.0


def load_reads(path):
    reads = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            uac = o.get("up_and_coming")
            if not uac or len(uac) < 20 or o.get("ok") is False:
                continue
            rows = sorted(uac, key=lambda g: g["rank"])
            reads.append({
                "t": parse_time(o["at"]),
                "ids": [int(g["universe_id"]) for g in rows],
                "playing": {int(g["universe_id"]): g.get("playing") for g in rows},
                "names": {int(g["universe_id"]): g.get("name") for g in rows},
            })
    reads.sort(key=lambda r: r["t"])
    out = []
    for r in reads:  # drop reads at the very same minute
        if out and abs(out[-1]["t"] - r["t"]) < 1.0 / 120:
            out[-1] = r
        else:
            out.append(r)
    return out


def blur_matrix(p, kmax=140):
    """B[kb][k-1] for k in 1..PLACES: share of an observation at place kb that is credited to place k."""
    B = [None] * (kmax + 1)
    for kb in range(1, kmax + 1):
        sg = p["s0"] + p["s1"] * kb
        w = [math.exp(-0.5 * ((j - kb) / sg) ** 2) for j in range(1, kmax + 40)]
        z = sum(w)
        B[kb] = [w[k - 1] / z for k in range(1, PLACES + 1)]
    return B


def effective_ranks(reads, rank_of, i, p):
    """Starting place of each game of read i, shifted by its recent trend (rising games count as higher)."""
    a = reads[i]
    cur = rank_of[i]
    beta = p["beta"]
    if beta == 0.0:
        return dict(cur)
    # read closest to lookback hours earlier (within a tolerance)
    target = a["t"] - p["lookback"]
    best, bd = None, 1e9
    for j in range(i - 1, -1, -1):
        d = abs(reads[j]["t"] - target)
        if d < bd:
            best, bd = j, d
        if reads[j]["t"] < target - p["lookback"]:
            break
    out = {}
    prev = rank_of[best] if best is not None and bd <= p["lookback"] * 0.5 else None
    for u, r in cur.items():
        if prev is None:
            out[u] = float(r)
            continue
        pr = prev.get(u, p["new_prev"])
        tau = math.log(pr) - math.log(r)  # > 0: moved up
        tau = max(-p["tau_cap"], min(p["tau_cap"], tau))
        out[u] = max(1.0, r * math.exp(-beta * tau))
    return out


def build_counts(reads, p):
    """counts[lag_hour][rbin] -> list of PLACES weighted counts; tot[lag_hour][rbin]."""
    B = blur_matrix(p)
    nl = int(p["max_lag"]) + 2
    nb = len(RBINS)
    counts = [[[0.0] * PLACES for _ in range(nb)] for _ in range(nl)]
    tot = [[0.0] * nb for _ in range(nl)]
    rank_of = [{u: i + 1 for i, u in enumerate(r["ids"])} for r in reads]
    eff = [effective_ranks(reads, rank_of, i, p) for i in range(len(reads))]
    for i, a in enumerate(reads):
        for j in range(i + 1, len(reads)):
            b = reads[j]
            lag = b["t"] - a["t"]
            if lag < p["min_lag"]:
                continue
            if lag > p["max_lag"]:
                break
            li = int(round(lag))
            rb = rank_of[j]
            for u in a["ids"]:
                bi = rbin(eff[i][u])
                tot[li][bi] += 1.0
                kb = rb.get(u)
                if kb is not None:
                    row = B[min(kb, 140)]
                    c = counts[li][bi]
                    for k in range(PLACES):
                        c[k] += row[k]
    return counts, tot


def kernel_for_lag(counts, tot, lag, p):
    """Weighted [rbin][k] probabilities for a target lag (hours)."""
    nl = len(counts)
    nb = len(RBINS)
    acc = [[0.0] * PLACES for _ in range(nb)]
    wt = [0.0] * nb
    ll = math.log(max(lag, 0.5))
    for li in range(1, nl):
        w = math.exp(-0.5 * ((math.log(li) - ll) / p["lag_sigma"]) ** 2)
        if w < 1e-4:
            continue
        for bi in range(nb):
            t = tot[li][bi]
            if t <= 0:
                continue
            wt[bi] += w * t
            c = counts[li][bi]
            a = acc[bi]
            for k in range(PLACES):
                a[k] += w * c[k]
    out = []
    for bi in range(nb):
        if wt[bi] > 0:
            out.append([x / wt[bi] for x in acc[bi]])
        else:
            out.append(None)
    return out


def predict(reads, win_from, slots, p=PARAMS, slot_minutes=60):
    """Return (probs, top10): probs maps (uid, place) -> p for every game on the last chart."""
    last = reads[-1]
    counts, tot = build_counts(reads, p)
    step = max(1, int(p["horizon_step"]))
    lags = []
    for s in range(0, slots, step):
        t = win_from + s * slot_minutes / 60.0
        lags.append(max(t - last["t"], 0.5))
    ks = [kernel_for_lag(counts, tot, lg, p) for lg in lags]
    nb = len(RBINS)
    # average kernels over the window horizons
    avg = [[0.0] * PLACES for _ in range(nb)]
    for kern in ks:
        for bi in range(nb):
            row = kern[bi] if kern[bi] is not None else kern[max(0, bi - 1)] or [0.0] * PLACES
            for k in range(PLACES):
                avg[bi][k] += row[k] / len(ks)
    ids = last["ids"][:MAX_ROWS - 1]
    rank_of = [{u: i + 1 for i, u in enumerate(r["ids"])} for r in reads]
    eff_last = effective_ranks(reads, rank_of, len(reads) - 1, p)
    raw = {}
    for r, u in enumerate(ids, start=1):
        bi = rbin(eff_last[u])
        for k in range(PLACES):
            raw[(u, k + 1)] = avg[bi][k]
    probs = {}
    for k in range(1, PLACES + 1):
        col = {u: raw[(u, k)] for u in ids}
        col = {u: (1 - p["blend"]) * v + p["floor"] for u, v in col.items()}
        s = sum(col.values())
        cap = 1.0 - p["reserve_min"]
        if s > cap:
            col = {u: v * cap / s for u, v in col.items()}
        for u, v in col.items():
            probs[(u, k)] = v
    # best-guess top10 by greedy assignment on the place marginals
    used, top = set(), []
    for k in range(1, PLACES + 1):
        best = max((u for u in ids if u not in used), key=lambda u: (probs[(u, k)], -ids.index(u)))
        used.add(best)
        top.append(best)
    return probs, top


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(in_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    reads = load_reads(os.path.join(in_dir, "history.ndjson"))
    win = task["window"]
    win_from = parse_time(win["from"])
    slots = int(win.get("slots", 72))
    slot_minutes = int(win.get("slot_minutes", 60))
    probs, top = predict(reads, win_from, slots, PARAMS, slot_minutes)
    rows = [{"universe_id": u, "place": k, "p": round(v, 6)} for (u, k), v in sorted(probs.items(), key=lambda x: (x[0][1], -x[1]))]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "forecast.json"), "w", encoding="utf-8") as f:
        json.dump({"probabilities": rows, "top10": top}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
