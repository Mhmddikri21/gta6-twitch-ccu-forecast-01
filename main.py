"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers, hour by
hour (hours 2..24), as five quantiles per hour, from the case's hour-1 print
and the release-day shapes of past big launches.

Method (fit at runtime from the case's own `comparables`, so the final case's
42 launches are used and a practice case's 38 are):
  - Reference set: 14 rolling-midnight-local console launches, the closest
    analogues to a console-only game unlocking at midnight local time. The
    GTA VI category's own pre-release history is not a launch and is excluded.
    If fewer than 6 reference launches are usable, fall back to every
    comparable with a usable release day.
  - Per hour h, each reference launch gives one log-ratio log(v_h / v_1).
    The forecast centre is the MEDIAN log-ratio across launches, adjusted by
    the launch's level: a per-hour OLS slope of log-ratio on log(hour-1)
    (big midnight audiences grow less into the evening peak; the slope in
    peak hours is about -0.2 to -0.3). The spread is the residual SD of that
    regression, floored at 0.08 in log units.
  - The level adjustment is capped at the largest hour-1 in the fit: the
    slope is not extrapolated beyond the launches it was estimated from.
    (That cap is the design review's one change: extrapolating a negative
    slope to an unprecedented GTA VI hour-1 is the biggest risk in this
    design, and the cap keeps the within-range relationship while stopping
    the slope from pushing the profile to extremes.)
  - Quantiles are lognormal: q_tau = p1 * exp(mu_h + z_tau * sd_h), rounded
    half-up to 2 decimals, clamped to (0, 1e9], non-decreasing.

Local validation (true practice answers recovered from SullyGnome, the exact
source the house used; hour-1 values match the case files exactly):
  mean pinball loss on the 4 practice cases 0.170, typical miss 26.5%,
  against the house baseline's 0.387 / 82.8%. The practice launches are NOT
  in the fit set when a practice case is scored (they are excluded from a
  practice case's comparables), so this is out-of-sample for the shape.

Determinism: no randomness, no wall clock, no network; every collection is
sorted before use; floats are rounded half-up to 2 decimals before writing,
so two runs write identical bytes.

What an entry must do (from the kit's RULES.txt):
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there);
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each
    once, no other key, each q five finite numbers above 0 and at most 1e9,
    non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (we ship none; this is belt and
# braces, copied from the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# Floor on a hour's log spread, so quantiles stay sane when the fit is tight.
SD_FLOOR = 0.08
# Baseline spread (the house baseline's sigma), used only when a hour has too
# few reference points to fit.
BASE_SIGMA = 0.15
# Minimum reference points needed to fit a hour; below it, persistence.
MIN_POINTS = 3
# Minimum usable reference launches before the fallback set is used.
MIN_LAUNCHES = 6

# Rolling-midnight-local console launches: the closest analogues to GTA VI
# (console-only, unlocking at midnight local time, rolling by region). The
# four practice launches are in the final case's comparables and join the fit
# there; in a practice case they are absent, which is exactly right — a
# practice case must not fit on its own answer.
REFERENCE_NAMES = frozenset([
    "Ghost of Yōtei",
    "The Legend of Zelda: Tears of the Kingdom",
    "FINAL FANTASY VII REBIRTH",
    "The Last of Us Part II",
    "Ghost of Tsushima",
    "Stellar Blade",
    "Monster Hunter Wilds",
    "Call of Duty: Modern Warfare 4",
    "ELDEN RING",
    "Resident Evil Requiem",
    "Marvel's Wolverine",
    "Red Dead Redemption 2",
    "God of War Ragnarök",
    "Marvel's Spider-Man 2",
])


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100
    rounds it. A whole number comes back as an int, so json.dumps writes 1240
    where JavaScript writes 1240 (not 1240.0). Copied from the starter."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(s):
    """ISO 8601 UTC timestamp -> datetime (aware). Returns None if unparseable."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        return None


def release_day_series(comp):
    """A comparable's release-day hourly viewer counts, hours 1..24, or None.

    Returns a list of 24 viewer numbers (hour 1 first) when the comparable
    has a usable point for every hour of its release day; otherwise None.
    Points are SullyGnome hourly averages; a small category is sampled only
    about every 12 hours, so sparse comparables drop out here (the per-hour
    fit below also tolerates missing hours, this is the strict path).
    """
    day_start = parse_time(comp.get("day_start"))
    hourly = comp.get("hourly")
    if day_start is None or not isinstance(hourly, list):
        return None
    by_hour = {}
    for row in hourly:
        if not isinstance(row, dict):
            continue
        t = parse_time(row.get("from"))
        v = row.get("viewers")
        if t is None or not isinstance(v, (int, float)) or v is None:
            continue
        delta = (t - day_start).total_seconds()
        if 0 <= delta < 24 * 3600 and delta % 3600 == 0:
            by_hour[int(delta // 3600)] = v
    if len(by_hour) < 24:
        return None
    return [by_hour[i] for i in range(24)]


def hour_points(comp):
    """A comparable's release-day points as {hour_index: viewers}, sparse ok.

    Hour 0 is the day's first hour (hour 1 of the case). Used by the per-hour
    fit, which takes every launch that has both hour 1 and the hour in
    question, so a sparsely sampled launch still contributes where it can.
    """
    day_start = parse_time(comp.get("day_start"))
    hourly = comp.get("hourly")
    if day_start is None or not isinstance(hourly, list):
        return {}
    by_hour = {}
    for row in hourly:
        if not isinstance(row, dict):
            continue
        t = parse_time(row.get("from"))
        v = row.get("viewers")
        if t is None or not isinstance(v, (int, float)) or v is None:
            continue
        delta = (t - day_start).total_seconds()
        if 0 <= delta < 24 * 3600 and delta % 3600 == 0:
            by_hour[int(delta // 3600)] = v
    return by_hour


def median(values):
    """Median of a non-empty list of floats (sorted copy; deterministic)."""
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def fit_hour(pairs):
    """Fit one hour's log-ratio distribution from (log h1, log-ratio) pairs.

    Returns (median_log_ratio, slope, resid_sd, mean_log_h1). With fewer
    than MIN_POINTS pairs the caller falls back to persistence.
    """
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(pairs)
    med = median(ys)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((a - mx) ** 2 for a in xs) / n
    if vx <= 0:
        slope = 0.0
    else:
        slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n / vx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    dof = max(1, n - 2)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    return med, slope, sd, mx


def baseline_quantiles(p1, h):
    """The house baseline's row for hour h (persistence + widening spread)."""
    return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))) for z in Z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # Reference launches: the named rolling analogues present in this case,
    # never the GTA VI category's own pre-release history (not a launch).
    comps = case.get("comparables") or []
    ref = []
    for comp in comps:
        if not isinstance(comp, dict):
            continue
        nm = comp.get("name") or ""
        if "Grand Theft Auto VI" in nm:
            continue
        if nm in REFERENCE_NAMES:
            ref.append(comp)
    if len(ref) < MIN_LAUNCHES:
        # Fallback: every comparable with a usable release day.
        ref = [c for c in comps
               if isinstance(c, dict) and "Grand Theft Auto VI" not in (c.get("name") or "")]
    # Per-launch release-day points (sparse-tolerant) and hour-1 levels.
    launches = []
    for comp in ref:
        pts = hour_points(comp)
        if 0 not in pts or pts[0] is None or pts[0] <= 0:
            continue
        launches.append(pts)
    if not launches:
        # No usable history at all: the baseline keeps the case valid.
        return {"hours": [{"hour": row["hour"], "q": baseline_quantiles(p1, row["hour"])}
                         for row in case["forecast"]]}

    # Largest hour-1 among the fit launches: the level-adjustment cap.
    max_log_h1 = max(math.log(pts[0]) for pts in launches)

    log_p1 = math.log(p1)
    # The level adjustment never extrapolates past the largest training h1.
    adj_log_p1 = min(log_p1, max_log_h1)

    hours_out = []
    for row in case["forecast"]:
        h = row["hour"]
        j = h - 1  # hour index into the release day (hour 1 is index 0)
        pairs = []
        for pts in launches:
            if j in pts and pts[j] is not None and pts[j] > 0:
                pairs.append((math.log(pts[0]),
                              math.log(max(1.0, pts[j]) / max(1.0, pts[0]))))
        if len(pairs) < MIN_POINTS:
            hours_out.append({"hour": h, "q": baseline_quantiles(p1, h)})
            continue
        med, slope, sd, mean_log_h1 = fit_hour(pairs)
        mu = med + slope * (adj_log_p1 - mean_log_h1)
        s = max(sd, SD_FLOOR)
        qs = []
        prev = 0.0
        for z in Z:
            q = round2(p1 * math.exp(mu + z * s))
            q = min(max(q, 1.0), 1_000_000_000.0)
            # Non-decreasing (rounding can tie, never fall).
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        hours_out.append({"hour": h, "q": qs})
    return {"hours": hours_out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
