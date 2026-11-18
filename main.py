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
  - The level adjustment is capped per hour at the largest hour-1 among the
    launches in that hour's fit: the slope is never extrapolated past the
    launches it was estimated from. (That cap is the design review's one
    change: extrapolating a negative slope to an unprecedented GTA VI hour-1
    is the biggest risk in this design.)
  - Quantiles are lognormal: q_tau = p1 * exp(mu_h + z_tau * sd_h), rounded
    half-up to 2 decimals, clamped to [1, 1e9], non-decreasing.
  - Spread choice: an IQR-based residual scale was tested on Luna's
    suggestion and REJECTED on the evidence — it improved the 4 practice
    cases slightly (0.164 vs 0.170) but lost on leave-one-out over all 37
    comparables (0.635 vs 0.576) and on the rolling subset (0.292 vs 0.276),
    so the residual SD stays.

Local validation (true practice answers recovered from SullyGnome, the exact
source the house used; hour-1 values match the case files exactly):
  mean pinball loss on the 4 practice cases 0.170, typical miss 26.5%,
  against the house baseline's 0.387 / 82.8%. The practice launches are NOT
  in the fit set when a practice case is scored (they are excluded from a
  practice case's comparables), so this is out-of-sample for the shape.

Determinism: no randomness, no wall clock, no network; launches are sorted
by name and each hour's pairs by log hour-1 before fitting, so floating
sums do not depend on input order; floats are rounded half-up to 2 decimals
before writing, so two runs write identical bytes.

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
from datetime import datetime, timedelta, timezone
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


def clamp_q(q):
    """A quantile into the valid range [1, 1e9]."""
    return min(max(q, 1.0), 1_000_000_000.0)


def parse_time(s):
    """ISO 8601 timestamp -> aware UTC datetime, or None if unparseable.

    A timestamp without a timezone is read as UTC (the data is UTC); this
    keeps naive/aware subtraction from ever raising TypeError.
    """
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hour_points(comp):
    """A comparable's release-day points as {hour_index: viewers}, sparse ok.

    Hour 0 is the day's first hour (hour 1 of the case). Non-finite values
    are dropped; a duplicated hour keeps its FIRST point (explicit rule,
    not input order). Points are SullyGnome hourly averages; a small
    category is sampled only about every 12 hours, so sparse comparables
    still contribute where they can.
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
        if t is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        delta = (t - day_start).total_seconds()
        if 0 <= delta < 24 * 3600 and delta % 3600 == 0:
            idx = int(delta // 3600)
            if idx not in by_hour:
                by_hour[idx] = v
    return by_hour


def usable_launches(comps):
    """(name, release-day points) for comparables with a usable hour 1.

    Sorted by name, so the fit does not depend on the input's order. The
    GTA VI category's own pre-release history is never a launch.
    """
    out = []
    for comp in comps:
        if not isinstance(comp, dict):
            continue
        name = comp.get("name") or ""
        if "Grand Theft Auto VI" in name:
            continue
        pts = hour_points(comp)
        if 0 in pts and pts[0] is not None and pts[0] > 0:
            out.append((name, pts))
    out.sort(key=lambda item: item[0])
    return out


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

    Returns (median_log_ratio, slope, resid_sd, mean_log_h1). Pairs must
    already be sorted by log h1, so the sums are order-independent.
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
    """The house baseline's row for hour h (persistence + widening spread),
    clamped to the valid range and non-decreasing."""
    qs = []
    prev = 0.0
    for z in Z:
        q = clamp_q(round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))))
        if q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def hour1_print(case):
    """The case's hour-1 print, or 1.0 when no usable hour-1 record exists
    (defensive: a missing record must not crash the run)."""
    known = case.get("known") or []
    for k in known:
        if isinstance(k, dict) and k.get("hour") == 1:
            v = k.get("viewers")
            if isinstance(v, (int, float)) and math.isfinite(v):
                return max(1.0, float(v))
            return 1.0
    return 1.0


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    p1 = hour1_print(case)

    comps = case.get("comparables") or []
    named = [c for c in comps
             if isinstance(c, dict) and (c.get("name") or "") in REFERENCE_NAMES]
    launches = usable_launches(named)
    if len(launches) < MIN_LAUNCHES:
        # Fallback: every comparable with a usable release day.
        launches = usable_launches(comps)
    if not launches:
        # No usable history at all: the baseline keeps the case valid.
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)}
                         for h in range(2, 25)]}

    log_p1 = math.log(p1)
    hours_out = []
    # The output is exactly hours 2..24, whatever the case's forecast list
    # holds: the contract fixes those 23 rows.
    for h in range(2, 25):
        j = h - 1  # hour index into the release day (hour 1 is index 0)
        pairs = []
        for _name, pts in launches:
            if j in pts and pts[j] is not None and pts[j] > 0 and math.isfinite(pts[j]):
                pairs.append((math.log(pts[0]),
                              math.log(pts[j] / pts[0])))
        if len(pairs) < MIN_POINTS:
            hours_out.append({"hour": h, "q": baseline_quantiles(p1, h)})
            continue
        pairs.sort(key=lambda p: p[0])
        med, slope, sd, mean_log_h1 = fit_hour(pairs)
        # The level adjustment never extrapolates past the largest hour-1
        # among the launches in THIS hour's fit.
        cap_log_h1 = max(p[0] for p in pairs)
        adj_log_p1 = min(log_p1, cap_log_h1)
        mu = med + slope * (adj_log_p1 - mean_log_h1)
        s = max(sd, SD_FLOOR)
        qs = []
        prev = 0.0
        for z in Z:
            q = clamp_q(round2(p1 * math.exp(mu + z * s)))
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
