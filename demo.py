"""Offline probability evaluation, not a trading system or a market backtest."""
import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import escape
import json
from math import exp
from pathlib import Path
import random

from tabpfn_filter import (
    FEATURES, RollingTradeClassifier, _chronological_split, _features,
    _timestamp, _utc_iso, probability_quantity,
)


SEED = 20261015
MODELS = {"empirical": "Causal empirical rate", "xgboost": "Tuned XGBoost",
          "tabpfn": "TabPFN-3.5"}
COLORS = {"empirical": "#526472", "xgboost": "#854db0", "tabpfn": "#155ac6"}


def synthetic_data():
    """Seeded features and Bernoulli labels; no options quotes or price paths."""
    rng = random.Random(SEED)
    start = datetime(2020, 1, 1, 15, 0, tzinfo=timezone.utc)

    def row(day):
        x = [round(rng.uniform(5, 95), 6), round(rng.uniform(-0.015, 0.09), 8),
             round(rng.uniform(0.78, 1.16), 6), round(rng.uniform(28, 78), 6),
             round(rng.uniform(0.06, 0.25), 6), round(rng.uniform(-0.22, -0.10), 6)]
        rank, vrp, slope, rsi, credit_yield, delta = x
        logit = (1.3 + 0.7 * (rank - 50) / 50 + 3 * vrp - 1.7 * (slope - 1)
                 + 0.5 * (rsi - 50) / 50 - 3 * (credit_yield - 0.14)
                 + (delta + 0.16) + (0.45 if 35 <= rsi <= 65 else -0.35))
        y = int(rng.random() < 1 / (1 + exp(-logit)))
        entry = start + timedelta(days=day)
        return {"x": x, "y": y, "entry_time": _utc_iso(entry),
                "label_time": _utc_iso(entry + timedelta(days=rng.randint(4, 24)))}

    history = [row(day) for day in range(560)]
    candidates = [dict(id=f"synthetic-{i + 1:02d}", **row(600 + i * 7)) for i in range(12)]
    return {"schema_version": 1, "data_kind": "synthetic",
            "history": history, "candidates": candidates}


def validate_input(payload):
    """Validate and normalize the public JSON contract without loading any model."""
    expected = {"schema_version", "data_kind", "history", "candidates"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"Input object must contain exactly {sorted(expected)}")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    if payload["data_kind"] not in ("synthetic", "authorized"):
        raise ValueError("data_kind must be 'synthetic' or 'authorized'")
    seen_rows, seen_ids = set(), set()
    normalized = {"schema_version": 1, "data_kind": payload["data_kind"]}
    for group in ("history", "candidates"):
        if not isinstance(payload[group], list) or not payload[group]:
            raise ValueError(f"{group} must be a nonempty list")
        normalized[group] = []
        previous = None
        for index, row in enumerate(payload[group]):
            keys = {"x", "y", "entry_time", "label_time"}
            if group == "candidates":
                keys.add("id")
            if not isinstance(row, dict) or set(row) != keys:
                raise ValueError(f"{group}[{index}] must contain exactly {sorted(keys)}")
            if not isinstance(row["x"], list):
                raise ValueError("x must be an array of six numeric features")
            x = _features(row["x"])
            if type(row["y"]) is not int or row["y"] not in (0, 1):
                raise ValueError("y must be integer 0 or 1; censored outcomes must be omitted")
            times = {}
            for name in ("entry_time", "label_time"):
                value = row[name]
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be an ISO timestamp with a UTC offset")
                parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise ValueError(f"{name} requires an explicit UTC offset, e.g. 2024-01-02T15:00:00Z")
                times[name] = _timestamp(value)
            if times["entry_time"] >= times["label_time"]:
                raise ValueError("entry_time must strictly precede label_time")
            identity = (times["entry_time"], x)
            if identity in seen_rows:
                raise ValueError("Duplicate entry_time + features across history/candidates")
            seen_rows.add(identity)
            clean = {"x": list(x), "y": row["y"],
                     **{name: _utc_iso(value) for name, value in times.items()}}
            if group == "candidates":
                identifier = row["id"]
                if (not isinstance(identifier, str) or not identifier.strip()
                        or len(identifier) > 120 or any(ord(c) < 32 for c in identifier)
                        or identifier in seen_ids):
                    raise ValueError("Candidate id must be unique, nonempty text of at most 120 characters")
                if previous is not None and times["entry_time"] <= previous:
                    raise ValueError("Candidates must have strictly increasing entry_time")
                previous = times["entry_time"]
                seen_ids.add(identifier)
                clean["id"] = identifier
            normalized[group].append(clean)
    return normalized


def evaluate(payload, device="auto"):
    payload = validate_input(payload)
    # Preflight every cutoff before expensive inference; never shrink the cohort.
    eligibility = RollingTradeClassifier("empirical", payload["history"])
    for candidate in payload["candidates"]:
        rows = eligibility._training_rows(candidate["entry_time"])
        training, validation = _chronological_split(rows)
        if len(rows) < 500 or {row["y"] for row in rows} != {0, 1}:
            raise ValueError(f"{candidate['id']}: require 500+ strictly prior resolved labels of both classes")
        if not validation or {row["y"] for row in training} != {0, 1}:
            raise ValueError(f"{candidate['id']}: purged XGBoost training requires both classes")
        eligibility.add_trade(candidate)

    models = {name: RollingTradeClassifier(name, payload["history"], device=device) for name in MODELS}
    predictions = []
    for candidate in payload["candidates"]:
        outputs = {}
        provenance = None
        for name, model in models.items():
            p = model.predict(candidate["x"], candidate["entry_time"])
            if p is None:
                raise ValueError(f"{candidate['id']}: {name} could not predict: {model.metadata['status']}")
            metadata = dict(model.metadata)
            current = tuple(metadata[key] for key in ("as_of", "n_train", "latest_label_time",
                                                      "first_training_entry_time"))
            if provenance is not None and current != provenance:
                raise ValueError("Models did not use identical causal training provenance")
            provenance = current
            outputs[name] = {"p": p, "illustrative_contracts": probability_quantity(p), **metadata}
        predictions.append({**candidate, "models": outputs})
        # Retrospective labels are stored now, but only become eligible after label_time.
        for model in models.values():
            model.add_trade(candidate)

    summary = {name: {"label": label, "n_candidates": len(predictions),
                      "brier_score": sum((row["models"][name]["p"] - row["y"]) ** 2
                                         for row in predictions) / len(predictions)}
               for name, label in MODELS.items()}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema_version": 1, "project": "TabPFN Options Lab", "data_kind": payload["data_kind"],
        "notice": ("SYNTHETIC DATA — generated features and Bernoulli labels, not market performance."
                   if payload["data_kind"] == "synthetic" else
                   "USER-PROVIDED DATA — declared authorized by the user; rights and point-in-time provenance not independently verified."),
        "input_sha256": sha256(canonical.encode()).hexdigest(),
        "features": list(FEATURES), "device_requested": device, "n_history": len(payload["history"]),
        "protocol": {
            "min_train": 500, "max_train": 5000, "label_cutoff": "label_time < candidate.entry_time",
            "rolling_order": "Filter resolved labels, sort by entry_time then label_time, take last 5000",
            "xgboost_tuning": "Last 20% by entry time validate; purge training labels >= first validation entry; minimize Brier over depth {2,3} × learning_rate {0.03,0.1}; refit all eligible rows",
            "tiers": "p < 0.82: 0; 0.82 <= p < 0.90: 1; p >= 0.90: 2 illustrative contracts",
            "limitations": "No orders, price paths, transaction costs, P&L, Sharpe, drawdown, Kelly optimization, or claim of real-market superiority. Candidate outcomes are for retrospective scoring only until resolved.",
        },
        "models": summary, "predictions": predictions,
    }


def probability_chart(report):
    rows = report["predictions"]
    width, height, left, top, plot_width, plot_height = 960, 300, 54, 22, 790, 224
    x = lambda index: left + (plot_width * index / max(1, len(rows) - 1))
    y = lambda p: top + plot_height * (1 - p)
    marks = []
    for p in (0, 0.25, 0.5, 0.75, 1):
        marks.append(f'<line x1="{left}" y1="{y(p)}" x2="{left + plot_width}" y2="{y(p)}" stroke="#dce4ed"/>')
        marks.append(f'<text x="{left - 10}" y="{y(p) + 4}" text-anchor="end">{p:.2f}</text>')
    for p, label in ((0.82, "1 contract ≥ .82"), (0.90, "2 contracts ≥ .90")):
        marks.append(f'<line x1="{left}" y1="{y(p)}" x2="{left + plot_width}" y2="{y(p)}" stroke="#8d9ca9" stroke-dasharray="4 5"/>')
        marks.append(f'<text x="{left + plot_width + 8}" y="{y(p) + 4}" class="threshold">{escape(label)}</text>')
    for name, label in MODELS.items():
        points = " ".join(f"{x(i):.2f},{y(row['models'][name]['p']):.2f}" for i, row in enumerate(rows))
        marks.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[name]}" stroke-width="2.5"/>')
        for i, row in enumerate(rows):
            p = row["models"][name]["p"]
            title = escape(f"{row['id']}: {label} p={p:.6f}")
            marks.append(f'<circle cx="{x(i):.2f}" cy="{y(p):.2f}" r="3.5" fill="{COLORS[name]}"><title>{title}</title></circle>')
    for i in range(len(rows)):
        if len(rows) <= 24 or i % max(1, len(rows) // 12) == 0 or i == len(rows) - 1:
            marks.append(f'<text x="{x(i):.2f}" y="{top + plot_height + 22}" text-anchor="middle">{i + 1}</text>')
    marks.append(f'<text x="{left + plot_width / 2}" y="{height - 6}" text-anchor="middle">Candidate order (not elapsed time)</text>')
    watermark = "Synthetic data" if report["data_kind"] == "synthetic" else "User-provided data"
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="chart-title chart-desc">'
            '<title id="chart-title">Causal probability estimates by candidate</title>'
            '<desc id="chart-desc">Three probability series between zero and one, with illustrative tier boundaries at 0.82 and 0.90. Exact values appear in the table below.</desc>'
            f'<text x="450" y="170" text-anchor="middle" class="watermark">{watermark}</text>'
            + "".join(marks) + '</svg>')


def render_html(report):
    synthetic = report["data_kind"] == "synthetic"
    title = "Synthetic probability lab" if synthetic else "Authorized-input probability lab"
    cards = "".join(
        f'<article class="score"><h3 style="color:{COLORS[name]}">{escape(model["label"])}</h3>'
        f'<p class="score-number">{model["brier_score"]:.4f}</p><p>Brier score · {model["n_candidates"]} candidates</p></article>'
        for name, model in report["models"].items())
    table_rows = []
    for row in report["predictions"]:
        shared = row["models"]["empirical"]
        values = "".join(f'<td><strong>{row["models"][name]["p"]:.4f}</strong>'
                         f'<small>{row["models"][name]["illustrative_contracts"]} illustrative contracts</small></td>'
                         for name in MODELS)
        table_rows.append(f'<tr><th scope="row">{escape(row["id"])}</th>'
                          f'<td><time>{escape(row["entry_time"])}</time></td>{values}'
                          f'<td>{shared["n_train"]}</td><td><time>{escape(shared["latest_label_time"])}</time></td>'
                          f'<td>{row["y"]}<small>Resolved {escape(row["label_time"])}</small></td></tr>')
    legend = "".join(f'<span><i style="background:{COLORS[name]}"></i>{escape(label)}</span>' for name, label in MODELS.items())
    headers = "".join(f'<th scope="col">{escape(label)}</th>' for label in MODELS.values())
    provenance = {name: report["predictions"][0]["models"][name] for name in MODELS}
    data_note = ("Every feature and outcome in this run is generated. Binary labels are Bernoulli draws, "
                 "not exits calculated from market prices. These scores demonstrate software behavior only."
                 if synthetic else
                 "Input was declared authorized by its provider. This program cannot verify ownership, "
                 "feature availability, or the economic meaning of the supplied outcomes. These scores alone do not establish an investment edge.")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>TabPFN Options Lab — {escape(title)}</title>
<style>
:root{{--ink:#163149;--muted:#536879;--line:#dce4ed;--paper:#fff;--base:#edf2f7;--blue:#155ac6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--base);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
.notice{{background:#ffd46b;color:#382b08;padding:13px 5vw;font-weight:750;text-align:center;border-bottom:2px solid #dfb44e}}
main{{max-width:1220px;margin:auto;padding:48px 34px}}header{{display:grid;grid-template-columns:1.4fr 1fr;gap:54px;align-items:end;margin-bottom:38px}}
.brand{{font-weight:750;color:var(--blue);margin:0 0 12px}}h1,h2,h3{{font-family:'Trebuchet MS',system-ui,sans-serif;line-height:1.13;letter-spacing:-.025em}}
h1{{font-size:clamp(34px,4.6vw,62px);margin:0 0 18px;max-width:13ch}}h2{{font-size:27px;margin:0 0 14px}}h3{{font-size:18px;margin:0}}
p{{max-width:73ch}}header p{{margin-top:0}}.intro{{font-size:18px}}.protocol{{background:#dfe9f5;border-left:4px solid var(--blue);padding:22px 25px}}
.protocol strong{{display:block;font-size:18px}}.protocol p{{margin:10px 0 0}}.scores{{display:grid;grid-template-columns:repeat(3,1fr);gap:0;margin:24px 0 20px;background:white;border:1px solid var(--line);border-radius:16px;overflow:hidden}}
.score{{padding:24px 28px}}.score+.score{{border-left:1px solid var(--line)}}.score p{{margin:0;color:var(--muted)}}.score .score-number{{font-size:45px;line-height:1.4;letter-spacing:-.04em;color:var(--ink);font-variant-numeric:tabular-nums}}
section{{margin-top:36px}}.chart{{padding:24px 24px 12px;background:white;border:1px solid var(--line);border-radius:16px}}.legend{{display:flex;gap:22px;flex-wrap:wrap;font-size:14px}}
.legend span{{display:flex;align-items:center;gap:8px}}.legend i{{display:inline-block;width:20px;height:3px}}svg{{width:100%;height:auto;display:block;margin-top:12px}}svg text{{font:12px system-ui,sans-serif;fill:var(--muted)}}svg .threshold{{font-size:10px}}svg .watermark{{font:700 52px 'Trebuchet MS',sans-serif;fill:#edf2f7}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;background:white}}table{{border-collapse:collapse;width:100%;font-size:13px;text-align:left;font-variant-numeric:tabular-nums}}
caption{{text-align:left;padding:16px 18px;color:var(--muted)}}th,td{{padding:14px 16px;border-top:1px solid var(--line);vertical-align:top}}thead th{{background:#e4ecf5;white-space:nowrap}}tbody th{{font-weight:600}}td small{{display:block;font-size:11px;color:var(--muted);min-width:110px}}time{{white-space:nowrap;font-size:12px}}
.notes{{display:grid;grid-template-columns:1fr 1fr;gap:40px}}.notes p{{color:var(--muted)}}details{{background:#fff;border:1px solid var(--line);padding:16px 20px;border-radius:10px}}summary{{cursor:pointer;font-weight:650}}summary:focus-visible,a:focus-visible,.table-wrap:focus-visible{{outline:3px solid var(--blue);outline-offset:4px}}pre{{overflow:auto;max-height:470px;font-size:12px;line-height:1.5;padding-top:10px}}a{{color:var(--blue)}}footer{{margin-top:32px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}}code{{overflow-wrap:anywhere}}.subtle{{color:var(--muted);font-size:14px}}
@media(max-width:760px){{main{{padding:30px 18px}}header,.notes{{grid-template-columns:1fr;gap:22px}}h1{{max-width:none}}.scores{{grid-template-columns:1fr}}.score+.score{{border-left:0;border-top:1px solid var(--line)}}.score{{padding:18px 22px}}.score .score-number{{font-size:36px}}.chart{{padding:18px 10px 8px}}.legend{{gap:12px;font-size:12px}}}}
@media print{{body{{background:#fff}}main{{padding:20px}}.notice{{border:2px solid #382b08}}.table-wrap,pre{{overflow:visible;max-height:none}}table{{font-size:9px}}th,td{{padding:6px}}details{{break-inside:avoid}}}}
</style></head><body><div class="notice">{escape(report['notice'])}</div><main>
<header><div><p class="brand">TabPFN Options Lab</p><h1>Probability, not hindsight.</h1><p class="intro">Three real estimators. One chronological cohort. Only outcomes resolved before each decision enter the model context.</p></div>
<aside class="protocol"><strong>{escape(title)}</strong><p>{escape(data_note)}</p><p>{report['n_history']} seed rows · {len(report['predictions'])} candidates · six features<br>TabPFN architecture: explicitly V3.5</p></aside></header>
<section aria-labelledby="scores-title"><h2 id="scores-title">Probability quality on this input</h2><p>Mean squared error against the binary outcome: <strong>Brier = mean((p − y)²)</strong>. Lower is better on this cohort; a small illustrative cohort cannot establish superiority or calibration.</p><div class="scores">{cards}</div></section>
<section class="chart" aria-labelledby="chart-heading"><h2 id="chart-heading">The same decisions, three estimates</h2><div class="legend">{legend}</div>{probability_chart(report)}<p class="subtle">Dashed lines show illustrative 0 / 1 / 2-contract tiers, not trades or optimized sizing. Hover a point for its exact value.</p></section>
<section aria-labelledby="audit-title"><h2 id="audit-title">Every cutoff is inspectable</h2><p>All models use the same eligible rows. A label exactly at the cutoff is excluded. Outcomes below are retrospective scoring targets, never inputs to their own predictions.</p><div class="table-wrap" tabindex="0" role="region" aria-label="Candidate probability and causal audit table"><table><caption>UTC timestamps · training context capped at 5,000 rows · no candidates silently skipped</caption><thead><tr><th scope="col">Candidate</th><th scope="col">Decision cutoff</th>{headers}<th scope="col">Train rows</th><th scope="col">Latest admitted label</th><th scope="col">Outcome y</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></section>
<section class="notes"><div><h2>Why the chronology matters</h2><p>Filter on <code>label_time &lt; entry_time</code> first, then keep the latest 5,000 eligible entries. Resolved earlier candidates can join later contexts. XGBoost tunes only within prior data: the last 20% validate, and training labels that overlap their first entry are purged.</p></div><div><h2>What this does not measure</h2><p>No option price paths, executable quotes, fees, spread costs, portfolio returns, Sharpe ratio, or drawdown are modeled. The empirical rate is a probability baseline, not a reconstructed mechanical trading strategy. Tiers are not fractional Kelly sizing.</p></div></section>
<section><details><summary>Model identity and first-cutoff provenance</summary><p class="subtle">Loaded identity is checked at runtime. The JSON includes every cutoff, refit, tuning split, and probability at full precision.</p><pre>{escape(json.dumps(provenance, indent=2, allow_nan=False))}</pre></details></section>
<footer><p><a href="report.json">Open the machine-readable report</a> · All visuals render locally without scripts, fonts, or external services.</p><p>Normalized input SHA-256: <code>{escape(report['input_sha256'])}</code></p><p>Educational research software. No broker connection. No investment advice. The original V3 study is distinct from this V3.5 demonstration.</p></footer>
</main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Authorized or synthetic JSON in the documented schema; defaults to generated synthetic data")
    parser.add_argument("--output", type=Path, default=Path("demo-output"), help="Output directory (existing reports are never overwritten)")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto", help="TabPFN inference device; XGBoost stays on CPU")
    args = parser.parse_args()
    destinations = [args.output / "report.json", args.output / "index.html"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Output report already exists; choose a new --output directory")
    payload = json.loads(args.input.read_text(encoding="utf-8")) if args.input else synthetic_data()
    report = evaluate(payload, device=args.device)
    contents = [json.dumps(report, indent=2, allow_nan=False) + "\n", render_html(report)]
    args.output.mkdir(parents=True, exist_ok=True)
    for path, content in zip(destinations, contents):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(report["notice"])
    for model in report["models"].values():
        print(f"{model['label']}: Brier={model['brier_score']:.6f} ({model['n_candidates']} candidates)")
    print(f"Report: {destinations[1].resolve()}")


if __name__ == "__main__":
    main()
