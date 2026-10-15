# TabPFN Options Lab — submission copy

## Eligibility and publication status

Prepared **15 October 2026**. The [official TabPFN-3.5 hackathon page and terms](https://platform.priorlabs.ai/hackathon-3.5) state a deadline of **6 October 2026, 23:59 CEST** and expressly exclude late submissions (§§3.6–3.7). This project is late under those published terms; eligibility is **unconfirmed** unless the organizer explicitly authorizes an extension or exception. A separate registration/acceptance communication does not by itself amend the project-submission deadline.

This file is project-description copy, not evidence that a submission was sent, accepted, judged, or awarded. No institutional endorsement is claimed. The canonical repository destination is [ArielleTolome/tabpfn-options-lab](https://github.com/ArielleTolome/tabpfn-options-lab); publication and event submission are separate actions.

## Title

**TabPFN Options Lab: Knowable Outcomes, Not Leaked Labels**

## Short description

A local TabPFN-3.5 options-research harness that asks what each model could have known at decision time. It compares real V3.5 inference, purged chronological XGBoost, and an empirical baseline on the same candidates. A complete synthetic dataset, transparent JSON/HTML reports, and a candid research preprint make the experiment reproducible without redistributing restricted market data or claiming a trading edge.

## Full description — ready to paste

Options trades resolve after they open. Training on every earlier entry can therefore leak outcomes that were still unknown at the next decision. TabPFN Options Lab makes that timing problem visible and executable.

The project explicitly uses **TabPFN-3.5**, pinned through `tabpfn==9.2.0` and `ModelVersion.V3_5`, for binary in-context classification over six pre-entry features: IV rank, implied-minus-historical volatility, VIX/VIX3M, RSI, credit/width, and short-put delta. It compares real model predictions with an empirical-frequency baseline and chronologically tuned XGBoost. Every fitting row must have resolved strictly before the candidate decision. XGBoost's internal validation is chronological and purges overlapping labels. All three methods forecast the identical cohort rather than reporting scores only for accepted trades.

The included, deterministic fixture has **560 synthetic seed rows and 12 candidates**. Features and Bernoulli labels are generated examples—not option quotes, historical trades, price paths, or reconstructed QuantConnect data. The same validated JSON interface accepts authorized user-supplied data. Local inference writes a standalone HTML report and a machine-readable JSON report with common-cohort Brier scores, candidate probabilities, illustrative fixed contract tiers, cutoff timestamps, fitting-context sizes, and model/tuning provenance. The tiers are not Kelly sizing, and the software neither sends orders nor calculates trading profit.

A public-edition preprint supplies the options-research motivation and preserves unfavorable findings. Its original ML specification is **TabPFN-3**, not V3.5, and its historical model comparison remains incomplete because authorized data export is unavailable. The new V3.5 harness does not overwrite that history, fabricate backtests, or claim calibration or financial superiority from synthetic scores. The contribution is a clear, runnable research interface for separating predictive experiments from economic and evidentiary claims.

Public source code is Apache-2.0 with the original derived helper's MIT notice retained. Model weights are obtained separately under Prior Labs' terms; the repository does not redistribute them, accept the model license for the user, or provide a hosted inference service. The paper and media retain their existing rights. No restricted market dataset, raw result dump, private log, correspondence, or credential is included.

## Repository, paper, and reproduction

- **Repository:** https://github.com/ArielleTolome/tabpfn-options-lab
- **Setup, feature units, schema, and interpretation:** [README.md](README.md)
- **Complete synthetic input:** [examples/synthetic.json](examples/synthetic.json)
- **Recorded synthetic inference:** [HTML report](examples/index.html) and [JSON](examples/report.json); keep both files together for the HTML's JSON link
- **Paper:** [PDF](research/RESEARCH_PAPER.pdf) and [Markdown](research/RESEARCH_PAPER.md), preprint, not peer-reviewed
- **Research boundary:** [research/PROVENANCE.md](research/PROVENANCE.md)
- **Source license:** [Apache-2.0](LICENSE), with [original MIT notice](research/ORIGINAL_CODE_LICENSE.txt)
- **Model terms:** [TabPFN-3.5 license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE)
- **Narrated demo:** [2:13 MP4 release asset](https://github.com/ArielleTolome/tabpfn-options-lab/releases/download/v0.1.0/tabpfn-options-lab.mp4), [contact sheet](video/contact-sheet.jpg), and [source/media](video/)

After installing dependencies and obtaining the authorized checkpoint as described in the README:

```bash
python demo.py --device cuda --output demo-output
# CPU alternative; use a different output directory for another run:
python demo.py --input examples/synthetic.json --device cpu --output demo-output-cpu
```

Read the generated `index.html` and `report.json` inside the selected output directory. The checked-in example records a completed genuine CUDA run on 560 synthetic seed rows and 12 candidates. It is a software demonstration, not a completed historical experiment or a reliable model-superiority result.

## Fit to the published evaluation criteria

The published rubric weights **showcasing TabPFN-3.5 at 50%**, **originality at 30%**, and **reproducibility at 20%**. This table describes the project's design, not a self-awarded judging score.

| Criterion | Concrete project element |
|---|---|
| TabPFN-3.5 showcase — 50% | Genuine `ModelVersion.V3_5` inference; loaded-checkpoint identity; in-context probabilities beside two transparent comparators; no substituted scores |
| Originality — 30% | Delayed-outcome options research framed around information availability and honest evidence boundaries; connects probability scoring to a candid negative-results study without inventing an investment claim |
| Reproducibility — 20% | Included deterministic input, pinned direct dependencies, exact commands, causal/purge provenance, CPU option, local HTML/JSON output, and lightweight regression checks |

The terms require public Apache-2.0 source (§3.5) and included input data or a public data URL (§3.2). The source release uses Apache-2.0 and includes its synthetic input. Runnable access still requires the user to satisfy the model's separate license and obtain the checkpoint; source licensing does not override those terms. Meeting artifact requirements does not resolve the late-submission restriction.

## Optional YouTube field

**Omit the YouTube field unless an actual published video URL is available.** A rendered 2:13 narrated demo accompanies the project as a GitHub release asset with reproducible source and media. This document does not invent a YouTube URL or claim a YouTube upload. Video is optional under the event's public guidance; it does not repair unconfirmed eligibility.
