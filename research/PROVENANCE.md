# Public paper provenance

**Public edition: 2026-10-15. Author: Ariel Tolome. Status: preprint, not peer-reviewed.**

This directory publishes a privacy-edited version of the original *Systematic Multi-Leg Options Volatility Arbitrage and Implied Volatility Surface Modeling* manuscript and its three aggregate research figures. It does not publish a trading dataset, cloud-account export, or an independently reproduced historical experiment.

- [Read the paper](RESEARCH_PAPER.md)
- [Download the paper PDF](RESEARCH_PAPER.pdf)
- [Runnable TabPFN-3.5 harness](../README.md)
- [Rights notice](NOTICE.txt)

## Two separate projects

| | Original options study | TabPFN Options Lab |
|---|---|---|
| Model specification | Explicit TabPFN-3 / `ModelVersion.V3` | Explicit TabPFN-3.5 / `ModelVersion.V3_5` |
| Input | Author-held historical research artifacts, subject to provider permissions | Included, deterministic synthetic inputs; or authorized user-supplied inputs |
| Completed evidence | Mechanical-strategy backtests, invalid-run diagnostics, original V3 synthetic API check | A runnable chronological classification harness; its outputs identify the input kind |
| Historical ML comparison | Not completed; authorized export remains a prerequisite | Not supplied, recreated, or claimed |
| Main question | Did tested mechanical changes improve after-cost simulated outcomes? | Can model forecasts be compared without admitting an outcome before it was knowable? |
| Reproduction scope | Not fully reproducible from this public release alone | Included synthetic example is independently runnable after package installation and authorized checkpoint access |

The new harness is not a relabeling of the original V3 work. Its synthetic labels are not reconstructed historical outcomes. It does not reproduce LEAN execution, calculate trading performance, or fill missing cells in the original three-model comparison.

## Editorial changes in this public edition

The scientific body, equations, aggregate result tables, citations, unfavorable findings, and failed-run disclosures are retained. This edition:

1. Adds a prominent V3-versus-V3.5 and preprint-status notice.
2. Removes personal affiliation identifiers, private correspondence details, account-operation references, and private artifact paths.
3. Describes author-held evidence without distributing the underlying market data, manifests, source snapshots, logs, ledgers, or email artifacts.
4. Adds the existing `refinement_equity_drawdown.png` figure beside the 2024–2025 exit-rule table, with its separate-reset and failed-confirmation caveats.
5. Replaces unavailable relative reproduction links with this public boundary notice; it does not imply that private cloud project links are publicly accessible.
6. Regenerates the public PDF from the public Markdown rather than distributing the original unredacted PDF.

The original manuscript, PDF, research workspace, experiments, and protocols are not modified by this release. Statements about original checks in the manuscript are part of that research record, not assertions that this release reran those checks.

## Source document fingerprints

These SHA-256 values identify the author-held inputs consulted for the public edition. The source manuscripts and full reproduction record are not copied wholesale into this repository; the original PDF hash is **not** the hash of the regenerated public PDF. Filenames below are provenance labels, not promised local files.

| Source input | SHA-256 |
|---|---|
| Original `RESEARCH_PAPER_DRAFT.md` | `026ce350bb274b84cb222b8ab74fb3ed8a28d3520bc2a02814b23eb77abc6a7c` |
| Original `RESEARCH_PAPER.pdf` | `a9a6ef3fab486976f4fd259865dd5a5926b11a972a89443b8df8561250d46205` |
| Original reproduction record | `c382861777fbfeb8c505b5420adc3d62044d78018230b09a5c85d3bc3843ffc0` |
| Original source-code MIT license | `47fbea8e37da216474639dea67862315358ddce0b59b934a87f2c6e8537e8ad7` |

The three figure files are copied without changing their content:

| Included figure | SHA-256 | Interpretation |
|---|---|---|
| [equity_drawdown.png](equity_drawdown.png) | `02ccbd919f07d39121f8f2b94050b0dab5fabc34e350e793309bb1b683105899` | Principal 2023 mechanical SPY comparisons; normalized equity and chart-close drawdown |
| [refinement_equity_drawdown.png](refinement_equity_drawdown.png) | `d63269c612eb6dc5dbbd8d302767564166397beaaa6e4699187054794597d9d3` | 2024 and 2025 exit-rule experiments; separate annual portfolios, failed confirmation |
| [credit_equity_drawdown.png](credit_equity_drawdown.png) | `7f7ed800ad2d9cc7c51ac0808298ca845af45ad9fd6109f56f0e84283037af10` | Exploratory credit-floor results on already-inspected periods; not a robust edge |

The figures are aggregate historical research exhibits, **not outputs of the public synthetic harness** and not TabPFN performance curves. Chart-close sampling may miss intraday extremes. All annual comparisons restart at $100,000; connecting them into a continuous compounded account would misrepresent the study.

## Historical run anchors

The manuscript supplies individual cloud backtest links for its tables. Those links can require permission from the project owner. No access credentials or data-export entitlement are supplied.

- **Engine:** LEAN `2.5.0.0.18179` for the Phase 1 controlled comparisons.
- **Phase 1 algorithm SHA-256:** `36fdc4dd2ae490744a5d9884fe5492e55e4a93e1ec5aa796a37482bbcab9921e`.
- **Corrected SPY 16-delta control:** project `37872579`, run [`bd4f9251218947d03db6dfdf6da3beea`](https://www.quantconnect.com/project/37872579/bd4f9251218947d03db6dfdf6da3beea); $110 after $884 modeled fees, 221 complete packages. This is not the Phase 2 common-cohort benchmark.
- **Invalid QQQ extension:** project `37872579`, run [`060aa3cb6c518eea1fc0c96700e2c0ec`](https://www.quantconnect.com/project/37872579/060aa3cb6c518eea1fc0c96700e2c0ec); diagnostic [`90b605fd9b9f7c297f7570640f6503f2`](https://www.quantconnect.com/project/37872579/90b605fd9b9f7c297f7570640f6503f2). Three unclosed packages and stale pre-adjustment identities invalidate the comparison around OCC Memo 53848. No stale fills or synthetic holding transfers repair it.
- **Failed reserved January–September 2026 control:** project `37872579`, run [`814844209a82996a1ecd43caad01ebf8`](https://www.quantconnect.com/project/37872579/814844209a82996a1ecd43caad01ebf8). The native result ends in July with a terminal error and unresolved holdings. Later January–June reruns are postinspection descriptive evidence, not untouched confirmation. Later recovery of failed-run output did not recover the original authenticated launch-source binding.
- **Original V3 observation collections:** seed [`c2660f5cf871ad5f96870f86e941742c`](https://www.quantconnect.com/project/37872579/c2660f5cf871ad5f96870f86e941742c), 479 labels; candidates [`22e484c404c6b63d1053f59e3377d2cd`](https://www.quantconnect.com/project/37872579/22e484c404c6b63d1053f59e3377d2cd), 193 labels. These are classifier-free collections, not ML performance runs.
- **On-platform ledger audit:** project `37876886`, run [`606ab0ccea222c9935315a45b9d69075`](https://www.quantconnect.com/project/37876886/606ab0ccea222c9935315a45b9d69075). Aggregate checks identify a 168-candidate comparison cohort with 501–667 strictly prior labels. The underlying rows remain excluded from this release.

## Findings that must not disappear in presentation

- None of the valid 2023 option configurations exceeded the separately simulated SPY buy-and-hold Sharpe.
- The combined IV-filter/sizing policy returned −1.334%, versus +26.298% for the SPY comparator under the stated simulation conventions.
- The selected 75%-target/21-DTE rule lost $363 in 2024 and $3,667 in 2025 and failed its frozen confirmation screen.
- The trend gate's less negative sum of separate 2023–2025 annual P&Ls is not a continuous multi-year return, proof of a causal mechanism, or consistent after-cost profitability.
- The fixed credit-floor candidate returned $8, $8, $185, and $42 across 49, 5, 9, and 2 packages. Three samples were too small; the additional $4-per-package sensitivity changed their separate-period P&L sum from +$243 to −$17.
- The original historical ML Sharpe, drawdown, package-win-rate, and Brier table remains **not measured**. A functioning model API or a synthetic score cannot fill those cells.

## Access and rights boundary

The original environment allowed Object Store persistence but refused official export under its Researcher entitlement. This release does not bypass that restriction, repackage private ledgers, or imply that an account-tier rule can be evaded through logs or another endpoint. Independent historical work needs lawful access to suitable data, causal feature construction, complete outcome/fill/fee records, and a separately frozen protocol.

The original source-code [MIT notice](ORIGINAL_CODE_LICENSE.txt) is retained for the derived filter. New public source code is under [Apache-2.0](../LICENSE). The paper and figures retain their existing rights; the code licenses do not relicense them. TabPFN weights and outputs remain governed by Prior Labs' applicable model terms. There is no institutional endorsement or claim of journal acceptance.
