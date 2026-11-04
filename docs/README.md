# Tikitalia docs: start here

State of 04/11/2026. Deadline: 17/11/2026. Setup and data: [TEAM.md](../TEAM.md).

## What is here

`master` holds the official challenge material, `docs/` and every report. The code of `HMM_first_imp/` and `combined_algorithm/` is not pushed yet: it is being cleaned up first, because this repository is the one for the submission.

Never in git, on any branch: `data/`, `cache/`, `outputs/`, submission CSVs, virtual environments, and detector weights. The submission repository has to be public (SUBMISSION.md), and what enters the history stays there.

## Read in this order

To catch up in ten minutes:

1. [model_history.html](model_history.html): the week from the baseline to the H/A/D HMM, with negative results and ideas on hold. Open it in a browser.
2. [model_map_hmm.html](model_map_hmm.html): how the H/A/D HMM reads its inputs and decides the states, with a glossary. Open it in a browser.
3. [REPORT_COMBINED.md](../combined_algorithm/REPORT_COMBINED.md): the current pipeline, its numbers and the open questions to the organisers.

The two pages stop at the morning of 04/11. `REPORT_COMBINED.md` comes after them and uses another benchmark, so its numbers cannot be compared with theirs.

## All reports, oldest first

| report | date | what it holds |
|---|---|---|
| [REPORT_HMM_v0.md](../HMM_first_imp/REPORT_HMM_v0.md) | 30/10 | the first complete pipeline: rules, light network, Bakis HMM |
| [REPORT_HMM_v1.md](../HMM_first_imp/REPORT_HMM_v1.md) | 30/10 | v1: rule-gated team changes, turnover rules, machine-independent network |
| [REPORT_VALIDATION_v0_v1.md](../HMM_first_imp/REPORT_VALIDATION_v0_v1.md) | 30/10 | v0 and v1 on the IDSSE matches, and how IDSSE is degraded |
| [REPORT_C_v2.md](../HMM_first_imp/REPORT_C_v2.md) | 30/10 | approach C: segments first, then the team |
| [REPORT_C_v3.md](../HMM_first_imp/REPORT_C_v3.md) | 30/10 | approach C v3: dead ball from Smart Tagging, goalkeeper |
| [EVAL_TRACKING.md](../HMM_first_imp/EVAL_TRACKING.md), [notes](../HMM_first_imp/EVAL_TRACKING_notes.md) | 30/10 | the public window across versions (Italian) |
| [REPORT_B.md](../HMM_first_imp/REPORT_B.md) | 30/10 to 04/11 | the technical diary, 29 sections: approach B (1-9), cleaning and rules (10-13), learned team (14-16), H/A/D HMM (17), whistle (18), what IDSSE can say about BAR-ATH (19-21, 23), ball detector (22, 24-25, 27-28), last rules (26), tactical camera (29) |
| [REPORT_COMBINED.md](../combined_algorithm/REPORT_COMBINED.md) | 04/11 | **the current state** |

## Design documents

Written before the first pipeline, up to 28/10, in Italian. They explain the ideas and the terms; for what was actually built and measured, the reports above win.

| document | what it holds |
|---|---|
| [STARTING_POINTS.md](STARTING_POINTS.md) | first steps for someone starting from zero |
| [POSSIBLE_PATHS.md](POSSIBLE_PATHS.md) | the paths considered and why |
| [REGOLE_PSEUDO_LABEL.md](REGOLE_PSEUDO_LABEL.md) | the pseudo-labelling rules, in pseudocode |
| [NOMENCLATURA.md](NOMENCLATURA.md) | the terms: noise, strength, quality, weight |
| [ESEMPI_VALUTAZIONE_REGOLE.md](ESEMPI_VALUTAZIONE_REGOLE.md) | worked examples of those terms, with illustrative numbers |
| [FEATURES.md](FEATURES.md) | the per-frame features of `src/tikitalia/features.py` |
| [MOTORE.md](MOTORE.md) | the decision engine as first designed |
| [STATO_LAVORI.md](STATO_LAVORI.md) | who did what, up to 28/10 |
| [IMPROVEMENTS_DA_DISCUTERE.md](IMPROVEMENTS_DA_DISCUTERE.md) | proposals for a team call |

## Before choosing anything

- Choices are made on external matches, never on the provided match. The public window is checked once, as a structure check (RULES.md, mail of 03/11).
- Open questions to the organisers, and what depends on them: [REPORT_COMBINED.md, section 8](../combined_algorithm/REPORT_COMBINED.md#8-rules).
