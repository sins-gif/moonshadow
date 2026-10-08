[中文](README.md) | **English**

# Moonshadow

A local-first, traceable time-layered memory system for agents. **Raw text stored cold; short cards tiered; time drives decay; recall on demand; exact retrieval from raw when precision matters.**

| | |
|---|---|
| **Repository** | <https://github.com/sins-gif/moonshadow> |
| **Version** | `v1.2.0` ([code at this version](https://github.com/sins-gif/moonshadow/tree/v1.2.0) ｜ `git tag v1.2.0`) |
| **Minimal rerun** | `python -m unittest discover -s tests` → `Ran 198 tests` + `OK` |

> **This version is the data-driven parameter-convergence release**: `f = 0.5925`, `θ = θ′ = 0.3941`, default scheme multiplicative;
> `T9` is covered on the negative side only (the gap is on record). Sealed-release reading: **all 13 tools exit `0`**.

- Current specification (**read this first**): **`docs/v1.2-spec.md`** ← definitions, contracts and **fixed parameter values**
- **Documentation index / where to start**: `docs/README.md` ← authority map, reading order, maintenance rules
- Technical report (evidence): `docs/v1.2-weights.md` ｜ Handover overview: `docs/v1.2-summary.md`
- Reserved proposals and batch-acceptance rules: `docs/v1.2-roadmap.md` ｜ Proposal evaluation: `docs/v1.2-candidates.md`
- **Historical versions (records of the thinking process; their data is not to be relied on)**: `docs/v1.1-*.md`, `docs/v1.0-moonshadow.md`
- Data contract: `schema.json` ｜ DDL: `ddl.sql`

## Core features

- **Pure Python standard library, zero third-party dependencies** (Python 3.10+) — the imports contain no numpy, and nothing beyond the database driver that ships with Python.
- **Bounded multiplicative correction** (`q = r · m`, `m ∈ [f, 1]`) avoids the **threshold-inseparability** defect of additive fusion:
  on the same case set, additive reverses `117` pairs inside the region where multiplicative is **guaranteed** not to, while multiplicative goes out of bounds `0` times;
  additive's ranking agreement **also ties at full marks** (`1.000` vs `1.000`) — **a defect that ranking metrics cannot see, exposed by threshold calibration and the drift experiment**.
- **198 unit assertions + 13 audit/verification commands**: parameters and data are strictly controlled — each case's age window and gate preconditions,
  whether negatives are really blocked by the gate, whether the holdout set has been modified, whether the assignment forms in the docs match the code — **all executable and re-runnable**.

## Project status (v1.2)

- The time floor is fixed at **`f = 0.5925`** (`= 237/400`), completing the global parameter convergence:
  `θ = θ′ = 0.3941`, default scheme `multiplicative`, and a default call to `scoring.select()` is the production configuration.
- A **frozen-split mechanism** is in place: the dev/holdout split is defined by `eval/split-manifest.json` (seed `20260601`),
  and the holdout set is **strongly verified by SHA-256** (`tools/run_split_integrity.py`); modifying a holdout file is a violation.
- **Known defects (listed honestly)**:
  - tier `T9` **has no positive case yet**; it is currently covered on the negative side (`22-t9-noise-blocked-by-tier`); the gap is recorded in `docs/backlog.md`;
  - **the three grouped `f` values are still undetermined**; this version adopts a single floor;
  - the holdout set **has already been spent once** on the `θ` family, so the next parameter decision needs a new holdout set;
  - the extractor is not wired in, and the compression ratio is `0.35x` (**inflating** tokens);
  - 10 half-lives and 2 age windows are still placeholders and have never been swept.
  See [`docs/v1.2-summary.md`](docs/v1.2-summary.md) and [`docs/v1.2-roadmap.md`](docs/v1.2-roadmap.md) §7.

## Quick start

No dependencies to install (Python 3.10+, standard library only):

```bash
python -m unittest discover -s tests   # deterministic tests: count and status are whatever this command prints (measured for this delivery: Ran 198 tests + OK)

# Audits and contracts (four of them; all should report 0 violations)
python tools/run_case_audit.py          # case age windows and T8/T9 gate preconditions
python tools/run_gate_audit.py          # whether negatives are really blocked by the production gate, and whether reason matches the actual rule
python tools/run_split_integrity.py     # whether the holdout set has been modified (SHA-256 of the frozen manifest)
python tools/audit_docs.py              # documentation guard: links, fixed values, index

# Parameters and evidence
python tools/run_rank_eval.py           # ranking-scheme comparison + θ calibration + floor sweep
python tools/run_holdout.py             # dev vs holdout: is the conclusion overfitted
python tools/run_grouped_floor.py       # grouped time floor, two-stage sweep (v1.2-A reserved proposal)
python tools/run_batch_audit.py            # batch acceptance R1/R2: by default judges the "added after freezing" batch; exit code 1 if rejected
python tools/run_drift_check.py         # reversal regions under parameter drift: additive vs bounded multiplicative
python tools/run_network_eval.py        # network assignment accuracy and confusion matrix
python tools/audit_appendix.py          # appendix numeric recheck: recompute from the inputs and compare cell by cell with the document
python tools/run_eval.py                # extraction evaluation gate: the exit code is not `0` when it fails (warns when compression ratio < 1)
python tools/resplit_cases.py           # re-split dev/holdout by seed and write the frozen manifest (run only when deciding to create a new holdout set)

python examples/demo.py                 # end-to-end demo, printing the design rationale of each step
```

**Currently adopted parameters** (the defaults of `scoring.select()` are the production configuration):

| Parameter | Value | Location |
|---|---|---|
| Fusion scheme | `multiplicative` (fixed floor) | `scoring.ADOPTED_SCHEME` |
| Time floor `f` | `0.5925` | `scoring.TIME_FLOOR` |
| Gate thresholds `θ` / `θ′` | `0.3941` / `0.3941` (the separate higher bar has been dropped) | `scoring.THETA` / `THETA_HIGH` |
| Relevance weights | `(0.60, 0.27, 0.13)` | `scoring.RELEVANCE_WEIGHTS` |

`demo.py` walks the whole chain: append raw text → compress into `T0-T9` short cards → mark same-source rewrites as superseded →
cold-store into frames → retrieve byte-exactly by `source_id` → gate and score for recall → pack under the token budget with degradation → bill access and promote/demote tiers →
fail on a lost key field.

## A note from the author

I am a senior undergraduate. This project's architecture design and code generation were completed with AI assistance: my main contributions were **proposing the core hypotheses,
setting the audit standards, finding and fixing the AI's logic defects, and performing the data-driven parameter calibration**.
The causal chain of every correction is kept in the documents — the "conclusion differences table" in `docs/v1.2-weights.md` records which earlier conclusions
were overturned and why, and `docs/v1.2-summary.md` §4 summarises the invalidated conclusions. Technical discussion is welcome.

> Every number in this README can be recomputed with a command in this repository; numbers that cannot be recomputed are **not** written here.

## Evaluation gate

`tools/run_eval.py` runs the extraction pipeline over the gold standard in `eval/gold/` and reports **two** metrics:

| Metric | Meaning | Current baseline |
|---|---|---|
| Key-field retention | whether dates/amounts/URLs/code were lost (threshold 0.98) | 1.000 |
| Compression ratio | raw tokens / card tokens; higher is better | **0.35x** |

Watching retention alone invites gaming, and watching compression alone loses fields. The measured baseline compression ratio is **0.35x** — it **inflates** tokens,
because the spec requires `T0`/`T1` to "keep the raw text", and the baseline copies the raw passage into `raw_quote`. This is the baseline cost of complying with the spec,
not a bug: it marks the recall ceiling for "lose no field", and the model extractor's job is to push the compression ratio above 1 while holding that retention. The gate warns about this explicitly.

## Ranking-scheme comparison

Whether "time should compete with relevance for weight" is settled by data, not argument. `tools/run_rank_eval.py` compares seven schemes over **26** dev cases
(historical decisions / current status / task follow-up / long-term preferences, plus negatives, adversarial cases and boundary cases);
all seven schemes call the same real scoring code:

| Scheme | Fusion arithmetic | Context source | Pairwise agreement | Failures |
|---|---|---|---|---|
| Additive (v1.0's `0.45/0.25/0.20/0.10`) | additive | none | `1.000` | 0/26 |
| **Multiplicative correction (adopted, fixed `f = 0.5925`)** | multiplicative | none | **`1.000`** | **0/26** |
| Relevance only + time as tie-break | none | — | `0.731` | 7/26 |
| Multiplicative + hard intent switch | multiplicative | oracle intent | `1.000` | 0/26 |
| Multiplicative + continuous sensitivity interpolation | multiplicative | inferred from wording | `0.962` | 1/26 |
| RRF rank fusion | RRF | inferred from wording | `0.756` | 7/26 |
| 50/50 average of the two scores | score averaging | inferred from wording | `0.962` | 1/26 |

**This table now discriminates**: three schemes tie at full marks (additive / multiplicative fixed floor / hard intent switch),
while the two schemes that **depend on a context source** fall to `0.962` — **both fail on `24-t7-borderline` and nowhere else**.
The cause is not a missed marker: the query in `24` contains "现在" (now), the sensitivity is **correctly** detected as `1.0`, so the floor is pushed down to
`TIME_FLOOR_MIN = 0.35`, time gains more say, and the newer but less relevant `T7` overturns the order.
**Discriminating power is supplied by the cases, it does not appear by itself** — the previous version said "no discriminating power" because the case set lacked boundary cases.

The same case set also runs **θ calibration** and the **floor sweep**, which are the parts that matter:

| Check | Result |
|---|---|
| Time-sensitivity detection (binary; a known-bad intermediate metric) | accuracy `0.731`, status-class recall `0.500`, precision `0.857` |
| **Floor band-hit rate** (the correct intermediate metric) | **`26/26`** — the adopted value `0.5925` falls inside every case's feasible interval |
| Adopted-scheme `θ` calibration (multiplicative, dev set) | interval feasible `(0.386948, 0.401191)`, recommended `0.3941` (= the current value) |
| Adopted-scheme `θ′` calibration (multiplicative, dev set) | recommended `0.412684`, **rejected by the holdout set**, hence `θ′ = θ = 0.3941` |
| Additive-scheme `θ` calibration | feasible on the dev set `(0.530219, 0.550015)`; **infeasible once the holdout set is merged in** (negative upper bound `0.530219` > positive lower bound `0.503608`) |
| **Global floor sweep** | all-pass window `[0.500, 0.660]`, width `0.160`, midpoint `0.5800` |

Four conclusions:

1. **The additive scheme's threshold is inseparable**: a fresh but irrelevant card collects free points from the `0.25δ` term and overtakes a genuinely relevant card.
   **Ranking agreement cannot see this defect (additive also scores full marks); only threshold calibration reveals it** —
   over the fully labelled set (dev ∪ holdout), the feasible interval for additive `θ` is empty.
2. **The fusion form discriminates again, but the direction favours a fixed floor**: `24-t7-borderline` pushes continuous interpolation and
   the 50/50 average down to `0.962`; handing "how much say time gets" to the context makes things worse.
   Additive can still only be rejected through threshold separability (a counterexample pair plus exhaustive threshold search) and the synthetic-grid experiment (`run_drift_check.py`:
   additive reverses `117` pairs inside the region where multiplicative is "guaranteed", multiplicative goes out of bounds `0` times) — its ranking agreement is full marks.
3. **"Do we need intent detection" is a false dichotomy**: a single global value `0.5925` already passes everything;
   intent switching adds nothing (dev set `0.923`, still below the pure global `1.000`),
   yet introduces a component whose status-class recall is only `0.500` — a net negative.
4. **Binary intent accuracy is a wrong intermediate metric**: the dev set reports `7` missed/false detections,
   while the band-hit rate — measured as "does the floor fall inside the case's feasible interval" — is `26/26`: **not one of them caused a ranking error**.

**The conclusion has therefore been fixed as: default scheme `multiplicative`, `f = 0.5925`, `θ = θ′ = 0.3941`, and no intent detection.**
(`θ/θ′` are no longer the placeholder values `0.35/0.55` from the additive dimension.)

## Holdout validation (where the conclusions live)

The dev set (`eval/rank`, **26** cases) is used to propose hypotheses, the holdout set (`eval/holdout`, **9** cases) to test them.
The holdout manifest is **frozen** by `eval/split-manifest.json` (seed `20260601`), recording a SHA-256 for each file;
modifying the holdout set is a violation detected by `tools/run_split_integrity.py` (the reason for the re-split: during the v1.1 era, fixing cases modified holdout files,
so the holdout set had been contaminated by the development process). Labels use `needs_fresh` (whether the answer depends on the latest state) rather than intent classes —
the latter has no objective answer for a question like "did the quote come through".

| Policy | Dev set | Holdout set | Difference |
|---|---|---|---|
| Global **`0.70` (v1.1 old default)** | `0.808` | `1.000` | `+0.192` |
| Global **`0.5925` (currently adopted)** | **`1.000`** | **`1.000`** | `+0.000` |
| Global `0.55` | `1.000` | `1.000` | `+0.000` |
| Global `0.50` | `1.000` | `0.963` | `-0.037` |
| Global `0.35` | `0.885` | `0.889` | `+0.004` |
| Intent switch, default `0.70`/status `0.35` | `0.808` | `1.000` | `+0.192` |
| Intent switch, default `0.5925`/status `0.35` | `0.923` | `1.000` | `+0.077` |
| Continuous interpolation, default `0.70` | `0.846` | `1.000` | `+0.154` |
| Continuous interpolation, default `0.5925` | `0.962` | `1.000` | `+0.038` |

Read the tool's output literally: **the best dev-set value is `0.5925`, the best holdout value is `0.70`**,
so `run_holdout.py` prints "**the two disagree → the dev-set optimum is direct evidence of overfitting**",
along with "testing '`0.5925` better than `0.70`': holdout `1.000` vs `1.000` → **does not hold**".
**That comparison does not hold on the holdout set** — there, `0.70` is no worse than `0.5925`.

Computing feasible windows per set:

| Set | Window | Width | Midpoint |
|---|---|---|---|
| Dev set (26 cases) | `[0.5000, 0.6600]` | `0.1600` | `0.5800` |
| Holdout set (9 cases) | `[0.5225, 0.9800]` | `0.4575` | `0.7512` |
| **Intersection** | **`[0.5225, 0.6600]`** | **`0.1375`** | **`0.5913`** |

**Contrary to the v1.1 record, the holdout window still does not contain the dev window** (its lower bound `0.5225` is above the dev set's `0.5000`),
but it is much wider (`0.4575` vs `0.1600`), so it can only reject extreme values, not confirm a precise one.
The dev window is pinned by the two cases added in this round: upper bound `0.6600` ← `23-t6-tight-race`, lower bound `0.5000` ← `24-t7-borderline`.

The adopted `0.5925` differs from the intersection midpoint `0.5913` by `0.00125` — **exactly half a `FLOOR_SWEEP_STEP` (`0.0025`)**.
The midpoint itself is a **sweep-step-dependent** quantity: `0.005 → 0.5925`, `0.0025 → 0.5913`, `0.00125 → 0.5919`,
`0.0005 → 0.5922`, with a true value of about `0.592 ± 0.001`. So **"exact midpoint" must not be written**; the honest statement is
"inside the intersection, with approximately equal margins to both ends (lower `0.0700`, upper `0.0675`)".
**The narrow v1.1 window of `0.026` was manufactured by one over-window card in case `03`; that case has been fixed, so the narrow-window conclusion is void.**

**Three conclusions:**

1. **The only thing independently confirmed is that "`0.70` is too high for the dev set"** — the holdout set actually scores full marks at `0.70`,
   so the test "`0.5925` better than `0.70`" **does not hold**. `0.5925` is approximately centred inside the intersection,
   not a verified true value.
2. **Intent detection need not be built** — with a better default in place, intent switching still scores only `0.923` on the dev set (below the pure global `0.5925`),
   and ties at `1.000` on the holdout set: **no gain, yet it introduces a component whose status-class recall is only `0.250`** — a net negative.
3. **Binary time-sensitivity detection is again shown to be a wrong metric** — holdout status-class recall `0.250`,
   and all three misses (`08`, `21`, `H07`) fall inside their own feasible intervals: **none of them caused a ranking error**.
   (Note: the failure of `24-t7-borderline` on the dev set is **not** a miss — its "现在" is detected correctly;
   it is the floor being pushed to `0.35` **after** detection takes effect that flips the order.)

Boundaries: the holdout set is likewise hand-written and shares the same "numeric world" as the dev set, so what it tests is **generalisation under the same design assumptions**,
not generalisation over a real query distribution. For the limits of validity see `docs/v1.2-weights.md` §7.3 (threats to validity):
`n` is small, the expected orders and the marker word list were fixed by hand, some schemes were added after seeing the data, and there are only 3 score-type negatives
(so the `θ` interval is weakly constrained).

**Part of the holdout set's one-shot budget has already been spent**: the dev-set recommendation `0.412684` for `θ′` was rejected by the holdout set,
and the adopted value became `θ′ = θ = 0.3941`. Later parameter decisions need a **new** holdout set.

## Directory

```text
moonshadow/
├── README.md            front page: features, status, known defects, quick start, author's note
├── README.en.md         English version of this file
├── LICENSE              MIT
├── .gitignore           ignores __pycache__ / .env / credentials / large files and archives / runtime artifacts
├── .gitattributes       `* -text`: disables newline conversion (otherwise the frozen holdout hashes all break after checkout)
├── schema.json          memory-card JSON Schema (enforced validation, additionalProperties=false)
├── ddl.sql              SQLite DDL: card / claim / source / card_source / cursor / entity*
├── docs/                11 files: 4 current + 1 authoritative + index + idea backlog + 4 historical (see below)
├── src/moonshadow/      17 files / 16 modules (standard library only)
├── tools/               13 executable commands (audits + evaluation + re-split + guards)
├── eval/                case sets: gold 3 + rank 26 + holdout 9 (frozen) + network 48 + frozen manifest
├── tests/               14 test files / 198 assertions
├── examples/demo.py     end-to-end demo
└── experiments/         the user's own hand computations (kept verbatim, must not be modified; see the notes in the files)
```

```
docs/                   ordered by version number; the filename prefix makes lexicographic order = version order
  README.md             documentation index: authority map, reading order, maintenance rules (**start here**)
  backlog.md            idea backlog: architecturally important ideas that should not be built now (with decidable preconditions)
  v1.2-spec.md          current specification: contracts, definitions 1–6, **table of fixed parameter values** (**the single authority**)
  v1.2-weights.md       technical report (paper style): theorems, proofs, ablations, threats to validity, appendices A.3/A.4
  v1.2-summary.md       handover overview: status, confirmed/invalidated conclusions, defects, next steps, rerun index
  v1.2-roadmap.md       reserved proposals (grouped f, adaptive h) and the §7.1 batch-acceptance rules R1/R2/R3
  v1.2-candidates.md    v1.2 proposal evaluation: item-by-item disposition, conflicts with existing evidence, re-split, v1.2-A measurements
  v1.1-spec.md          ┐
  v1.1-weights.md       │ historical versions · records of the thinking process (kept alongside the current
  v1.1-summary.md       │ ones; **their data is not to be relied on** — read them to trace the reasoning chain)
  v1.0-moonshadow.md    ┘
eval/gold/*.json        extraction gold standard (input + the fields and tiers that must survive), 3 cases
eval/rank/*.json        ranking dev cases (positives + negatives + human expected order + adversarial and boundary cases), 26 cases
eval/holdout/*.json     ranking holdout cases, 9 cases (frozen, see below)
eval/network/cases.json network labelling set, 48 cases (24 dev + 24 holdout)
eval/split-manifest.json split freeze manifest: seed + SHA-256 of every holdout file
tools/run_eval.py       extraction evaluation gate entry point, can be wired into CI directly
tools/run_rank_eval.py  ranking comparison + intent evaluation + θ calibration + floor sweep entry point
tools/run_holdout.py    holdout validation entry point (dev vs holdout + cross-validated windows)
tools/run_grouped_floor.py grouped time floor, two-stage sweep (coarse 0.0395 → fine 0.0075 inside the box; both divide the adopted value)
tools/run_batch_audit.py batch acceptance R1/R2 (by default judges the batch added after freezing; exit code 1 if rejected)
tools/run_case_audit.py audit of case age windows and T8/T9 gate preconditions
tools/run_gate_audit.py audit of the assertion property of negatives: are they really blocked by the production gate, and does reason match the actual rule
tools/run_split_integrity.py holdout freeze verification (SHA-256)
tools/resplit_cases.py  re-split dev/holdout by seed and write the frozen manifest
tools/run_drift_check.py measurement of reversal regions under parameter drift (additive vs bounded multiplicative)
tools/audit_appendix.py appendix numeric recheck: recompute from the inputs and compare cell by cell with the document
tools/audit_docs.py     documentation guard: links reachable, assignment forms consistent with the code, index complete
tools/run_network_eval.py network assignment accuracy and confusion matrix
src/moonshadow/
  clock.py              clock injection (core code must not call datetime.now())
  ids.py                content-addressed IDs
  schema.py             dependency-free JSON Schema validator
  cards.py              make_card: fills derived fields and enforces validation
  tiers.py              authoritative T0-T9 definition table (single source of half-lives)
  store.py              SQLite source of truth + raw-layer addressing + cold storage + backlinks + run log
  chunks.py             frame-based compression container (zlib by default, zstd pluggable)
  scoring.py            scoring (seven schemes) + gate + time say + rank fusion + tier lifecycle + **adopted parameters**
  pack.py               token-budget packing (quota / degradation / carry-over)
  verify.py             deterministic precision check (a lost key field is FAIL)
  compress.py           Phase 2 extraction pipeline: prompt + rule fast path + cursor idempotence + baseline extractor
  eval.py               Phase 2.5 extraction evaluation: gold-standard loading + two-metric report + gate verdict
  rank_eval.py          ranking comparison + intent evaluation + θ calibration + floor sweep + grouped sweep: agreement / recall / feasible intervals
  network_eval.py       network assignment evaluation: confusion matrix + strict/attainable dual readings
  network_rules.py      class-level vocabulary rules for the four network types (independent of tier)
  drift.py              measurement of reversal regions under parameter drift (empirical evidence that the additive scheme has no ratio bound)
tests/                  acceptance tests for Phase 1 / 2 / 2.5 / 3 (14 test files; `Ran 198 tests` + `OK`, whatever the command prints is authoritative)
experiments/            the user's own hand computations (kept verbatim, must not be modified; see the notes in the files)
examples/demo.py        end-to-end demo
```

Runtime data (gitignored): `memory.db`, `raw/`, `chunks/`, `.tmp/`.

> **On "large files"**: Git itself imposes no size limit, and `.gitignore` can only block **known types**
> (`*.zip`/`*.tar.gz`/`*.bin`/`*.model`/`*.safetensors`, etc.). A real size gate needs a
> pre-commit hook or a CI check; this repository currently has neither, so **adding binary files requires a manual check**.

## Minimal usage

```python
from datetime import datetime, timezone
from moonshadow import FixedClock, Store, make_card
from moonshadow.scoring import select
from moonshadow.pack import Item, pack

clock = FixedClock(datetime.now(timezone.utc))   # use SystemClock() in production
store = Store("./memory", clock=clock)

sid = store.append_message("session-1", "客户说预算 12 万，截止 2026-10-15。")
card = make_card(
    clock=clock, session_id="session-1", tier="T2", importance=8, network="experience",
    source_ids=[sid], summary="预算 12 万，截止 2026-10-15。",
    entities=["客户"], facts=["预算 12 万", "截止 2026-10-15"],
)
store.put_card(card)                      # idempotent: a repeated write returns inserted=False

ranked = select(store.cards(), now=clock.now(), query_cos={card["id"]: 0.8})
store.archive_day(clock.now().date().isoformat())   # byte-exact retrieval still works after archiving
assert store.get_raw(sid) == "客户说预算 12 万，截止 2026-10-15。"
```

## Current status

**Implemented and tested**: raw-layer addressing and byte-level retrieval, cold-storage framing, memory-card contract and idempotent writes,
supersession (conflicts are marked, never overwritten), bidirectional backlink lookup, scoring (seven schemes) and the gate, budget packing, deterministic precision checking,
the extraction pipeline (prompt + rule fast path + cursor idempotence + a replaceable extractor contract),
the extraction evaluation gate (gold standard + two metrics + CI entry point), the ranking comparison, intent-detection evaluation, the θ calibration tool,
access billing and tier promotion/demotion.

**Two things landed in v1.2**: ① fixed parameter values (see `docs/v1.2-spec.md` §5.7; this file does not restate them);
② the evaluation set became contractual — the split is frozen by `eval/split-manifest.json` (seed `20260601`),
the holdout set must not be modified (enforced by `run_split_integrity.py`); and a case's `exclude` entries are **assertions**,
which must really be blocked by the production gate with a `reason` matching the actual rule (enforced by `run_gate_audit.py`).

**Not implemented yet** (the contracts are fixed; see `docs/v1.2-spec.md` §12): a callable LLM extractor
(the current compression ratio is `0.35x`, inflating tokens), an intent classifier (`time_sensitivity` relies only on a hand-written word list,
and the experiments have exposed its ceiling), FTS5/vector retrieval wiring, entity resolution, a conflict detector, an MCP server,
daily checks and retention cleanup, and encryption.

**Latest batch-acceptance readings**: the new batch (`22`–`25`) **passes R1/R2 for the first time** (`3/4`, `0.750`),
but **the older `12`–`21` batch still fails on the dev side** (R1 `1/7`, R2 `0`) —
R1/R2 are **prospective** rules, and re-splitting invalidates older batches (`docs/v1.2-roadmap.md` §7.1).
Other open issues are in `docs/v1.2-summary.md` §6 and in "Project status" above.

## Three non-negotiable constraints

1. **Traceable**: any summary can be retrieved back to its byte-exact raw text by `source_id`, verified with sha256, and this still holds after cold storage.
2. **No silent information loss**: `T0`/`T1` stay resident regardless of budget; conflicts are only marked as superseded, never overwritten; a lost key field is FAIL.
3. **Reproducible**: time is always injected, so identical input yields identical output — otherwise production incidents cannot be reconstructed and acceptance metrics cannot be regressed.
