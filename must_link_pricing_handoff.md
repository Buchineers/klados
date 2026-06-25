# Must-Link Pricing Handoff

Date: 2026-06-25

## Context

Hard high-`m` BP failures in `exact_pub-101-150` are dominated by branch-node pricing exits of the form `IMPROVING (uncertified)`. These exits prevent using the branch-node LP objective as a sound lower bound.

Telemetry across `pub101`, `pub104`, `pub135`, and `pub140` showed:

- `6805 / 6805` observed `IMPROVING` branch-pricer rows came after completed full scans.
- `0 / 6805` were trial-limited.
- Positive anchors in improving rows: `1,827,837`.
- Branch-blocked positives in improving rows: `1,827,781`.
- Must-link blocked positives: `1,754,205`.
- Cannot-link blocked positives: `280,761`.

Conclusion: the pricer is usually doing a full scan, but over the wrong space. It finds positive unconstrained witnesses that violate must-link decisions, then cannot certify the branch-feasible pricing problem.

## Current Implementation State

Changed files:

- `crates/klados-solve/src/solvers/bp/solver.rs`
- `crates/klados-solve/src/solvers/bp/pricer/leaf_pair_dp.rs`
- `crates/klados-solve/src/solvers/bp/pricer/scratch.rs`

### Behavior Added

1. Must-link infeasibility pruning at BP node entry.

A node is pruned before LP solve if:

- a cannot-link pair lies inside one transitive must-link class, or
- a must-link class of size at least 3 is not itself a valid AF component.

2. Must-link closure before drop-repair in the multi-tree leaf-pair pricer.

For each raw positive DP candidate:

- compute its transitive must-link closure,
- if the closure is a valid AF component, apply existing branch repair and emit it if still positive,
- otherwise fall back to the old raw-subset repair path.

Certification behavior is unchanged. Full scans with no emitted branch-feasible positive column still return `Improving`, not `Converged`.

3. Branch-pricing telemetry.

The pricer logs scanned anchors, positive anchors, emitted columns, branch-blocked positives, must-link versus cannot-link blocking, completed versus trial-limited status, and `global_max` at branch depth greater than zero.

## Validation Already Run

Commands run successfully:

```bash
cargo test -p klados-solve leaf_pair_dp::tests
just check
cargo test -p klados-solve
just build
```

Focused tests added:

- `must_link_closure_is_transitive`
- `branch_feasible_labels_prefers_valid_must_closure`
- `repair_drops_whole_must_class_after_cannot_conflict`

## Preliminary Probe After Change

Probe command shape from the workspace root:

```bash
RUST_LOG=klados::bp=debug STRIDE_TIMEOUT=40 ./klados/target/x86_64-unknown-linux-musl/release/klados solve bp < stride-downloads/43/b2/a19809f7eb952aced6a314bd1876
```

Comparison against the previous `pub101` probe:

| Metric | Before | After |
| --- | ---: | ---: |
| Branch-pricer rows | 4039 | 3073 |
| `Improving` rows | 3746 | 2815 |
| `Found` rows | 293 | 258 |
| Positive anchors | 461168 | 365718 |
| Branch-blocked positives | 460375 | 364994 |
| Must-link blocked | 417159 | 333703 |
| Cannot-link blocked | 50467 | 36883 |
| Repair failed | 234868 | 167813 |
| Repair non-profitable | 123806 | 104586 |
| Must-class infeasible prunes | 0 | 0 |

Interpretation:

- Directionally positive on `pub101`.
- The improvement came from closure emission / changed trajectory, not infeasible-class pruning.
- The remaining issue is still certification: completed full scans can still return `Improving` when no branch-feasible positive column is emitted.

## Update 2026-06-25: Closure Telemetry + Controlled A/B → Decision

### Methodology correction

`STRIDE_TIMEOUT` is a **soft** budget the exact B&P search loop does not poll, so direct
`klados solve bp` probes run unbounded and never self-terminate. Earlier probes left
straggler processes competing for CPU, so their branch-pricer **row counts** are throughput
artifacts (varied 2–8× run-to-run). Use `timeout -k 5 40 <bin> solve bp < ...` for a hard
wall bound and run arms sequentially. The `pub101` slice table in this doc and in the
research-directions doc should be read with that caveat.

### Telemetry added

`PricingStats` (`pricer/scratch.rs`) and the three branch-pricer log lines now carry:
`must_closure_attempted`, `must_closure_valid_positive`, `must_closure_valid_nonprofitable`,
`must_closure_invalid`, `must_closure_fallback`. `branch_feasible_labels` now returns a
`MustClosureOutcome` so the caller can classify each positive candidate. Behavior unchanged.

### Clean A/B (closure disabled vs enabled, hard-bounded, sequential)

- **`Converged` rows at branch depth: `0` in every run, both arms.** Closure emission
  converts no `Improving` to `Converged`.
- Branch-pricer rows roughly flat between arms (`pub104` 2132→1783, `pub135` 270→286,
  `pub140` 2205→2163); all `improving` rows are `completed`, none trial-limited.
- Of all positive constraint-blind DP candidates (after arm): **~90% close to a set that is
  not a valid AF component** (`must_closure_invalid`); of the ~7–8% that close validly, the
  majority are non-profitable; profitable closure emissions are <0.4% of attempts.

### Decision

**Closure emission is not enough — move directly to must-link-class-aware (unit) DP
certification.** `Improving` persists because the certification quantity `global_max` is the
constraint-blind max of `solve_pair`, driven by must-link-violating witnesses. Post-hoc
closure/repair of a constraint-blind max-score column cannot fix this. The DP must compute
its max over the branch-feasible space `C(B)` directly.

## Update 2026-06-25: Branch-Feasible Anchor-Level Certification (shipped)

The first sound layer of the unit-DP design is implemented, proven, and tested.

### What shipped (`leaf_pair_dp.rs`)

- `rebuild_branch_classes`: union-find over `must_link()` → transitive must-link classes, plus
  the set of cannot-link-conflicting class-root pairs.
- `anchor_feasible(la, lb)`: false iff the two leaves' must-link classes are cannot-conflicting
  (lifts cannot-link to whole classes; also covers direct cannot-link). A *sufficient*
  infeasibility test — too-permissive only loosens the bound, never unsounds it.
- `feasible_global_max`: max of `solve_pair` over scanned **feasible** anchors, logged on every
  branch-pricer row next to `global_max`.
- The `Converged` gate now reads `completed ∧ feasible_global_max ≤ 1+ε`. Since
  `feasible_global_max ≤ global_max`, this **subsumes** the old gate and only certifies more,
  never wrongly. A `debug_assert` enforces the `feasible ≤ global` invariant.

### Why this is sound (proof in the in-code comment at the gate)

`feasible_global_max` is a sound upper bound on `max_{C ∈ C(B)} score(C)`: `solve_pair`
dominates every column at its anchor (§3), and every feasible `C` has a leaf pair whose whole
classes lie in `C` (must-closure) — so that pair is feasible and was scanned. Tested by
`certification_never_over_certifies_vs_brute_force` (500 random small instances: never returns
`Converged` while a branch-feasible improving column exists; `feasible_global_max` dominates
the brute feasible max on completed scans) and `anchor_feasibility_lifts_cannot_link_to_must_classes`.

### Empirical result — sound but does not fire on the hard nodes

Hard-bounded sequential probes on `pub104`/`pub135`/`pub140`: **zero `Converged` at branch
depth**. On `improving` rows, `feasible_global_max == global_max` ~99% of the time and never
drops to ≤ 1+ε. The certification max sits at anchors that *are* feasible — the pair `(a,b)`
can co-occur — but the constraint-blind optimal column there violates must-link in its **deep
extensions**, which anchor-level feasibility cannot see.

### Remaining open work — per-anchor must-closed VALUE

The anchor-level half is done; the value half is not. The certification max must replace
`solve_pair(a,b)` with `must_solve_pair(a,b) =` max score of a must-**closed** component
anchored at `(a,b)`. That is the constrained DP whose dominance theorem (and the
contraction-soundness obligations O1–O5, especially the foreign-leaf issue) remain unproven —
see the workspace-root `multi_tree_bp_research_directions.md`, **"Design: Must-Link-Class-Aware
(Unit) DP Certification"** and **"Implementation Slice: Branch-Feasible Anchor-Level
Certification"**.

Target theorem (unchanged):

```text
For a branch state B whose surviving must-link classes are all valid AF components,
the contracted-unit leaf-pair DP dominates every branch-feasible AF component in C(B).
Therefore a completed unit scan with max score <= 1 + eps certifies branch-node convergence.
```

Do not gate `Converged` on a must-closed value bound until that dominance theorem is proven and
tested. The exact track cannot tolerate an unsound lower-bound prune. The shipped anchor-level
gate is the sound skeleton: plug `must_solve_pair` into the certification max once proven.

## Update 2026-06-25: Lagrangian Must-Link Certification (shipped, default on for m≥3)

Rather than build an exact must-closed value DP (barriered by cross-side class coupling), we
relax the must-link equalities into the pricing objective with multipliers. For any `μ`,
`max_C [score(C) + Σ μ(1_x − 1_y)]` upper-bounds the must-closed feasible max, and the term is
linear, so it is just `solve_pair` on **shifted α** — the existing DP, no new machinery. A
bounded subgradient loop (`with_lagrangian_certify(6)` in `MafPricer::new`) tries to drive the
bound `≤ 1+ε`; if it does at any `μ`, `Converged` is sound (`μ=0` = the base bound, so it never
over-certifies). The loop honours the deadline.

This is the first lever that attacks the *value*-level must-link blocker (the ~90% case), not
just anchor feasibility. Sound + tested:
`lagrangian_certifies_must_blocked_node_base_bound_misses` proves it certifies a node the base
bound misses; `certification_never_over_certifies_vs_brute_force` (Lagrangian on) proves it
never over-certifies over 500 random instances.

**Unverified (no probing):** the Lagrangian dual gap may leave hard-core nodes uncertified, and
the loop adds up to 6 DP scans per must-link `Improving` node — a runtime cost that could net
negative if it rarely certifies there. Disable with `with_lagrangian_certify(0)` for an A/B.
See workspace-root `multi_tree_bp_research_directions.md`, "Implementation Slice: Lagrangian
Must-Link Certification".
