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

## Next Work

1. Run broader before/after probes on `pub104`, `pub135`, and `pub140`.
2. Add direct closure telemetry: attempted, valid positive, valid non-profitable, invalid, fallback.
3. Design and prove unit-aware DP certification.

Target theorem:

```text
For a branch state B, a completed scan over must-link-class anchors dominates
every branch-feasible AF component in C(B). Therefore, if its maximum score is
<= 1 + eps, branch-node pricing has converged and the LP bound is certified.
```

Do not enable new branch-node `Converged` behavior until this dominance theorem is proven and tested. The exact track cannot tolerate an unsound lower-bound prune.
