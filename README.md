![logo](logo.svg)

A solver for the [PACE 2026](https://pacechallenge.org/2026/) Rooted Maximum Agreement Forest (rMAF) challenge.

## What it does

Given two or more rooted phylogenetic trees on the same leaf set, klados computes the **maximum agreement forest** — a partition of the leaf set into the fewest blocks such that all input trees agree when restricted to each block. It is a generalization of the classic tree reconciliation / hybridization problem to more than two trees.

The solver includes over a dozen exact and heuristic approaches: Branch & Price (BP), ILP, SAT/MaxSAT, corridor method, Lagrangian column generation, RSPR branch-and-bound, and several greedy heuristics.

### Solver description
The full solver description for the PACE 2026 submission can be found [here](https://buchineers.github.io/klados-paper/paper.pdf).

## Build

```bash
make build              # release build of the main `klados` binary (static musl)
make build-submission   # release build of all per-solver binaries (static musl)
make check              # cargo check + clippy + fmt-check (PR gate)
make test               # run workspace tests
make fmt                # auto-format code
```

`make build` and `make build-submission` produce statically linked `x86_64-unknown-linux-musl` binaries (via `cargo-zigbuild` if available, with a fallback to plain `cargo`).

### Debian container build

`docker_setup.sh` installs all dependencies needed inside a Debian 13.5 container and builds the solver binaries:

```bash
docker run --rm -it \
  -v "$PWD:/work" \
  -w /work \
  debian:13.5 \
  ./docker_setup.sh
```

The script places the selected solvers in the following layout:

```text
solvers/<track>/<solver-name>
```

## Usage

```bash
# list available solvers
klados solve

# solve an instance from stdin with a chosen solver
klados solve bp < input.nw

# compute lower bounds
klados bounds --algo chen-pair < input.nw

# apply kernelization rules
klados kernelize < input.nw > reduced.nw

# print instance info
klados info < input.nw

# run an individual solver directly
klados-bp < input.nw
klados-sat < input.nw
klados-corridor < input.nw
...
```

All commands read a tree instance from **stdin** in the PACE 2026 `.gr` format and write results to **stdout**.

## Solvers

| Name | Description |
|------|-------------|
| `bp` | Branch & Price for multi-tree MAF (exact) |
| `bp-multi` | Branch & Price, legacy multi-tree variant (exact) |
| `ilp` | Integer Linear Programming via HiGHS |
| `sat` | SAT encoding via rustsat/cadical |
| `sat-olver` | SAT with Olver 2-approx LB seeding |
| `chen-rspr` | Chen rSPR branch-and-bound (2-tree only) |
| `whidden` | Whidden 3-way branch-and-bound (2-tree only) |
| `corridor` | Reduced-cost corridor solver (m=2 native) |
| `root-corridor` | Certified root-corridor probe with B&P fallback |
| `root-pool` | Root column generation + integer pool cover (prototype) |
| `overlay-exchange` | Incumbent-overlay replacement (prototype) |
| `lagrangian` | Dual-guided set-packing, Lagrangian column generation (anytime) |
| `agglomerative` | Agglomerative clustering heuristic |
| `greedy-partition` | Greedy partition heuristic with union-add-one refinement |
| `maxsat` | MaxSAT via open-wbo (legacy) |
| `lower` | Lower-bound track racer: fastest #a-bounded forest |

## License

GPL-3.0-or-later
