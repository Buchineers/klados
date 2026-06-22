# klados — PACE 2026 MAF Solver
#
# Targets:
#   make build               release build of the klados binary (static musl)
#   make build-submission    release build of all per-solver binaries (static musl)
#   make check               check + clippy + fmt-check (pre-PR gate)
#   make test                workspace tests
#   make fmt                 auto-format

.ONESHELL:
SHELL := /bin/bash

.PHONY: build build-submission check test fmt

# Pick a musl builder (prefer cargo-zigbuild, fall back to cargo with a local
# musl toolchain) and point the C/C++ compilers at musl so native crates
# (highs-sys, rustsat-cadical) cross-compile correctly.
MUSL_SETUP := set -euo pipefail; \
	if command -v cargo-zigbuild >/dev/null 2>&1; then builder="cargo-zigbuild"; \
	elif [ -x "$$HOME/.cargo/bin/cargo-zigbuild" ]; then builder="$$HOME/.cargo/bin/cargo-zigbuild"; \
	else builder="cargo"; echo "cargo-zigbuild not found; falling back to cargo (needs local musl)" >&2; fi; \
	export CC="zig cc -target x86_64-linux-musl"; \
	export CXX="zig c++ -target x86_64-linux-musl"

build:
	$(MUSL_SETUP)
	if [ "$$builder" = "cargo-zigbuild" ] || [ "$$builder" = "$$HOME/.cargo/bin/cargo-zigbuild" ]; then
		"$$builder" zigbuild --release --target x86_64-unknown-linux-musl --bin klados
	else
		"$$builder" build --release --target x86_64-unknown-linux-musl --bin klados
	fi

build-submission:
	$(MUSL_SETUP)
	if [ "$$builder" = "cargo-zigbuild" ] || [ "$$builder" = "$$HOME/.cargo/bin/cargo-zigbuild" ]; then
		"$$builder" zigbuild --release --target x86_64-unknown-linux-musl --bins
	else
		"$$builder" build --release --target x86_64-unknown-linux-musl --bins
	fi

check:
	set -euo pipefail
	cargo check --all-targets --workspace
	cargo clippy --all-targets --workspace -- -D clippy::all
	cargo fmt --check

test:
	cargo test --workspace

fmt:
	cargo fmt
