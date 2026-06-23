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
.SHELLFLAGS := -euo pipefail -c
.SILENT:

.PHONY: build build-submission check test fmt

# Pick a musl builder (prefer cargo-zigbuild, fall back to cargo with a local
# musl toolchain) and point the C/C++ compilers at musl so native crates
# (highs-sys, rustsat-cadical) cross-compile correctly.
define run_musl_build
	if command -v cargo-zigbuild >/dev/null 2>&1; then
		builder="cargo-zigbuild"
	elif [ -x "$$HOME/.cargo/bin/cargo-zigbuild" ]; then
		builder="$$HOME/.cargo/bin/cargo-zigbuild"
	else
		builder="cargo"
		echo "cargo-zigbuild not found; falling back to cargo (needs local musl)" >&2
	fi

	if [ "$$builder" = "cargo" ]; then
		"$$builder" build --release --target x86_64-unknown-linux-musl $(1)
	else
		export CC="zig cc -target x86_64-linux-musl"
		export CXX="zig c++ -target x86_64-linux-musl"
		"$$builder" zigbuild --release --target x86_64-unknown-linux-musl $(1)
	fi
endef

build:
	$(call run_musl_build,--bin klados)

build-submission:
	$(call run_musl_build,--bins)

check:
	cargo fmt --check
	cargo clippy --all-targets --workspace -- -D warnings

test:
	cargo test --workspace

fmt:
	cargo fmt
