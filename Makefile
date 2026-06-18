# klados — PACE 2026 MAF Solver
#
# Targets:
#   make build          release build
#   make check          check + clippy + fmt-check (pre-PR gate)
#   make test           workspace tests
#   make fmt            auto-format
#   make build-musl     build all per-solver musl binaries

.ONESHELL:
SHELL := /bin/bash

.PHONY: build check test fmt build-musl

build:
	cargo build --release

check:
	set -euo pipefail
	cargo check --all-targets --workspace
	cargo clippy --all-targets --workspace -- -D clippy::all
	cargo fmt --check

test:
	cargo test --workspace

fmt:
	cargo fmt

build-musl:
	set -euo pipefail
	if command -v cargo-zigbuild >/dev/null 2>&1; then
		builder="cargo-zigbuild"
	elif [ -x "$$HOME/.cargo/bin/cargo-zigbuild" ]; then
		builder="$$HOME/.cargo/bin/cargo-zigbuild"
	else
		builder="cargo"
		echo "cargo-zigbuild not found; falling back to cargo (needs local musl)" >&2
	fi
	export CC="zig cc -target x86_64-linux-musl"
	export CXX="zig c++ -target x86_64-linux-musl"
	if [ "$$builder" = "cargo-zigbuild" ] || [ "$$builder" = "$$HOME/.cargo/bin/cargo-zigbuild" ]; then
		"$$builder" zigbuild --release --target x86_64-unknown-linux-musl --bins
	else
		"$$builder" build --release --target x86_64-unknown-linux-musl --bins
	fi
