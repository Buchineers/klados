#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    libclang-dev \
    llvm-dev \
    make \
    pipx \
    pkg-config \
    python3-pip \
    python3-venv \
    xz-utils
rm -rf /var/lib/apt/lists/*

if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/zig:$PATH"

rustup default stable
rustup target add x86_64-unknown-linux-musl

ZIG_VERSION="${ZIG_VERSION:-0.16.0}"
ZIG_DIR="/opt/zig"
if ! command -v zig >/dev/null 2>&1; then
    mkdir -p "$ZIG_DIR"
    curl -L "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
        | tar xJ -C "$ZIG_DIR" --strip-components=1
fi

if ! command -v cargo-zigbuild >/dev/null 2>&1; then
    pipx install cargo-zigbuild
fi

cd "$(dirname "${BASH_SOURCE[0]}")"
make build-submission

BUILD_DIR="target/x86_64-unknown-linux-musl/release"
rm -rf solvers
mkdir -p solvers/exact solvers/lower solvers/heuristic
mv "$BUILD_DIR/klados-bp" solvers/exact/
mv "$BUILD_DIR/klados-lower" solvers/lower/
mv "$BUILD_DIR/klados-lagrangian" solvers/heuristic/
