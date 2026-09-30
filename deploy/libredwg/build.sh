#!/usr/bin/env bash
# Build LibreDWG (GPL-3.0-or-later; runs only as a separate process, ADR-S16) from the pinned
# release tarball. Ubuntu 24.04 has no LibreDWG package. Used by deploy/Dockerfile and CI.
#   deploy/libredwg/build.sh [PREFIX]      (default /opt/libredwg)
set -euo pipefail
VERSION=0.14
SHA256=62ebb73b984f865960f20ed26619ea5f8789d5e3fd088fa40a2598384da81275
PREFIX=${1:-/opt/libredwg}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
curl -fsSLo libredwg.tar.xz "https://github.com/LibreDWG/libredwg/releases/download/${VERSION}/libredwg-${VERSION}.tar.xz"
echo "${SHA256}  libredwg.tar.xz" | sha256sum -c -
tar -xJf libredwg.tar.xz
cd "libredwg-${VERSION}"
# read-only use: no DWG writing, no bindings, no docs
./configure --prefix="$PREFIX" --disable-write --disable-bindings --disable-python --disable-docs >/dev/null
make -j"$(nproc)" >/dev/null
make install >/dev/null
"$PREFIX/bin/dwg2dxf" --version
