#!/bin/zsh
export RUSTDOCFLAGS="--html-in-header $(pwd)/docs/html/custom-header.html --document-private-items"
export CARGO_FEATURE_ENABLE_TRACE=1
export TOPO_LOG=all=info
