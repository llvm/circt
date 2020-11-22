#!/usr/bin/env bash

# This script should be run in the docker container started in the
# 'run-docker.sh' script.

set -e

UTILS_DIR=$(dirname "$BASH_SOURCE[0]")

if [ ! -e llvm/build_20.04 ]; then
  echo "=== Building MLIR"
  $UTILS_DIR/build-llvm.sh build_20.04 build_20.04/install
fi

echo "=== Building CIRCT"
cmake -Bdocker_build \
  -DMLIR_DIR=llvm/build_20.04/lib/cmake/mlir \
  -DLLVM_DIR=llvm/build_20.04/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DVERILATOR_PATH=/usr/bin/verilator \
  -DCAPNP_PATH=/usr \
  -DCMAKE_BUILD_TYPE=DEBUG

cmake --build docker_build -j$(nproc) --target check-circt
cmake --build docker_build -j$(nproc) --target check-circt-integration
