#!/bin/bash

# This script should be run in the docker container started in the
# 'run-docker.sh' script.

set -e

cmake -Bdocker_build \
  -DMLIR_DIR=llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=llvm/build/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG

cmake --build docker_build -j$(nproc) --target check-circt
cmake --build docker_build -j$(nproc) --target check-circt-integration
