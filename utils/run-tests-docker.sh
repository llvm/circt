#!/usr/bin/env bash

# This script should be run in the docker container started in the
# 'run-docker.sh' script.

set -e

echo "=== Building MLIR"
mkdir -p llvm/build_20.04
cd llvm/build_20.04
cmake ../llvm \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_PROJECTS='mlir' \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build . -j$(nproc)

echo "=== Building CIRCT"
cmake -Bdocker_build \
  -DMLIR_DIR=llvm/build_20.04/lib/cmake/mlir \
  -DLLVM_DIR=llvm/build_20.04/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG

cmake --build docker_build -j$(nproc) --target check-circt
cmake --build docker_build -j$(nproc) --target check-circt-integration
