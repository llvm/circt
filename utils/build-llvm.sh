#!/usr/bin/env bash

BUILD_DIR=${1:-"build"}
INSTALL_DIR=${2:-"install"}

# Build LLVM with the standard options. Intended to be called from the github
# workflows.
mkdir -p llvm/$BUILD_DIR
mkdir -p llvm/$INSTALL_DIR
cd llvm/$BUILD_DIR
cmake ../llvm \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
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

cmake --build . --target install -- -j$(nproc)
