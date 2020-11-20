#!/bin/bash

# Build LLVM with the standard options. Intended to be called from the github
# workflows.
mkdir llvm/build
mkdir llvm/install
cd llvm/build
cmake ../llvm \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_INSTALL_PREFIX=../install \
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
