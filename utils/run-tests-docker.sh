#!/usr/bin/env bash
##===- utils/run-tests-docker.sh - Run tests in docker -------*- Script -*-===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# This script should be run in the docker container started in the
# 'run-docker.sh' script.
#
##===----------------------------------------------------------------------===##

set -e
BUILD_DIR=${1:-"20.04"}

UTILS_DIR=$(dirname "$BASH_SOURCE[0]")

if [ ! -e llvm/build_$BUILD_DIR ]; then
  echo "=== Building MLIR"
  $UTILS_DIR/build-llvm.sh build_$BUILD_DIR build_$BUILD_DIR/install
fi

echo "=== Building CIRCT"
cmake -Bbuild_$BUILD_DIR \
  -DMLIR_DIR=llvm/build_$BUILD_DIR/lib/cmake/mlir \
  -DLLVM_DIR=llvm/build_$BUILD_DIR/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG

cmake --build build_$BUILD_DIR -j$(nproc) --target check-circt
cmake --build build_$BUILD_DIR -j$(nproc) --target check-circt-integration
