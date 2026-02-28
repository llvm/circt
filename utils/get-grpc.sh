#!/usr/bin/env bash
##===- utils/get-grpc.sh - Install gRPC (for ESI runtime) ----*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
#
##===----------------------------------------------------------------------===##

mkdir -p "$(dirname "$BASH_SOURCE[0]")/../ext"
EXT_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../ext" && pwd)
# v1.54.2 is the version in Ubuntu 22.04
GRPC_VER=1.54.2
echo "Installing gRPC..."

echo $EXT_DIR
cd $EXT_DIR

if [ ! -d grpc ]; then
  git clone --recurse-submodules -b v$GRPC_VER https://github.com/grpc/grpc
fi
cd grpc
mkdir -p cmake/build
cd cmake/build
cmake -S ../.. -B . -DCMAKE_INSTALL_PREFIX=$EXT_DIR \
  -GNinja \
  -DgRPC_INSTALL=ON \
  -DgRPC_ZLIB_PROVIDER=package \
  -DCMAKE_BUILD_TYPE=Debug
ninja
cmake --install . --prefix $EXT_DIR

cd ../../../
rm -rf grpc

echo "Done."
