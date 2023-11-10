#!/usr/bin/env bash
##===- utils/get-boolector.sh - Install Boolector & Btormc -*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# Downloads, compiles, and installs Boolector into $/ext
# Boolector is the solver backend for btormc and enables its use.
#
##===----------------------------------------------------------------------===##

mkdir -p "$(dirname "$BASH_SOURCE[0]")/../ext"
EXT_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../ext" && pwd)

echo $EXT_DIR
cd $EXT_DIR

# Download and build Boolector
git clone https://github.com/boolector/boolector
cd boolector

# Download and build Lingeling
./contrib/setup-lingeling.sh

# Download and build BTOR2Tools
./contrib/setup-btor2tools.sh

# Build Boolector
./configure.sh && cd build && make

# export boolector binaries
export PATH="bin:$PATH"
