#!/bin/bash
##===- utils/get-capnp.sh - Install CapnProto ----------------*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script downloads, compiles, and installs CapnProto into $/ext
# Cap'nProto is use by ESI cosim aka Elastic Silicon Interfaces cosimulation as
# a message format and RPC client/server.
#
# It will also optionally install pycapnp.
#
##===----------------------------------------------------------------------===##

echo "Do you wish to install pycapnp? Cosim integration tests require pycapnp."
read -p "Yes to confirm: " yn
case $yn in
    [Yy]* ) pip3 install pycapnp; break ;;
    * ) echo "Skipping.";;
esac

EXT_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../ext" && pwd)
CAPNP_VER=0f1bf4fce79923fb4974aa55a53e26450f83f286
echo "Installing capnproto..."

echo $EXT_DIR
cd $EXT_DIR

git clone https://github.com/capnproto/capnproto.git
cd capnproto
git checkout $CAPNP_VER
cd c++
autoreconf -i
./configure --prefix=$EXT_DIR
make -j$(nproc)
make install
cd ../../

echo "Done."
