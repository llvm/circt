#!/bin/bash
# Downloads, compiles, and installs CapnProto into $/ext

EXT_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../ext" && pwd)
CAPNP_VER=0f1bf4fce79923fb4974aa55a53e26450f83f286

echo $EXT_DIR
cd $EXT_DIR

git clone https://github.com/capnproto/capnproto.git
cd capnproto
git checkout $CAPNP_VER
cd c++
autoreconf -i
./configure --prefix=$EXT_DIR
make -j$(nprocs)
make install
cd ../../
