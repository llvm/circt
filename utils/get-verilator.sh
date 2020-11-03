#!/bin/bash
# Downloads, compiles, and installs Verilator into $/ext

EXT_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../ext" && pwd)
VERILATOR_VER=4.102

echo $EXT_DIR
cd $EXT_DIR

wget https://github.com/verilator/verilator/archive/v$VERILATOR_VER.tar.gz
tar -zxf v$VERILATOR_VER.tar.gz
cd verilator-$VERILATOR_VER
autoconf
./configure --prefix=$EXT_DIR
make -j$(nproc)
make install
cd ..
rm -r verilator-$VERILATOR_VER v$VERILATOR_VER.tar.gz
