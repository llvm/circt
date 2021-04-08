// REQUIRES: vivado
// RUN: circt-opt %S/../../../test/handshake-runner/simple_loop.mlir --create-dataflow --canonicalize --cse --handshake-insert-buffer > %handshake.mlir
// RUN: circt-opt %handshake.mlir --lower-handshake-to-firrtl --firrtl-lower-types --lower-firrtl-to-rtl > %rtl.mlir
// RUN: circt-translate %rtl.mlir --export-verilog > %export.sv
// RUN: circt-rtl-sim.py %export.sv %S/driver.sv --sim %xsim% --no-default-driver | FileCheck %s
// CHECK: 42
