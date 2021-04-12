// REQUIRES: vivado
// RUN: circt-opt %S/../../../test/handshake-runner/simple_loop.mlir --create-dataflow --canonicalize --cse --handshake-insert-buffer > %loop-handshake.mlir
// RUN: circt-opt %loop-handshake.mlir --lower-handshake-to-firrtl --firrtl-lower-types --lower-firrtl-to-rtl > %loop-rtl.mlir
// RUN: circt-translate %loop-rtl.mlir --export-verilog > %loop-export.sv
// RUN: circt-rtl-sim.py %loop-export.sv %S/driver.sv --sim %xsim% --no-default-driver | FileCheck %s
// CHECK: 42
