// REQUIRES: vivado
// RUN: circt-opt %S/../../../test/handshake-runner/matmul-check-std.mlir --create-dataflow --canonicalize --cse --handshake-insert-buffer > %matmul-handshake.mlir
// RUN: circt-opt %matmul-handshake.mlir --lower-handshake-to-firrtl --firrtl-lower-types --lower-firrtl-to-rtl > %matmul-rtl.mlir
// RUN: circt-translate %matmul-rtl.mlir --export-verilog > %matmul-export.sv
// RUN: circt-rtl-sim.py %matmul-export.sv %S/driver.sv --sim %xsim% --no-default-driver | FileCheck %s
// CHECK: 200
