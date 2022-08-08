// REQUIRES: ieee-sim
// UNSUPPORTED: ieee-sim-iverilog

// This test is executed with all different buffering strategies

// RUN: circt-opt %s --lower-std-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %t.sv && \
// RUN: circt-rtl-sim.py %t.sv %S/driver_with_input.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s

// RUN: circt-opt %s --lower-std-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=allFIFO --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %t.sv && \
// RUN: circt-rtl-sim.py %t.sv %S/driver_with_input.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s

// RUN: circt-opt %s --lower-std-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=cycles --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %t.sv && \
// RUN: circt-rtl-sim.py %t.sv %S/driver_with_input.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s

// CHECK: ## run -all
// CHECK-NEXT: Result={{.*}}100
// CHECK-NEXT: Result={{.*}}24

module {
  func.func @top(%val: i64) -> i64 {
    %c0 = arith.constant 0 : i64
    %cond = arith.cmpi eq, %c0, %val : i64
    cf.cond_br %cond, ^1(%c0: i64), ^2(%val: i64)
  ^1(%i: i64):
    %c10 = arith.constant 100 : i64
    %lcond = arith.cmpi eq, %c10, %i : i64
    cf.cond_br %lcond, ^2(%i: i64), ^body
  ^body:
    %c1 = arith.constant 1 : i64
    %ni = arith.addi %i, %c1 : i64
    cf.br ^1(%ni: i64)
  ^2(%res: i64):
    return %res : i64
  }
}

