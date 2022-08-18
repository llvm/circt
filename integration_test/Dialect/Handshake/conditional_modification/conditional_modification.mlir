// REQUIRES: iverilog,cocotb

// This test is executed with all different buffering strategies

// RUN: circt-opt %s --lower-std-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=conditional_modification --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: circt-opt %s --lower-std-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=allFIFO --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=conditional_modification --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: circt-opt %s --lower-std-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=cycles --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=conditional_modification --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK:      ** TEST
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** conditional_modification.oneInput
// CHECK-NEXT: ** conditional_modification.multiple
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** TESTS=2 PASS=2 FAIL=0 SKIP=0
// CHECK-NEXT: ********************************

module {
  func.func @top(%n0: i32, %n1: i32, %cond: i1) -> (i32, i32) {
    cf.cond_br %cond, ^bb1(%n0, %n1 : i32, i32), ^bb2(%n0, %n1 : i32, i32)
  ^bb1(%t0: i32, %t1: i32):
    %o0 = arith.addi %t0, %t1 : i32
    %o1 = arith.subi %t0, %t1 : i32
    cf.br ^bb2(%o0, %o1 : i32, i32)
  ^bb2(%r0: i32, %r1: i32):
    return %r0, %r1 : i32, i32
  }
}

