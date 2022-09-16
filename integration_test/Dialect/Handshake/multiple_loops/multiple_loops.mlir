// REQUIRES: iverilog,cocotb

// This test is executed with all different buffering strategies

// RUN: hlstool %s --dynamic --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=multiple_loops --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: circt-opt %s --lower-std-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=allFIFO --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=multiple_loops --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: circt-opt %s --lower-std-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=cycles --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=multiple_loops --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[NUM:.*]] PASS=[[NUM]] FAIL=0 SKIP=0

func.func @top(%N: i64) -> (i64, i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  cf.br ^bb1(%c0, %c0: i64, i64)
^bb1(%i: i64, %acc: i64):
  %cond = arith.cmpi sle, %i, %N : i64
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %na = arith.addi %i, %acc : i64
  %ni = arith.addi %i, %c1 : i64
  cf.br ^bb1(%ni, %na : i64, i64)
^bb3:
  cf.br ^bb4(%c1, %c1 : i64, i64)
^bb4(%j: i64, %acc2: i64):
  %cond2 = arith.cmpi sle, %j, %N : i64
  cf.cond_br %cond2, ^bb5, ^bb6
^bb5:
  %na2 = arith.muli %j, %acc2 : i64
  %nj = arith.addi %j, %c1 : i64
  cf.br ^bb4(%nj, %na2 : i64, i64)
^bb6:
  return %acc, %acc2: i64, i64
}
