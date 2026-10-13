// REQUIRES: iverilog,cocotb

// Test the original HandshakeToHW flow.

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: hlstool %s --dynamic-hw --input-level core --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=sync_backedge --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: hlstool %s --dynamic-hw --input-level core --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=sync_backedge --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// Test the DC lowering flow.
// TODO: This test does not pass with DC as DC to HW does not support buffer initValues.
// RUNx: rm -rf %t.dir && mkdir %t.dir
// RUNx: hlstool %s --dynamic-hw --dc --input-level core --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv
// RUNx: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=sync_backedge --pythonFolder="%S,%S/.." %t.sv %esi_prims 2>&1 | FileCheck %s

// RUNx: rm -rf %t.dir && mkdir %t.dir
// RUNx: hlstool %s --dynamic-hw --dc --input-level core --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv
// RUNx: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=sync_backedge --pythonFolder="%S,%S/.." %t.sv %esi_prims 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

handshake.func @top(%arg0: i64, %arg1: none, ...) -> (i64, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  %0 = buffer [1] seq %10 {initValues = [0]} : none
  %1:3 = sync %arg0, %arg1, %0 : i64, none, none
  %2:2 = fork [2] %1#0 : i64
  %3:2 = fork [2] %1#1 : none
  %4 = constant %3#0 {value = 0 : i64} : i64
  %5 = arith.cmpi eq, %4, %2#1 : i64
  %6:2 = fork [2] %5 : i1
  %trueResult, %falseResult = cond_br %6#1, %2#0 : i64
  %trueResult_0, %falseResult_1 = cond_br %6#0, %3#1 : none
  // introduce additional delay to simulate a slow branch
  %buffer = buffer [10] seq %trueResult_0 : none
  %result, %index = control_merge %buffer, %falseResult_1 : none, index
  %7:2 = fork [2] %result : none
  %8 = mux %index [%trueResult, %falseResult] : index, i64
  %9:2 = fork [2] %8 : i64
  %10 = join %9#0, %7#0, %1#2 : i64, none, none
  return %9#1, %7#1 : i64, none
}
