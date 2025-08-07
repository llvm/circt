// REQUIRES: iverilog,cocotb

// Test the original HandshakeToHW flow.

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: hlstool %s --dynamic-hw --input-level core --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=sync_op --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: hlstool %s --dynamic-hw --input-level core --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=sync_op --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// Test the DC lowering flow.
// TODO: This test does not pass with DC. Debug.
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: hlstool %s --dynamic-hw --dc --input-level core --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=sync_op --pythonFolder="%S,%S/.." %t.sv %esi_prims 2>&1 | FileCheck %s

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: hlstool %s --dynamic-hw --dc --input-level core --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=sync_op --pythonFolder="%S,%S/.." %t.sv %esi_prims 2>&1 | FileCheck %s


// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

handshake.func @top(%arg0: i64, %arg1: none) -> (i64, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  %0:2 = sync %arg0, %arg1 : i64, none
  return %0#0, %0#1 : i64, none
}
