// REQUIRES: iverilog,cocotb

// This test is executed with all different buffering strategies

// RUN: hlstool %s --dynamic-firrtl --ir-input-level 1 --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=sync_op --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-firrtl --ir-input-level 1 --buffering-strategy=allFIFO --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=sync_op --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-firrtl --ir-input-level 1 --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=sync_op --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-hw --ir-input-level 1 --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=sync_op --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

handshake.func @top(%arg0: i64, %arg1: none) -> (i64, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  %0:2 = sync %arg0, %arg1 : i64, none
  return %0#0, %0#1 : i64, none
}
