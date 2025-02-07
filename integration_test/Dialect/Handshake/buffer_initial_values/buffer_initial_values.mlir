// REQUIRES: iverilog,cocotb

// Test the original HandshakeToHW flow.

// RUN: hlstool %s --dynamic-hw --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=buffer_initial_values --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=buffer_initial_values --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// Test the DC lowering flow.
// TODO: This test does not pass with DC as DC to HW does not support buffer initValues.
// RUNx: hlstool %s --dynamic-hw --dc --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUNx: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=buffer_initial_values --pythonFolder="%S,%S/.." %t.sv %esi_prims 2>&1 | FileCheck %s

// RUNx: hlstool %s --dynamic-hw --dc --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUNx: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=buffer_initial_values --pythonFolder="%S,%S/.." %t.sv %esi_prims 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

module {
  handshake.func @top(%arg0: none) -> (i32, none) attributes {argNames = ["inCtrl"], resNames = ["out0", "outCtrl"]} {
    %d = buffer [2] seq %dTrue {initValues = [20, 10]} : i32
    %ctrl = merge %arg0, %cTrue : none

    %6 = constant %ctrl {value = 10 : i32} : i32
    %cond = arith.cmpi ne, %d, %6 : i32

    %dTrue, %dFalse = cond_br %cond, %d : i32
    %cTrue, %cFalse = cond_br %cond, %ctrl : none

    return %dFalse, %cFalse : i32, none
  }
}
