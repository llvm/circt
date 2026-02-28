// REQUIRES: iverilog,cocotb

// Test the original HandshakeToHW flow.

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: hlstool %s --dynamic-hw --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=buffer_arg_init --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=buffer_arg_init --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// Test the DC lowering flow.
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: hlstool %s --dynamic-hw --dc --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=buffer_arg_init --pythonFolder="%S,%S/.." %t.sv %esi_prims 2>&1 | FileCheck %s

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: hlstool %s --dynamic-hw --dc --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=buffer_arg_init --pythonFolder="%S,%S/.." %t.sv %esi_prims 2>&1 | FileCheck %s


// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

handshake.func @top(%arg0: none, %arg1: none, ...) -> (none) attributes {argNames = ["inCtrl", "startTokens"], resNames = ["outCtrl"]}{
  %bin = merge %arg1, %f#1 : none
  %3 = buffer [1] seq %bin : none
  %ctrlOut = join %3, %arg0 : none, none
  %f:2 = fork [2] %ctrlOut : none
  return %f#0 : none
}
