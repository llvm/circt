// REQUIRES: iverilog,cocotb

// This test is executed with all different buffering strategies

// RUN: circt-opt %s \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=buffer_init_none --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: circt-opt %s \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=allFIFO --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=buffer_init_none --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: circt-opt %s \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=cycles --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=buffer_init_none --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

module {
  handshake.func @top(%arg0: none, ...) -> (none) attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]}{
    %3 = buffer [1] seq %f#1 {initValues = [0]}: none
    %ctrlOut = join %3, %arg0 : none
    %f:2 = fork [2] %ctrlOut : none
    return %f#0 : none
  }
}

