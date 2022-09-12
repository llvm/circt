// REQUIRES: iverilog,cocotb

// This test is executed with all different buffering strategies

// RUN: hlstool %s --dynamic --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=tp_memory --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: circt-opt %s --lower-std-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=allFIFO --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=tp_memory --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: circt-opt %s --lower-std-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=cycles --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=tp_memory --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK:      ** TEST
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** tp_memory.oneInput
// CHECK-NEXT: ** tp_memory.multipleInputs
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** TESTS=2 PASS=2 FAIL=0 SKIP=0
// CHECK-NEXT: ********************************

module {
  func.func @top(%val: i64, %write: i1) -> i64 {
    %mem = memref.alloc() : memref<1xi64>
    %0 = arith.constant 0 : index
    cf.cond_br %write, ^w, ^r
  ^w:
    memref.store %val, %mem[%0] : memref<1xi64>
    cf.br ^end(%val: i64)
  ^r:
    %r = memref.load %mem[%0] : memref<1xi64>
    cf.br ^end(%r: i64)
  ^end(%res: i64):
    return %res: i64
  }
}
