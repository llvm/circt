// This test lowers an SCF construct through Calyx, FSM and (TODO)
// to RTL.
// RUN: hlstool --calyx-hw --output-level=post-compile --ir %s | FileCheck %s

// This is the end of the road for this example since there (as of writing)
// does not yet exist a lowering for calyx.memory operations.

// CHECK: fsm.machine @control
// CHECK: calyx.component @main

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %0 = memref.alloc() : memref<64xi32>
  %1 = memref.alloc() : memref<64xi32>
  scf.while(%arg0 = %c0) : (index) -> (index) {
      %cond = arith.cmpi ult, %arg0, %c64 : index
      scf.condition(%cond) %arg0 : index
  } do {
  ^bb0(%arg1: index):
      %v = memref.load %0[%arg1] : memref<64xi32>
      memref.store %v, %1[%arg1] : memref<64xi32>
      %inc = arith.addi %arg1, %c1 : index
      scf.yield %inc : index
  }
  return
}
