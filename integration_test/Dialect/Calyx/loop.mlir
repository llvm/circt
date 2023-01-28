// This test lowers an SCF construct through Calyx, FSM and (TODO)
// to RTL.
// RUN: circt-opt %s \
// RUN:     --lower-scf-to-calyx -canonicalize \
// RUN:     --calyx-remove-comb-groups --canonicalize \
// RUN:     --calyx-go-insertion --canonicalize \
// RUN:     --lower-calyx-to-fsm --canonicalize \
// RUN:     --materialize-calyx-to-fsm

// This is the end of the road (for now) for Calyx in CIRCT.
// The materialized FSM now needs to be outlined from within the
// calyx module, and within the Calyx module it can be instantiated
// as any other HW component. The FSM will then be lowered through
// the existing FSM-to-HW flow.

// CHECK: calyx.control {
// CHECK: fsm.machine @control(

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
