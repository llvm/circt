// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(arc-lower-hw-triggered))' %s | FileCheck %s

hw.module @simple(in %clk : !seq.clock, in %cond : i1, in %value : i8) {
  %trg = seq.from_clock %clk
  hw.triggered posedge %trg(%cond, %value) : i1, i8 {
  ^bb0(%arg0 : i1, %arg1 : i8):
    scf.if %arg0 {
      arc.sim.emit "value", %arg1 : i8
    }
  }
  // CHECK-LABEL: hw.module @simple
  // CHECK: arc.clock_domain (%cond, %value) clock %clk : (i1, i8) -> () {
  // CHECK-NEXT:   ^bb0(%[[COND:.*]]: i1, %[[VALUE:.*]]: i8):
  // CHECK-NEXT:   arc.execute (%[[COND]], %[[VALUE]] : i1, i8) {
  // CHECK-NEXT:   ^bb0(%[[EXEC_COND:.*]]: i1, %[[EXEC_VALUE:.*]]: i8):
  // CHECK-NEXT:     scf.if %[[EXEC_COND]] {
  // CHECK-NEXT:       arc.sim.emit "value", %[[EXEC_VALUE]] : i8
  // CHECK-NEXT:     }
  // CHECK-NEXT:     arc.output
  // CHECK-NEXT:   }
  hw.output
}
