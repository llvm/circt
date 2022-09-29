// RUN: circt-opt %s -export-verilog -verify-diagnostics -o %t.mlir --lowering-options=disallowInlineMux | FileCheck %s

// CHECK-LABEL

// CHECK-LABEL: module ShiftMux
hw.module @ShiftMux(%p: i1, %x: i45) -> (o: i45) {
  // CHECK: wire [44:0] [[GEN:.+]] = p ? 45'h5 : 45'h8
  // CHECK: assign o = $signed($signed(x) >>> [[GEN]]
  %c5_i45 = hw.constant 5 : i45
  %c8_i45 = hw.constant 8 : i45
  %0 = comb.mux %p, %c5_i45, %c8_i45 : i45
  %1 = comb.shrs %x, %0 : i45
  hw.output %1 : i45
}

// CHECK-LABEL: module AssignMux
hw.module @AssignMux(%p: i1, %x: i45) -> (o: i45) {
  // CHECK-NOT: wire
  // CHECK: assign o = p ? 45'h5 : x
  %c5_i45 = hw.constant 5 : i45
  %0 = comb.mux %p, %c5_i45, %x : i45
  hw.output %0 : i45
}
