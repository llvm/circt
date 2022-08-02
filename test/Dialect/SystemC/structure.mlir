// RUN: circt-opt %s | FileCheck %s

// CHECK: systemc.module @adder (sc_in %summand_a: i32, sc_in %summand_b: i32, sc_out %sum: i32) {
systemc.module @adder (sc_in %summand_a: i32, sc_in %summand_b: i32, sc_out %sum: i32) {
  //CHECK-NEXT: systemc.ctor {
  systemc.ctor {
    // CHECK-NEXT: systemc.method %addFunc
    systemc.method %addFunc
    // CHECK-NEXT: systemc.thread %addFunc
    systemc.thread %addFunc
  //CHECK-NEXT: }
  }
  // CHECK-NEXT: %addFunc = systemc.func {
  %addFunc = systemc.func {
    // CHECK-NEXT: [[RES:%.*]] = comb.add %summand_a, %summand_b : i32
    %res = comb.add %summand_a, %summand_b : i32
    // CHECK-NEXT: systemc.alias %sum, [[RES]] : i32
    systemc.alias %sum, %res : i32
  // CHECK-NEXT: }
  }
// CHECK-NEXT: }
}

// CHECK: systemc.module @mixedPorts (sc_out %port0: i4, sc_in %port1: i32, sc_out %port2: i4, sc_inout %port3: i8)
systemc.module @mixedPorts (sc_out %port0: i4, sc_in %port1: i32, sc_out %port2: i4, sc_inout %port3: i8) {}

// CHECK-LABEL: systemc.module @signals
systemc.module @signals () {
  // CHECK-NEXT: %signal0 = systemc.signal : i32
  %signal0 = systemc.signal : i32
  // CHECK-NEXT: %signal1 = systemc.signal : i1
  %signal1 = systemc.signal : i1
}
