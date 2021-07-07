// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK: calyx.program {
calyx.program {

  // CHECK-LABEL:  calyx.component @ComponentWithInAndOutPorts(%in1: i32, %in2: i16) -> (%out1: i32, %out2: i8) {
  calyx.component @ComponentWithInAndOutPorts(%in1: i32, %in2: i16) -> (%out1: i32, %out2: i8) {
    // CHECK:        calyx.wires {
    // CHECK:        calyx.control {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @ComponentWithInPort(%x: i64) -> () {
  calyx.component @ComponentWithInPort(%x: i64) -> () {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @ComponentWithOutPort() -> (%y: i64) {
  calyx.component @ComponentWithOutPort() -> (%y: i64) {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @ComponentWithNoPorts() -> () {
  calyx.component @ComponentWithNoPorts() -> () {
    calyx.wires {}
    calyx.control {}
  }

  calyx.component @A(%in: i8) -> (%out: i8) {
    calyx.wires {}
    calyx.control {}
  }

  calyx.component @B () -> (%out: i1) {
    calyx.wires {}
    calyx.control {}
  }

  calyx.component @main() -> () {
    %in1, %out1 = calyx.cell "c0" @A : i8, i8
    %in2, %out2 = calyx.cell "c1" @A : i8, i8
    %out3 = calyx.cell "c2" @B : i1
    %c1_i1 = constant 1 : i1

    calyx.wires {
      // CHECK: calyx.group @Group1 {
      calyx.group @Group1 {
        // CHECK: calyx.assign %1#0 = %0#1 : i8
        // CHECK-NEXT: calyx.done %true : i1
        calyx.assign %in2 = %out1 : i8
        calyx.done %c1_i1 : i1
      }
      calyx.group @Group2 {
        // CHECK:  calyx.assign %1#0 = %0#1, %2 ? : i8
        calyx.assign %in2 = %out1, %out3 ?  : i8

        // CHECK: calyx.done %true, %3 ? : i1
        %guard = comb.and %c1_i1, %out3 : i1
        calyx.done %c1_i1, %guard ? : i1
      }
    }
    calyx.control {
      // CHECK:      calyx.seq {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: calyx.enable @Group2
      // CHECK-NEXT: calyx.seq {
      // CHECK-NEXT: calyx.enable @Group1
      calyx.seq {
        calyx.enable @Group1
        calyx.enable @Group2
        calyx.seq {
          calyx.enable @Group1
        }
      }
    }
  }
}
