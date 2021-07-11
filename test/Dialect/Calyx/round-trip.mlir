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
    // CHECK:      %r.in, %r.out, %r.write_en, %r.done = calyx.register "r" : i8, i8, i1, i1
    // CHECK-NEXT: %c0.in, %c0.out = calyx.cell "c0" @A : i8, i8
    // CHECK-NEXT: %c1.in, %c1.out = calyx.cell "c1" @A : i8, i8
    // CHECK-NEXT: %c2.out = calyx.cell "c2" @B : i1
    %in, %out, %write_en, %done = calyx.register "r" : i8, i8, i1, i1
    %in1, %out1 = calyx.cell "c0" @A : i8, i8
    %in2, %out2 = calyx.cell "c1" @A : i8, i8
    %out3 = calyx.cell "c2" @B : i1
    %c1_i1 = constant 1 : i1

    calyx.wires {
      // CHECK: calyx.group @Group1 {
      calyx.group @Group1 {
        // CHECK: calyx.assign %c1.in = %c0.out : i8
        // CHECK-NEXT: %0 = calyx.group_done %true : i1
        calyx.assign %in2 = %out1 : i8
        %d0 = calyx.group_done %c1_i1 : i1
      }
      calyx.group @Group2 {
        // CHECK:  calyx.assign %c1.in = %c0.out, %c2.out ? : i8
        calyx.assign %in2 = %out1, %out3 ?  : i8

        // CHECK: %1 = calyx.group_done %true, %0 ? : i1
        %guard = comb.and %c1_i1, %out3 : i1
        %d1 = calyx.group_done %c1_i1, %guard ? : i1
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
