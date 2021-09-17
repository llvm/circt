// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-go-insertion))' %s | FileCheck %s

calyx.program {
  calyx.component @A(%in: i8, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i8, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i8, i1, i1, i1, i8, i1
    calyx.wires {
      // CHECK: %0 = calyx.undef : i1
      // CHECK-LABEL: calyx.group @Group1 {
      // CHECK-NEXT:    %Group1.go = calyx.group_go %0 : i1
      // CHECK-NEXT:    %1 = comb.and %c0.done, %Group1.go : i1
      // CHECK-NEXT:    calyx.assign %c0.in = %c0.out, %Group1.go ? : i8
      // CHECK-NEXT:    calyx.assign %c0.in = %c0.out, %1 ? : i8
      // CHECK-NEXT:    calyx.group_done %c0.done : i1
      // CHECK-NEXT:  }
      calyx.group @Group1 {
        calyx.assign %c0.in = %c0.out : i8
        calyx.assign %c0.in = %c0.out, %c0.done ? : i8
        calyx.group_done %c0.done : i1
      }
    }
    calyx.control {
      calyx.seq { calyx.enable @Group1 }
    }
  }
}
