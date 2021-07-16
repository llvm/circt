// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK: calyx.program {
calyx.program {
  // CHECK-LABEL: calyx.component @A(%in: i8, %go: i1, %clk: i1, %reset: i1) -> (%out: i8, %done: i1) {
  calyx.component @A(%in: i8, %go: i1, %clk: i1, %reset: i1) -> (%out: i8, %done: i1) {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @B(%go: i1, %clk: i1, %reset: i1) -> (%out: i1, %done: i1) {
  calyx.component @B (%go: i1, %clk: i1, %reset: i1) -> (%out: i1, %done: i1) {
    calyx.wires {}
    calyx.control {}
  }

<<<<<<< HEAD
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    // CHECK:      %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8
    // CHECK-NEXT: %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.cell "c0" @A : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %c1.in, %c1.go, %c1.clk, %c1.reset, %c1.out, %c1.done = calyx.cell "c1" @A : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %c2.go, %c2.clk, %c2.reset, %c2.out, %c2.done = calyx.cell "c2" @B : i1, i1, i1, i1, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.cell "c0" @A : i8, i1, i1, i1, i8, i1
    %c1.in, %c1.go, %c1.clk, %c1.reset, %c1.out, %c1.done = calyx.cell "c1" @A : i8, i1, i1, i1, i8, i1
    %c2.go, %c2.clk, %c2.reset, %c2.out, %c2.done = calyx.cell "c2" @B : i1, i1, i1, i1, i1
=======
  calyx.component @main() -> () {
    // CHECK:      %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8
    // CHECK-NEXT: %c0.in, %c0.out = calyx.cell "c0" @A : i8, i8
    // CHECK-NEXT: %c1.in, %c1.out = calyx.cell "c1" @A : i8, i8
    // CHECK-NEXT: %c2.out = calyx.cell "c2" @B : i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8
    %in1, %out1 = calyx.cell "c0" @A : i8, i8
    %in2, %out2 = calyx.cell "c1" @A : i8, i8
    %out3 = calyx.cell "c2" @B : i1
>>>>>>> origin/main
    %c1_i1 = constant 1 : i1

    calyx.wires {
      // CHECK: calyx.group @Group1 {
      calyx.group @Group1 {
        // CHECK: calyx.assign %c1.in = %c0.out : i8
<<<<<<< HEAD
        // CHECK-NEXT: calyx.group_done %c1.done : i1
        calyx.assign %c1.in = %c0.out : i8
        calyx.group_done %c1.done : i1
      }
      calyx.group @Group2 {
        // CHECK: calyx.assign %c1.in = %c0.out, %c2.out ? : i8
        calyx.assign %c1.in = %c0.out, %c2.out ? : i8

        // CHECK: calyx.group_done %c1.done, %0 ? : i1
        %guard = comb.and %c1_i1, %c2.out : i1
        calyx.group_done %c1.done, %guard ? : i1
=======
        // CHECK-NEXT: calyx.group_done %true : i1
        calyx.assign %in2 = %out1 : i8
        calyx.group_done %c1_i1 : i1
      }
      calyx.group @Group2 {
        // CHECK:  calyx.assign %c1.in = %c0.out, %c2.out ? : i8
        calyx.assign %in2 = %out1, %out3 ?  : i8
        // CHECK: calyx.group_done %true, %0 ? : i1
        %guard = comb.and %c1_i1, %out3 : i1
        calyx.group_done %c1_i1, %guard ? : i1
>>>>>>> origin/main
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
