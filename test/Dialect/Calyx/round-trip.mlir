// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK: calyx.program "main" {
calyx.program "main" {
  // CHECK-LABEL: calyx.component @A(%in: i8, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i8, %done: i1 {done}) {
  calyx.component @A(%in: i8, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i8, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @B(%in: i8, %clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
  calyx.component @B (%in: i8, %clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }

  // CHECK-LABEL:   calyx.component @main(%clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%done: i1 {done}) {
  calyx.component @main(%clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // CHECK:      %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory "m"<[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    // CHECK-NEXT: %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %c1.in, %c1.go, %c1.clk, %c1.reset, %c1.out, %c1.done = calyx.instance "c1" @A : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %c2.in, %c2.clk, %c2.go, %c2.reset, %c2.out, %c2.done = calyx.instance "c2" @B : i8, i1, i1, i1, i1, i1
    // CHECK-NEXT: %adder.left, %adder.right, %adder.out = calyx.std_add "adder" : i8, i8, i8
    // CHECK-NEXT: %gt.left, %gt.right, %gt.out = calyx.std_gt "gt" : i8, i8, i1
    // CHECK-NEXT: %pad.in, %pad.out = calyx.std_pad "pad" : i8, i9
    // CHECK-NEXT: %slice.in, %slice.out = calyx.std_slice "slice" : i8, i7
    // CHECK-NEXT: %not.in, %not.out = calyx.std_not "not" : i8, i8
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8, i1, i1, i1, i8, i1
    %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory "m"<[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i8, i1, i1, i1, i8, i1
    %c1.in, %c1.go, %c1.clk, %c1.reset, %c1.out, %c1.done = calyx.instance "c1" @A : i8, i1, i1, i1, i8, i1
    %c2.in, %c2.clk, %c2.go, %c2.reset, %c2.out, %c2.done = calyx.instance "c2" @B : i8, i1, i1, i1, i1, i1
    %adder.left, %adder.right, %adder.out = calyx.std_add "adder" : i8, i8, i8
    %gt.left, %gt.right, %gt.out = calyx.std_gt "gt" : i8, i8, i1
    %pad.in, %pad.out = calyx.std_pad "pad" : i8, i9
    %slice.in, %slice.out = calyx.std_slice "slice" : i8, i7
    %not.in, %not.out = calyx.std_not "not" : i8, i8
    %c1_i1 = hw.constant 1 : i1
    %c0_i6 = hw.constant 0 : i6
    %c0_i8 = hw.constant 0 : i8

    calyx.wires {
      // CHECK: calyx.group @Group1 {
      calyx.group @Group1 {
        // CHECK: calyx.assign %c1.in = %c0.out : i8
        // CHECK-NEXT: calyx.group_done %c1.done : i1
        calyx.assign %c1.in = %c0.out : i8
        calyx.group_done %c1.done : i1
      }
      calyx.comb_group @ReadMemory {
        // CHECK: calyx.assign %m.addr0 = %c0_i6 : i6
        // CHECK-NEXT: calyx.assign %m.addr1 = %c0_i6 : i6
        // CHECK-NEXT: calyx.assign %gt.left = %m.read_data : i8
        // CHECK-NEXT: calyx.assign %gt.right = %c0_i8 : i8
        calyx.assign %m.addr0 = %c0_i6 : i6
        calyx.assign %m.addr1 = %c0_i6 : i6
        calyx.assign %gt.left = %m.read_data : i8
        calyx.assign %gt.right = %c0_i8 : i8
      }
      calyx.group @Group3 {
        calyx.assign %r.in = %c0.out : i8
        calyx.assign %r.write_en = %c1_i1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      // CHECK:      calyx.seq {
      // CHECK-NEXT: calyx.seq {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: calyx.enable @Group3
      // CHECK-NEXT: calyx.seq {
      // CHECK-NEXT: calyx.if %gt.out with @ReadMemory {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: } else {
      // CHECK-NEXT: calyx.enable @Group3
      // CHECK-NEXT: }
      // CHECK-NEXT: calyx.if %c2.out {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: }
      // CHECK-NEXT: calyx.while %gt.out with @ReadMemory {
      // CHECK-NEXT: calyx.while %c2.out {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK:      calyx.par {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: calyx.enable @Group3
      calyx.seq {
        calyx.seq {
          calyx.enable @Group1
          calyx.enable @Group3
          calyx.seq {
            calyx.if %gt.out with @ReadMemory {
              calyx.enable @Group1
            } else {
              calyx.enable @Group3
            }
            calyx.if %c2.out {
              calyx.enable @Group1
            }
            calyx.while %gt.out with @ReadMemory {
              calyx.while %c2.out {
                calyx.enable @Group1
              }
            }
          }
        }
        calyx.par {
          calyx.enable @Group1
          calyx.enable @Group3
        }
      }
    }
  }
}
