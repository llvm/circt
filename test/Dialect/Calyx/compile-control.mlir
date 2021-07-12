// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-compile-control))' %s | FileCheck %s

calyx.program {
  calyx.component @Z(%go : i1, %reset : i1, %clk : i1) -> (%done: i1) {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @main(%go : i1, %reset : i1, %clk : i1) -> (%done: i1) {
  // CHECK-NEXT: %fsm.in, %fsm.write_en, %fsm.clk, %fsm.reset, %fsm.out, %fsm.done = calyx.register "r" : i2
  calyx.component @main(%go : i1, %reset : i1, %clk : i1) -> (%done: i1) {
    %z.go, %z.reset, %z.clk, %z.done = calyx.cell "z" @Z : i1, i1, i1, i1

    // CHECK-LABEL: calyx.wires
    calyx.wires {
      %undef = calyx.undef : i1

      // CHECK-NEXT:  %true = hw.constant true
      // CHECK-NEXT:  %c0_i2 = hw.constant 0 : i2
      // CHECK-NEXT:  %1 = comb.icmp eq %fsm.out, %c0_i2 : i2
      // CHECK-NEXT:  %2 = comb.xor %0, %true : i1
      // CHECK-NEXT:  %3 = comb.and %1, %2 : i1
      // CHECK-NEXT:  calyx.group @A {
      // CHECK-NEXT:    %10 = calyx.group_go %true, %3 ? : i1
      // CHECK-NEXT:    calyx.group_done %z.done : i1
      // CHECK-NEXT:  }
      calyx.group @A {
        %A.go = calyx.group_go %undef : i1
        calyx.group_done %z.done : i1
      }

      // CHECK-NEXT: %c1_i2 = hw.constant 1 : i2
      // CHECK-NEXT:  %4 = comb.icmp eq %fsm.out, %c1_i2 : i2
      // CHECK-NEXT:  %5 = comb.xor %0, %true : i1
      // CHECK-NEXT:  %6 = comb.and %4, %5 : i1
      // CHECK-NEXT:  calyx.group @B {
      // CHECK-NEXT:    %10 = calyx.group_go %true, %6 ? : i1
      // CHECK-NEXT:    calyx.group_done %0 : i1
      // CHECK-NEXT:  }
      calyx.group @B {
        %B_go = calyx.group_go %undef : i1
        calyx.group_done %z.done : i1
      }

      // CHECK-NEXT:  %7 = comb.and %1, %0 : i1
      // CHECK-NEXT:  %c1_i2_0 = hw.constant 1 : i2
      // CHECK-NEXT:  %8 = comb.and %4, %0 : i1
      // CHECK-NEXT:  %c-2_i2 = hw.constant -2 : i2
      // CHECK-NEXT:  %c-2_i2_1 = hw.constant -2 : i2
      // CHECK-NEXT:  %9 = comb.icmp eq %fsm.out, %c-2_i2_1 : i2
      // CHECK-NEXT:  calyx.group @seq {
      // CHECK-NEXT:    calyx.assign %fsm.in = %c1_i2_0, %7 ? : i2
      // CHECK-NEXT:    calyx.assign %fsm.write_en = %true, %7 ? : i1
      // CHECK-NEXT:    calyx.assign %fsm.in = %c-2_i2, %8 ? : i2
      // CHECK-NEXT:    calyx.assign %fsm.write_en = %true, %8 ? : i1
      // CHECK-NEXT:    calyx.group_done %true, %9 ? : i1
      // CHECK-NEXT:  }
    }

    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.enable @seq {groups = [@A, @B]}
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.enable @A
        calyx.enable @B
      }
    }
  }
}
