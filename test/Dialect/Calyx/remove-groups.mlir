// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-remove-groups))' %s | FileCheck %s

calyx.program {
  calyx.component @Z(%go : i1, %reset : i1, %clk : i1) -> (%flag: i1, %out :i2, %done: i1) {
    calyx.wires {}
    calyx.control {}
  }

  calyx.component @main(%go : i1, %reset : i1, %clk : i1) -> (%done: i1) {
    %z.go, %z.reset, %z.clk, %z.flag, %z.out, %z.done = calyx.cell "z" @Z : i1, i1, i1, i1, i2, i1
    %fsm.in, %fsm.write_en, %fsm.clk, %fsm.reset, %fsm.out, %fsm.done = calyx.register "fsm" : i2

    // CHECK-LABEL: calyx.wires
    calyx.wires {
      %undef = calyx.undef : i1

      %true = hw.constant true
      %c0_i2 = hw.constant 0 : i2
      %0 = comb.icmp eq %fsm.out, %c0_i2 : i2
      %1 = comb.xor %z.done, %true : i1
      %2 = comb.and %0, %1 : i1
      // CHECK: calyx.assign %z.go = %go, A.go
      calyx.group @A {
        %A.go = calyx.group_go %true, %2 ? : i1
        calyx.assign %z.go = %z.flag : i1
        calyx.group_done %z.done : i1
      }

      %c1_i2 = hw.constant 1 : i2
      %3 = comb.and %z.flag, %z.done : i1
      %4 = comb.icmp eq %fsm.out, %c1_i2 : i2
      %5 = comb.xor %3, %true : i1
      %6 = comb.and %4, %5 : i1
      calyx.group @B {
        %B.go = calyx.group_go %true, %6 ? : i1
        calyx.group_done %z.done, %z.flag ? : i1
      }

      %7 = comb.and %0, %z.done : i1
      %c1_i2_0 = hw.constant 1 : i2
      %8 = comb.and %4, %3 : i1
      %c-2_i2 = hw.constant -2 : i2
      %9 = comb.icmp eq %fsm.out, %c-2_i2 : i2
      calyx.group @seq {
        calyx.assign %fsm.in = %c1_i2_0, %7 ? : i2
        calyx.assign %fsm.write_en = %true, %7 ? : i1
        calyx.assign %fsm.in = %c-2_i2, %8 ? : i2
        calyx.assign %fsm.write_en = %true, %8 ? : i1
        calyx.group_done %true, %9 ? : i1
      }
      %c0_i2_1 = hw.constant 0 : i2
      calyx.assign %fsm.in = %c0_i2_1, %9 ? : i2
      calyx.assign %fsm.write_en = %true, %9 ? : i1
    }

    calyx.control {
      calyx.enable @seq {groups = [@A, @B]}
    }
  }
}
