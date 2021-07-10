// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-compile-control))' %s | FileCheck %s

calyx.program {
  calyx.component @Z() -> (%out: i1) {
    calyx.wires {}
    calyx.control {}
  }
  // CHECK-LABEL: calyx.component @register(in: i2, write_en: i1) -> (out: i2)
  // CHECK: calyx.wires
  // CHECK: calyx.control

  // CHECK-LABEL: calyx.component @main() -> ()
  // CHECK-NEXT: %0:3 = calyx.cell "fsm" @register : i2, i1, i2
  // CHECK-NEXT: %1 = calyx.cell "z" @Z : i1
  calyx.component @main() -> () {
    %out = calyx.cell "z" @Z : i1

    // CHECK-LABEL: calyx.wires
    calyx.wires {
      %undef = calyx.undef : i1

      // CHECK-NEXT:  %true = hw.constant true
      // CHECK-NEXT:  %c0_i2 = hw.constant 0 : i2
      // CHECK-NEXT:  %2 = comb.icmp eq %0#2, %c0_i2 : i2
      // CHECK-NEXT:  %3 = comb.xor %1, %true : i1
      // CHECK-NEXT:  %4 = comb.and %2, %3 : i1
      // CHECK-NEXT:  calyx.group @A {
      // CHECK-NEXT:    %11 = calyx.group_go %true, %4 ? : i1
      // CHECK-NEXT:    calyx.group_done %1 : i1
      // CHECK-NEXT:  }
      calyx.group @A {
        %A_go = calyx.group_go %undef : i1
        calyx.group_done %out : i1
      }

      // CHECK-NEXT: %c1_i2 = hw.constant 1 : i2
      // CHECK-NEXT:  %5 = comb.icmp eq %0#2, %c1_i2 : i2
      // CHECK-NEXT:  %6 = comb.xor %1, %true : i1
      // CHECK-NEXT:  %7 = comb.and %5, %6 : i1
      // CHECK-NEXT:  calyx.group @B {
      // CHECK-NEXT:    %11 = calyx.group_go %true, %7 ? : i1
      // CHECK-NEXT:    calyx.group_done %1 : i1
      // CHECK-NEXT:  }
      calyx.group @B {
        %B_go = calyx.group_go %undef : i1
        calyx.group_done %out : i1
      }

      // CHECK-NEXT:  %8 = comb.and %2, %1 : i1
      // CHECK-NEXT:  %c1_i2_0 = hw.constant 1 : i2
      // CHECK-NEXT:  %9 = comb.and %5, %1 : i1
      // CHECK-NEXT:  %c-2_i2 = hw.constant -2 : i2
      // CHECK-NEXT:  %c-2_i2_1 = hw.constant -2 : i2
      // CHECK-NEXT:  %10 = comb.icmp eq %0#2, %c-2_i2_1 : i2
      // CHECK-NEXT:  calyx.group @seq {
      // CHECK-NEXT:    calyx.assign %0#0 = %c1_i2_0, %8 ? : i2
      // CHECK-NEXT:    calyx.assign %0#1 = %true, %8 ? : i1
      // CHECK-NEXT:    calyx.assign %0#0 = %c-2_i2, %9 ? : i2
      // CHECK-NEXT:    calyx.assign %0#1 = %true, %9 ? : i1
      // CHECK-NEXT:    calyx.group_done %true, %10 ? : i1
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
