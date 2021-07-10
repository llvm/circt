// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-go-insertion))' %s | FileCheck %s

calyx.program {
  calyx.component @A(%in: i8) -> (%out: i8, %flag: i1) {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main() -> () {
    %in, %out, %flag = calyx.cell "c0" @A : i8, i8, i1

    // CHECK:       %1 = calyx.undef : i1
    // CHECK-NEXT:  calyx.group @Group1 {
    // CHECK-NEXT:    %2 = calyx.group_go %1 : i1
    // CHECK-NEXT:    %3 = comb.and %0#2, %2 : i1
    // CHECK-NEXT:    calyx.assign %0#0 = %0#1, %2 ? : i8
    // CHECK-NEXT:    calyx.assign %0#0 = %0#1, %3 ? : i8
    // CHECK-NEXT:    calyx.group_done %0#2 : i1
    // CHECK-NEXT:  }
    calyx.wires {
      calyx.group @Group1 {
        calyx.assign %in = %out : i8
        calyx.assign %in = %out, %flag ? : i8
        calyx.group_done %flag : i1
      }
    }
    calyx.control {
      calyx.seq { calyx.enable @Group1 }
    }
  }
}
