// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-go-insertion))' %s | FileCheck %s

calyx.program {
  calyx.component @A(%in: i8) -> (%out: i8, %flag: i1) {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main() -> () {
    // CHECK: %0 = calyx.undef : i1
    %in, %out, %flag = calyx.cell "c0" @A : i8, i8, i1

    calyx.wires {
      // CHECK-LABEL: calyx.group @Group1
      // CHECK-NEXT:  %2 = calyx.group_go %0 : i1
      // CHECK-NEXT:  %3 = comb.and %1#2, %2 : i1
      // CHECK-NEXT:  calyx.assign %1#0 = %1#1, %2 ? : i8
      // CHECK-NEXT:  calyx.assign %1#0 = %1#1, %3 ? : i8
      // CHECK-NEXT:  %4 = calyx.group_done %1#2 : i1
      // CHECK-NEXT:  }
      calyx.group @Group1 {
        calyx.assign %in = %out : i8
        calyx.assign %in = %out, %flag ? : i8
        %done = calyx.group_done %flag : i1
      }
    }
    calyx.control {
      calyx.seq { calyx.enable @Group1 }
    }
  }
}
