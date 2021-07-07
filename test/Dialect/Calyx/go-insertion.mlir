// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-go-insertion))' %s | FileCheck %s

calyx.program {
  calyx.component @A(%in: i8) -> (%out1: i8, %flag: i1) {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main() -> () {
    %in1, %out1, %flag = calyx.cell "c0" @A : i8, i8, i1

    calyx.wires {
      // CHECK-LABEL: calyx.group @Group1
      // CHECK-NEXT:  %1 = calyx.go %false : i1
      // CHECK-NEXT:  %2 = comb.and %0#2, %1 : i1
      // CHECK-NEXT:  calyx.assign %0#0 = %0#1, %1 ? : i8
      // CHECK-NEXT:  calyx.assign %0#0 = %0#1, %2 ? : i8
      // CHECK-NEXT:  calyx.done %0#2 : i1
      // CHECK-NEXT:  }
      calyx.group @Group1 {
        calyx.assign %in1 = %out1 : i8
        calyx.assign %in1 = %out1, %flag ? : i8
        calyx.done %flag : i1
      }
    }
    calyx.control {}
  }
}
