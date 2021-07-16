// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-go-insertion))' %s | FileCheck %s

calyx.program {
  calyx.component @A(%in: i8) -> (%out: i8, %flag: i1) {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main() -> () {
    %in, %out, %flag = calyx.cell "c0" @A : i8, i8, i1
    calyx.wires {
      // CHECK-LABEL: calyx.group @Group1 {
      // CHECK-NEXT:    %Group1.go = calyx.group_go %0 : i1
      // CHECK-NEXT:    %1 = comb.and %c0.flag, %Group1.go : i1
      // CHECK-NEXT:    calyx.assign %c0.in = %c0.out, %Group1.go ? : i8
      // CHECK-NEXT:    calyx.assign %c0.in = %c0.out, %1 ? : i8
      // CHECK-NEXT:    calyx.group_done %c0.flag : i1
      // CHECK-NEXT:  }
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
