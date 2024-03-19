// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @print
hw.module @print(in %clock : !seq.clock, in %cond : i1, in %val : i32) {
  // CHECK:      [[HW_CLK:%.+]] = seq.from_clock %clock
  // CHECK-NEXT: sv.ifdef  @SYNTHESIS {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.always posedge [[HW_CLK]] {
  // CHECK-NEXT:     sv.if %cond {
  // CHECK-NEXT:       %c-2147483646_i32 = hw.constant -2147483646 : i32
  // CHECK-NEXT:       sv.fwrite %c-2147483646_i32, "print %d\0A"(%val) : i32
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sim.print %clock, %cond, "print %d\n" (%val) : i32
}
