// RUN: circt-opt --kanagawa-argify-blocks %s | FileCheck %s

// CHECK-LABEL: kanagawa.class sym @Argify {
// CHECK-NEXT:   kanagawa.method @foo() -> () {
// CHECK-NEXT:     %c32_i32 = hw.constant 32 : i32
// CHECK-NEXT:     %0:2 = kanagawa.sblock.isolated (%arg0 : i32 = %c32_i32) -> (i32, i32) {
// CHECK-NEXT:       %c31_i32 = hw.constant 31 : i32
// CHECK-NEXT:       %1 = arith.addi %arg0, %c31_i32 : i32
// CHECK-NEXT:       kanagawa.sblock.return %1, %arg0 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     kanagawa.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

kanagawa.design @foo {
kanagawa.class sym @Argify {

  kanagawa.method @foo()  {
    %c32 = hw.constant 32 : i32
    %0:2 = kanagawa.sblock() -> (i32, i32) {
      %c31 = hw.constant 31 : i32
      %res = arith.addi %c31, %c32 : i32
      kanagawa.sblock.return %res, %c32 : i32, i32
    }
    kanagawa.return
  }
}
}
