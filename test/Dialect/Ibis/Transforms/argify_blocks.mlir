// RUN: circt-opt --ibis-argify-blocks %s | FileCheck %s

// CHECK-LABEL: ibis.class @Argify {
// CHECK-NEXT:   %this = ibis.this <@Argify> 
// CHECK-NEXT:   ibis.method @foo() -> () {
// CHECK-NEXT:     %c32_i32 = hw.constant 32 : i32
// CHECK-NEXT:     %0:2 = ibis.sblock.isolated (%arg0 : i32 = %c32_i32) -> (i32, i32) {
// CHECK-NEXT:       %c31_i32 = hw.constant 31 : i32
// CHECK-NEXT:       %1 = arith.addi %arg0, %c31_i32 : i32
// CHECK-NEXT:       ibis.sblock.return %1, %arg0 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     ibis.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

ibis.design @foo {
ibis.class @Argify {
  %this = ibis.this <@Argify>

  ibis.method @foo()  {
    %c32 = hw.constant 32 : i32
    %0:2 = ibis.sblock() -> (i32, i32) {
      %c31 = hw.constant 31 : i32
      %res = arith.addi %c31, %c32 : i32
      ibis.sblock.return %res, %c32 : i32, i32
    }
    ibis.return
  }
}
}
