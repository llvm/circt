// RUN: circt-opt --hw-flatten-io="recursive=true" %s | FileCheck %s

// Ensure that non-struct-using modules pass cleanly through the pass.

// CHECK-LABEL: hw.module @level0(%arg0: i32) -> (out0: i32) {
// CHECK-NEXT:    hw.output %arg0 : i32
// CHECK-NEXT:  }
hw.module @level0(%arg0 : i32) -> (out0 : i32) {
    hw.output %arg0: i32
}

// CHECK-LABEL: hw.module @level1(%arg0: i32, %in.a: i1, %in.b: i2, %arg1: i32) -> (out0: i32, out.a: i1, out.b: i2, out1: i32) {
// CHECK-NEXT:    %0 = hw.struct_create (%in.a, %in.b) : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    %1:2 = hw.struct_explode %0 : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    hw.output %arg0, %1#0, %1#1, %arg1 : i32, i1, i2, i32
// CHECK-NEXT:  }
!Struct1 = !hw.struct<a: i1, b: i2>
hw.module @level1(%arg0 : i32, %in : !Struct1, %arg1: i32) -> (out0 : i32,out: !Struct1, out1: i32) {
    hw.output %arg0, %in, %arg1 : i32, !Struct1, i32
}

// CHECK-LABEL: hw.module @level2(%in.aa.a: i1, %in.aa.b: i2, %in.bb.a: i1, %in.bb.b: i2) -> (out.aa.a: i1, out.aa.b: i2, out.bb.a: i1, out.bb.b: i2) {
// CHECK-NEXT:    %0 = hw.struct_create (%in.aa.a, %in.aa.b) : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    %1 = hw.struct_create (%in.bb.a, %in.bb.b) : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    %2 = hw.struct_create (%0, %1) : !hw.struct<aa: !hw.struct<a: i1, b: i2>, bb: !hw.struct<a: i1, b: i2>>
// CHECK-NEXT:    %3:2 = hw.struct_explode %2 : !hw.struct<aa: !hw.struct<a: i1, b: i2>, bb: !hw.struct<a: i1, b: i2>>
// CHECK-NEXT:    %4:2 = hw.struct_explode %3#0 : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    %5:2 = hw.struct_explode %3#1 : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    hw.output %4#0, %4#1, %5#0, %5#1 : i1, i2, i1, i2
// CHECK-NEXT:  }
!Struct2 = !hw.struct<aa: !Struct1, bb: !Struct1>
hw.module @level2(%in : !Struct2) -> (out: !Struct2) {
    hw.output %in : !Struct2
}

// CHECK-LABEL: hw.module.extern @level1_extern(%arg0: i32, %in.a: i1, %in.b: i2, %arg1: i32) -> (out0: i32, out.a: i1, out.b: i2, out1: i32)
hw.module.extern @level1_extern(%arg0 : i32, %in : !Struct1, %arg1: i32) -> (out0 : i32,out: !Struct1, out1: i32)
