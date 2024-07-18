// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @Casts
func.func @Casts(%arg0: !moore.i1) -> (!moore.i1, !moore.i1) {
  // CHECK-NOT: moore.conversion
  // CHECK-NOT: moore.bool_cast
  %0 = moore.conversion %arg0 : !moore.i1 -> !moore.i1
  %1 = moore.bool_cast %arg0 : !moore.i1 -> !moore.i1
  // CHECK: return %arg0, %arg0
  return %0, %1 : !moore.i1, !moore.i1
}

// CHECK-LABEL: moore.module @SingleAssign
moore.module @SingleAssign() {
  // CHECK-NOT: moore.variable
  // CHECK: %a = moore.assigned_variable %0 : <i32>
  %a = moore.variable : <i32>
  // CHECK: %0 = moore.constant 32 : i32
  %0 = moore.constant 32 : i32
  // CHECK: moore.assign %a, %0 : i32
  moore.assign %a, %0 : i32
  moore.output
}

// CHECK-LABEL: moore.module @MultiAssign
moore.module @MultiAssign() {
  // CHECK-NOT: moore.assigned_variable
  // CHECK: %a = moore.variable : <i32>
  %a = moore.variable : <i32>
  // CHECK: %0 = moore.constant 32 : i32
  %0 = moore.constant 32 : i32
  // CHECK: moore.assign %a, %0 : i32
  moore.assign %a, %0 : i32
  // CHECK: %1 = moore.constant 64 : i32
  %1 = moore.constant 64 : i32
  // CHECK: moore.assign %a, %1 : i32
  moore.assign %a, %1 : i32
  moore.output
}

// CHECK-LABEL: moore.module @structAssign
moore.module @structAssign(out a : !moore.ref<struct<{a: i32, b: i32}>>) {
  %x = moore.variable : <i32>
  %y = moore.variable : <i32>
  %z = moore.variable : <i32>
  // CHECK: %0 = moore.constant 4 : i32
  // CHECK: %1 = moore.read %x
  // CHECK: %2 = moore.constant 1 : i32
  // CHECK: %3 = moore.add %1, %2 : i32
  // CHECK: %4 = moore.struct_create %3, %0 : !moore.i32, !moore.i32 -> <struct<{a: i32, b: i32}>>
  %ii = moore.variable : <struct<{a: i32, b: i32}>>
  %0 = moore.constant 4 : i32
  %1 = moore.conversion %0 : !moore.i32 -> !moore.i32
  %2 = moore.struct_inject %ii, "b", %1 : !moore.ref<struct<{a: i32, b: i32}>>
  %3 = moore.read %x : <i32>
  %4 = moore.constant 1 : i32
  %5 = moore.add %3, %4 : i32
  %6 = moore.struct_inject %2, "a", %5 : !moore.ref<struct<{a: i32, b: i32}>>
  %7 = moore.struct_extract %6, "a" : <struct<{a: i32, b: i32}>> -> i32
  // CHECK: moore.assign %y, %3 : i32
  moore.assign %y, %7 : i32
  %8 = moore.struct_extract %6, "a" : <struct<{a: i32, b: i32}>> -> i32
  // CHECK: moore.assign %z, %3 : i32
  moore.assign %z, %8 : i32
  // CHECK: moore.output %4 : !moore.ref<struct<{a: i32, b: i32}>>
  moore.output %6 : !moore.ref<struct<{a: i32, b: i32}>>
}

// CHECK-LABEL: moore.module @structInjectFold
moore.module @structInjectFold(out a : !moore.ref<struct<{a: i32, b: i32}>>) {
  %x = moore.variable : <i32>
  %y = moore.variable : <i32>
  %z = moore.variable : <i32>
  %ii = moore.variable : <struct<{a: i32, b: i32}>>
  // CHECK: %0 = moore.read %x
  // CHECK: %1 = moore.constant 1 : i32
  // CHECK: %2 = moore.add %0, %1 : i32
  // CHECK: %3 = moore.struct_inject %ii, "a", %2 : !moore.ref<struct<{a: i32, b: i32}>>
  %0 = moore.constant 4 : i32
  %1 = moore.conversion %0 : !moore.i32 -> !moore.i32
  %2 = moore.struct_inject %ii, "a", %1 : !moore.ref<struct<{a: i32, b: i32}>>
  %3 = moore.read %x : <i32>
  %4 = moore.constant 1 : i32
  %5 = moore.add %3, %4 : i32
  %6 = moore.struct_inject %2, "a", %5 : !moore.ref<struct<{a: i32, b: i32}>>
  %7 = moore.struct_extract %6, "a" : <struct<{a: i32, b: i32}>> -> i32
  // CHECK: moore.assign %y, %2 : i32
  moore.assign %y, %7 : i32
  %8 = moore.struct_extract %6, "a" : <struct<{a: i32, b: i32}>> -> i32
  // CHECK: moore.assign %z, %2 : i32
  moore.assign %z, %8 : i32
  // CHECK: moore.output %3 : !moore.ref<struct<{a: i32, b: i32}>>
  moore.output %6 : !moore.ref<struct<{a: i32, b: i32}>>
}

// CHECK-LABEL: moore.module @structCreateFold
moore.module @structCreateFold(in %a : !moore.i1, out b : !moore.i1) {
    %0 = moore.struct_create %a : !moore.i1 -> <struct<{a: i1}>>
    %1 = moore.struct_extract %0, "a" : <struct<{a: i1}>> -> i1
    // CHECK:  moore.output %a : !moore.i1
    moore.output %1 : !moore.i1
  }
