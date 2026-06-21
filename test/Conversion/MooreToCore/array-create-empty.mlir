// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @empty_packed_array_create
// CHECK-NOT: hw.array_create
// CHECK: %[[ZERO0:.+]] = hw.constant 0 : i0
// CHECK: %[[ARRAY0:.+]] = hw.bitcast %[[ZERO0]] : (i0) -> !hw.array<0xi8>
// CHECK: return %[[ARRAY0]] : !hw.array<0xi8>
// CHECK-NOT: hw.array_create
func.func @empty_packed_array_create() -> !moore.array<0 x !moore.i8> {
  %0 = "moore.array_create"() : () -> !moore.array<0 x !moore.i8>
  return %0 : !moore.array<0 x !moore.i8>
}

// CHECK-LABEL: func.func @empty_unpacked_array_create
// CHECK-NOT: hw.array_create
// CHECK: %[[ZERO1:.+]] = hw.constant 0 : i0
// CHECK: %[[ARRAY1:.+]] = hw.bitcast %[[ZERO1]] : (i0) -> !hw.array<0xi8>
// CHECK: return %[[ARRAY1]] : !hw.array<0xi8>
// CHECK-NOT: hw.array_create
func.func @empty_unpacked_array_create() -> !moore.uarray<0 x !moore.i8> {
  %0 = "moore.array_create"() : () -> !moore.uarray<0 x !moore.i8>
  return %0 : !moore.uarray<0 x !moore.i8>
}
