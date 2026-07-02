// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @empty_uarray_cmp
// CHECK-SAME: (%arg0: !hw.array<0xi8>, %arg1: !hw.array<0xi8>) -> (i1, i1)
// CHECK-NOT: moore.uarray_cmp
// CHECK-DAG: %[[TRUE:.*]] = hw.constant true
// CHECK-DAG: %[[FALSE:.*]] = hw.constant false
// CHECK: return %[[TRUE]], %[[FALSE]] : i1, i1
func.func @empty_uarray_cmp(%a: !moore.uarray<0x!moore.i8>, %b: !moore.uarray<0x!moore.i8>) -> (!moore.i1, !moore.i1) {
  %eq = moore.uarray_cmp eq %a, %b : !moore.uarray<0x!moore.i8> -> !moore.i1
  %ne = moore.uarray_cmp ne %a, %b : !moore.uarray<0x!moore.i8> -> !moore.i1
  return %eq, %ne : !moore.i1, !moore.i1
}

// CHECK-LABEL: func.func @zero_bit_uarray_cmp
// CHECK-SAME: (%arg0: !hw.array<4xi0>, %arg1: !hw.array<4xi0>) -> (i1, i1)
// CHECK-NOT: moore.uarray_cmp
// CHECK-DAG: %[[TRUE:.*]] = hw.constant true
// CHECK-DAG: %[[FALSE:.*]] = hw.constant false
// CHECK: return %[[TRUE]], %[[FALSE]] : i1, i1
func.func @zero_bit_uarray_cmp(%a: !moore.uarray<4x!moore.i0>, %b: !moore.uarray<4x!moore.i0>) -> (!moore.i1, !moore.i1) {
  %eq = moore.uarray_cmp eq %a, %b : !moore.uarray<4x!moore.i0> -> !moore.i1
  %ne = moore.uarray_cmp ne %a, %b : !moore.uarray<4x!moore.i0> -> !moore.i1
  return %eq, %ne : !moore.i1, !moore.i1
}
