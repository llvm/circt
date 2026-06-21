// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @zero_packed_array_slice
// CHECK-NOT: hw.array_concat
// CHECK: %[[ZERO:.*]] = hw.constant 0 : i0
// CHECK-NEXT: %[[ARR:.*]] = hw.bitcast %[[ZERO]] : (i0) -> !hw.array<0xi8>
// CHECK-NEXT: return %[[ARR]] : !hw.array<0xi8>
func.func @zero_packed_array_slice(%arg0: !moore.array<0 x !moore.i8>) -> !moore.array<0 x !moore.i8> {
  %0 = moore.extract %arg0 from 0 : !moore.array<0 x !moore.i8> -> !moore.array<0 x !moore.i8>
  return %0 : !moore.array<0 x !moore.i8>
}

// CHECK-LABEL: func.func @zero_unpacked_array_slice
// CHECK-NOT: hw.array_concat
// CHECK: %[[ZERO:.*]] = hw.constant 0 : i0
// CHECK-NEXT: %[[ARR:.*]] = hw.bitcast %[[ZERO]] : (i0) -> !hw.array<0xi8>
// CHECK-NEXT: return %[[ARR]] : !hw.array<0xi8>
func.func @zero_unpacked_array_slice(%arg0: !moore.uarray<0 x !moore.i8>) -> !moore.uarray<0 x !moore.i8> {
  %0 = moore.extract %arg0 from 0 : !moore.uarray<0 x !moore.i8> -> !moore.uarray<0 x !moore.i8>
  return %0 : !moore.uarray<0 x !moore.i8>
}
