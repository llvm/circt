// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @negative_low_array_slice
// CHECK-SAME: (%arg0: !hw.array<4xi8>) -> !hw.array<2xi8>
// CHECK: %[[ZERO:.*]] = hw.constant 0 : i16
// CHECK-NEXT: %[[PADDING:.*]] = hw.bitcast %[[ZERO]] : (i16) -> !hw.array<2xi8>
// CHECK-NEXT: return %[[PADDING]] : !hw.array<2xi8>
func.func @negative_low_array_slice(%arg0: !moore.array<4 x !moore.i8>) -> !moore.array<2 x !moore.i8> {
  %0 = moore.extract %arg0 from -3 : !moore.array<4 x !moore.i8> -> !moore.array<2 x !moore.i8>
  return %0 : !moore.array<2 x !moore.i8>
}

// CHECK-LABEL: func.func @negative_low_i32_min_array_slice
// CHECK-SAME: (%arg0: !hw.array<4xi8>) -> !hw.array<2xi8>
// CHECK: %[[ZERO:.*]] = hw.constant 0 : i16
// CHECK-NEXT: %[[PADDING:.*]] = hw.bitcast %[[ZERO]] : (i16) -> !hw.array<2xi8>
// CHECK-NEXT: return %[[PADDING]] : !hw.array<2xi8>
func.func @negative_low_i32_min_array_slice(%arg0: !moore.array<4 x !moore.i8>) -> !moore.array<2 x !moore.i8> {
  %0 = moore.extract %arg0 from -2147483648 : !moore.array<4 x !moore.i8> -> !moore.array<2 x !moore.i8>
  return %0 : !moore.array<2 x !moore.i8>
}

// CHECK-LABEL: func.func @negative_low_i32_min_uarray_slice
// CHECK-SAME: (%arg0: !hw.array<4xi8>) -> !hw.array<2xi8>
// CHECK: %[[ZERO:.*]] = hw.constant 0 : i16
// CHECK-NEXT: %[[PADDING:.*]] = hw.bitcast %[[ZERO]] : (i16) -> !hw.array<2xi8>
// CHECK-NEXT: return %[[PADDING]] : !hw.array<2xi8>
func.func @negative_low_i32_min_uarray_slice(%arg0: !moore.uarray<4 x !moore.i8>) -> !moore.uarray<2 x !moore.i8> {
  %0 = moore.extract %arg0 from -2147483648 : !moore.uarray<4 x !moore.i8> -> !moore.uarray<2 x !moore.i8>
  return %0 : !moore.uarray<2 x !moore.i8>
}

// CHECK-LABEL: func.func @negative_low_partial_array_slice
// CHECK-SAME: (%arg0: !hw.array<4xi8>) -> !hw.array<3xi8>
// CHECK: %[[ZERO:.*]] = hw.constant 0 : i8
// CHECK-NEXT: %[[PADDING:.*]] = hw.bitcast %[[ZERO]] : (i8) -> !hw.array<1xi8>
// CHECK-NEXT: %[[LOW:.*]] = hw.constant 0 : i2
// CHECK-NEXT: %[[SLICE:.*]] = hw.array_slice %arg0[%[[LOW]]] : (!hw.array<4xi8>) -> !hw.array<2xi8>
// CHECK-NEXT: %[[RESULT:.*]] = hw.array_concat %[[PADDING]], %[[SLICE]] : !hw.array<1xi8>, !hw.array<2xi8>
// CHECK-NEXT: return %[[RESULT]] : !hw.array<3xi8>
func.func @negative_low_partial_array_slice(%arg0: !moore.array<4 x !moore.i8>) -> !moore.array<3 x !moore.i8> {
  %0 = moore.extract %arg0 from -1 : !moore.array<4 x !moore.i8> -> !moore.array<3 x !moore.i8>
  return %0 : !moore.array<3 x !moore.i8>
}
