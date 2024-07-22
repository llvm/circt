// RUN: circt-opt %s -canonicalize='top-down=true region-simplify=aggressive' | FileCheck %s

// CHECK-LABEL: @sigExtractOp
func.func @sigExtractOp(%arg0 : !hw.inout<i32>, %arg1: i5) -> (!hw.inout<i32>, !hw.inout<i32>) {
  %zero = hw.constant 0 : i5

  // CHECK: %[[EXT:.*]] = llhd.sig.extract %arg0 from %arg1 : (!hw.inout<i32>) -> !hw.inout<i32>
  %0 = llhd.sig.extract %arg0 from %arg1 : (!hw.inout<i32>) -> !hw.inout<i32>

  %1 = llhd.sig.extract %arg0 from %zero : (!hw.inout<i32>) -> !hw.inout<i32>

  // CHECK-NEXT: return %[[EXT]], %arg0 : !hw.inout<i32>, !hw.inout<i32>
  return %0, %1 : !hw.inout<i32>, !hw.inout<i32>
}

// CHECK-LABEL: @sigArraySlice
func.func @sigArraySliceOp(%arg0: !hw.inout<array<30xi32>>, %arg1: i5)
    -> (!hw.inout<array<30xi32>>, !hw.inout<array<30xi32>>, !hw.inout<array<20xi32>>, !hw.inout<array<3xi32>>) {
  %zero = hw.constant 0 : i5

  // CHECK-NEXT: %c-13_i5 = hw.constant -13 : i5
  // CHECK-NEXT: hw.constant
  %a = hw.constant 3 : i5
  %b = hw.constant 16 : i5

  // CHECK: %[[EXT:.*]] = llhd.sig.array_slice %arg0 at %arg1 : (!hw.inout<array<30xi32>>) -> !hw.inout<array<30xi32>>
  %ext = llhd.sig.array_slice %arg0 at %arg1 : (!hw.inout<array<30xi32>>) -> !hw.inout<array<30xi32>>

  %identity = llhd.sig.array_slice %arg0 at %zero : (!hw.inout<array<30xi32>>) -> !hw.inout<array<30xi32>>

  // CHECK-NEXT: %[[RES1:.*]] = llhd.sig.array_slice
  // CHECK-NEXT: %[[RES2:.*]] = llhd.sig.array_slice %arg0 at %c-13_i5 : (!hw.inout<array<30xi32>>) -> !hw.inout<array<3xi32>>
  %1 = llhd.sig.array_slice %arg0 at %a : (!hw.inout<array<30xi32>>) -> !hw.inout<array<20xi32>>
  %2 = llhd.sig.array_slice %1 at %b : (!hw.inout<array<20xi32>>) -> !hw.inout<array<3xi32>>

  // CHECK-NEXT: return %[[EXT]], %arg0, %[[RES1]], %[[RES2]] : !hw.inout<array<30xi32>>, !hw.inout<array<30xi32>>, !hw.inout<array<20xi32>>, !hw.inout<array<3xi32>>
  return %ext, %identity, %1, %2 : !hw.inout<array<30xi32>>, !hw.inout<array<30xi32>>, !hw.inout<array<20xi32>>, !hw.inout<array<3xi32>>
}
