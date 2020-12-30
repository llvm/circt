// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: module
module {
  // CHECK-LABEL: func @i20x5array(%{{.*}}: !rtl.array<5xi20>)
  func @i20x5array(%A: !rtl.array<5 x i20>) {
    return
  }

  // CHECK-LABEL: func @inoutType(%arg0: !rtl.inout<i42>) {
  func @inoutType(%arg0: !rtl.inout<i42>) {
    return
  }

  // CHECK-LABEL: nestedType
  func @nestedType(
    // CHECK: %arg0: !rtl.inout<array<42xi8>>,
    %arg0: !rtl.inout<!rtl.array<42xi8>>,
     // CHECK: %arg1: !rtl.inout<array<42xi8>>,
    %arg1: !rtl.inout<array<42xi8>>,
    //CHECK: %arg2: !rtl.inout<array<2xarray<42xi8>>>
    %arg2: !rtl.inout<array<2xarray<42xi8>>>) {
    return
  }

}
