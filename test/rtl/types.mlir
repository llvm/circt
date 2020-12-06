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
}
