// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @Casts
func.func @Casts(%arg0: !moore.bit) -> (!moore.bit, !moore.bit) {
  // CHECK-NOT: moore.conversion
  // CHECK-NOT: moore.bool_cast
  %0 = moore.conversion %arg0 : !moore.bit -> !moore.bit
  %1 = moore.bool_cast %arg0 : !moore.bit -> !moore.bit
  // CHECK: return %arg0, %arg0
  return %0, %1 : !moore.bit, !moore.bit
}
