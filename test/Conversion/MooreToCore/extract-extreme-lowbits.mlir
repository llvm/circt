// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @ExtractIntLowI32Max
func.func @ExtractIntLowI32Max(%arg0: !moore.i4) -> !moore.i2 {
  // CHECK: hw.constant 0 : i2
  %0 = moore.extract %arg0 from 2147483647 : !moore.i4 -> !moore.i2
  return %0 : !moore.i2
}

// CHECK-LABEL: func.func @ExtractIntLowI32Min
func.func @ExtractIntLowI32Min(%arg0: !moore.i4) -> !moore.i2 {
  // CHECK: hw.constant 0 : i2
  %0 = moore.extract %arg0 from -2147483648 : !moore.i4 -> !moore.i2
  return %0 : !moore.i2
}
