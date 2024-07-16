// RUN: circt-opt --moore-merge-extractref %s | FileCheck %s

// CHECK-LABEL: moore.module @MergeExtractRef()
moore.module @MergeExtractRef() {
  // CHECK: %arr = moore.variable : <i32>
  // CHECK: %0 = moore.constant 16 : i32
  // CHECK: %1 = moore.extract_ref %arr from %0 : <i32>, i32 -> <i8>
  // CHECK: %2 = moore.constant 5 : i8
  // CHECK: moore.assign %1, %2 : i8
  %arr = moore.variable : <i32>
  %0 = moore.constant 16 : i32
  %1 = moore.extract_ref %arr from %0 : <i32>, i32 -> <i8>
  %2 = moore.constant 5 : i8
  moore.assign %1, %2 : i8

  moore.procedure always_comb {
  // CHECK:   %3 = moore.constant 8 : i32
  // CHECK:   %4 = moore.extract_ref %arr from %3 : <i32>, i32 -> <i8>
  // CHECK:   %5 = moore.constant 10 : i8
    %3 = moore.constant 8 : i32
    %4 = moore.extract_ref %arr from %3 : <i32>, i32 -> <i8>
    %5 = moore.constant 10 : i8

  // CHECK:   %6 = moore.concat %2, %5 : (!moore.i8, !moore.i8) -> i16
  // CHECK:   %7 = moore.constant 0 : i8
  // CHECK:   %8 = moore.read %arr : i32
  // CHECK:   %9 = moore.extract %8 from %7 : i32, i8 -> i8
  // CHECK:   %10 = moore.constant 24 : i8
  // CHECK:   %11 = moore.read %arr : i32
  // CHECK:   %12 = moore.extract %11 from %10 : i32, i8 -> i8
  // CHECK:   %13 = moore.concat %12, %6, %9 : (!moore.i8, !moore.i16, !moore.i8) -> i32
  // CHECK:   %arr_0 = moore.variable name "arr" %13 : <i32>

  // CHECK:   moore.blocking_assign %4, %5 : i8
    moore.blocking_assign %4, %5 : i8
  }
  // CHECK: moore.output
  moore.output
}

