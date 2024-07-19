// RUN: circt-opt --sroa %s | FileCheck %s

// CHECK-LABEL: moore.module @LocalVar
moore.module @LocalVar() {
  // CHECK: %x = moore.variable : <i32>
  // CHECK: %y = moore.variable : <i32>
  // CHECK: %z = moore.variable : <i32>
  %x = moore.variable : <i32>
  %y = moore.variable : <i32>
  %z = moore.variable : <i32>
  moore.procedure always_comb {
    // CHECK: %a = moore.variable : <i32>
    // CHECK: %b = moore.variable : <i32>
    // CHECK: %0 = moore.constant 1 : i32
    // CHECK: %1 = moore.conversion %0 : !moore.i32 -> !moore.i32
    // CHECK: moore.blocking_assign %a, %1 : i32
    // CHECK: %2 = moore.constant 4 : i32
    // CHECK: %3 = moore.conversion %2 : !moore.i32 -> !moore.i32
    // CHECK: moore.blocking_assign %b, %3 : i32
    // CHECK: %4 = moore.read %x
    // CHECK: %5 = moore.constant 1 : i32
    // CHECK: %6 = moore.add %4, %5 : i32
    // CHECK: moore.blocking_assign %a, %6 : i32
    // CHECK: %7 = moore.read %a
    // CHECK: moore.blocking_assign %y, %7 : i32
    // CHECK: %8 = moore.read %a
    // CHECK: %9 = moore.constant 1 : i32
    // CHECK: %10 = moore.add %8, %9 : i32
    // CHECK: moore.blocking_assign %a, %10 : i32
    // CHECK: %11 = moore.read %a
    // CHECK: moore.blocking_assign %z, %11 : i32
    %ii = moore.variable : <struct<{a: i32, b: i32}>>
    %0 = moore.struct_extract_ref %ii, "a" : <struct<{a: i32, b: i32}>> -> <i32>
    %1 = moore.constant 1 : i32
    %2 = moore.conversion %1 : !moore.i32 -> !moore.i32
    moore.blocking_assign %0, %2 : i32
    %3 = moore.struct_extract_ref %ii, "b" : <struct<{a: i32, b: i32}>> -> <i32>
    %4 = moore.constant 4 : i32
    %5 = moore.conversion %4 : !moore.i32 -> !moore.i32
    moore.blocking_assign %3, %5 : i32
    %6 = moore.struct_extract_ref %ii, "a" : <struct<{a: i32, b: i32}>> -> <i32>
    %7 = moore.read %x : <i32>
    %8 = moore.constant 1 : i32
    %9 = moore.add %7, %8 : i32
    moore.blocking_assign %6, %9 : i32
    %10 = moore.struct_extract %ii, "a" : <struct<{a: i32, b: i32}>> -> i32
    moore.blocking_assign %y, %10 : i32
    %11 = moore.struct_extract_ref %ii, "a" : <struct<{a: i32, b: i32}>> -> <i32>
    %12 = moore.struct_extract %ii, "a" : <struct<{a: i32, b: i32}>> -> i32
    %13 = moore.constant 1 : i32
    %14 = moore.add %12, %13 : i32
    moore.blocking_assign %11, %14 : i32
    %15 = moore.struct_extract %ii, "a" : <struct<{a: i32, b: i32}>> -> i32
    moore.blocking_assign %z, %15 : i32
  }
}


