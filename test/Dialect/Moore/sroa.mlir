// RUN: circt-opt --sroa %s | FileCheck %s

// CHECK-LABEL: moore.module @LocalVar() {
moore.module @LocalVar() {
// CHECK: %x = moore.variable : <i32>
// CHECK: %y = moore.variable : <i32>
// CHECK: %z = moore.variable : <i32>
%x = moore.variable : <i32>
%y = moore.variable : <i32>
%z = moore.variable : <i32>
moore.procedure always_comb {
    // CHECK: %0 = moore.constant 0 : i32
    // CHECK: %a = moore.variable %0 : <i32>
    // CHECK: %1 = moore.constant 0 : i32
    // CHECK: %b = moore.variable %1 : <i32>
    // CHECK: %2 = moore.struct_create %0, %1 : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
    // CHECK: %3 = moore.constant 1 : i32
    // CHECK: %4 = moore.conversion %3 : !moore.i32 -> !moore.i32
    // CHECK: moore.blocking_assign %a, %4 : i32
    // CHECK: %5 = moore.constant 4 : i32
    // CHECK: %6 = moore.conversion %5 : !moore.i32 -> !moore.i32
    // CHECK: moore.blocking_assign %b, %6 : i32
    // CHECK: %7 = moore.read %x : i32
    // CHECK: %8 = moore.constant 1 : i3
    // CHECK: %9 = moore.add %7, %8 : i32
    // CHECK: %10 = moore.read %a : i32
    // CHECK: moore.blocking_assign %y, %10 : i32
    // CHECK: %11 = moore.read %a : i32
    // CHECK: %12 = moore.constant 1 : i32
    // CHECK: %13 = moore.add %11, %12 : i32
    // CHECK: moore.blocking_assign %a, %13 : i32
    // CHECK: %14 = moore.read %a : i32
    // CHECK: moore.blocking_assign %z, %14 : i32
   %ii = moore.variable : <struct<{a: i32, b: i32}>>
    %0 = moore.struct_extract_ref %ii, "a" : <struct<{a: i32, b: i32}>> -> <i32>
    %1 = moore.constant 1 : i32
    %2 = moore.conversion %1 : !moore.i32 -> !moore.i32
    moore.struct_inject %ii, "a", %2 : <struct<{a: i32, b: i32}>> i32
    %3 = moore.struct_extract_ref %ii, "b" : <struct<{a: i32, b: i32}>> -> <i32>
    %4 = moore.constant 4 : i32
    %5 = moore.conversion %4 : !moore.i32 -> !moore.i32
    moore.struct_inject %ii, "b", %5 : <struct<{a: i32, b: i32}>> i32
    %6 = moore.struct_extract_ref %ii, "a" : <struct<{a: i32, b: i32}>> -> <i32>
    %7 = moore.read %x : i32
    %8 = moore.constant 1 : i32
    %9 = moore.add %7, %8 : i32
    moore.struct_inject %ii, "a", %9 : <struct<{a: i32, b: i32}>> i32
    %10 = moore.struct_extract %ii, "a" : <struct<{a: i32, b: i32}>> -> i32
    moore.blocking_assign %y, %10 : i32
    %11 = moore.struct_extract_ref %ii, "a" : <struct<{a: i32, b: i32}>> -> <i32>
    %12 = moore.struct_extract %ii, "a" : <struct<{a: i32, b: i32}>> -> i32
    %13 = moore.constant 1 : i32
    %14 = moore.add %12, %13 : i32
    moore.struct_inject %ii, "a", %14 : <struct<{a: i32, b: i32}>> i32
    %15 = moore.struct_extract %ii, "a" : <struct<{a: i32, b: i32}>> -> i32
    moore.blocking_assign %z, %15 : i32
}
// CHECK: moore.output
moore.output
}


