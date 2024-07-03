// RUN: circt-opt --sroa %s | FileCheck %s

// CHECK-LABEL: moore.module @LocalVar() {
moore.module @LocalVar() {
// CHECK: %x = moore.variable : <i32>
// CHECK: %y = moore.variable : <i32>
// CHECK: %a = moore.variable : <i32>
// CHECK: %b = moore.variable : <i32>
// CHECK: %0 = moore.struct_create %a, %b : !moore.ref<i32>, !moore.ref<i32> -> <struct<{a: i32, b: i32}>>
%x = moore.variable : <i32>
%y = moore.variable : <i32>
%z = moore.variable : <struct<{a: i32, b: i32}>>
moore.procedure always_comb {
    // CHECK: %1 = moore.read %a : i32
    // CHECK: moore.blocking_assign %x, %1 : i32
    // CHECK: %2 = moore.read %y : i32
    // CHECK: moore.blocking_assign %a, %2 : i32
    %0 = moore.struct_extract %z, "a" : <struct<{a: i32, b: i32}>> -> i32
    moore.blocking_assign %x, %0 : i32
    %1 = moore.struct_extract_ref %z, "a" : <struct<{a: i32, b: i32}>> -> <i32>
    %2 = moore.read %y : i32
    moore.struct_inject %z, "a", %2 : <struct<{a: i32, b: i32}>> i32
}
// CHECK: moore.output
moore.output
}
