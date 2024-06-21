// RUN: circt-opt --mem2reg %s | FileCheck %s

// CHECK-LABEL: moore.module @LocalVar() {
moore.module @LocalVar() {
// CHECK: %x = moore.variable : <i32>
// CHECK: %y = moore.variable : <i32>
// CHECK: %z = moore.variable : <i32>
%x = moore.variable : <i32>
%y = moore.variable : <i32>
%z = moore.variable : <i32>
moore.procedure always_comb {
    // CHECK: %0 = moore.read %x : i32
    // CHECK: %1 = moore.constant 1 : i32
    // CHECK: %2 = moore.add %0, %1 : i32
    // CHECK: moore.blocking_assign %z, %2 : i32
    // CHECK: %3 = moore.constant 1 : i32
    // CHECK: %4 = moore.add %2, %3 : i32
    // CHECK: moore.blocking_assign %y, %4 : i32
    %a = moore.variable : <i32>
    %0 = moore.read %x : i32
    %1 = moore.constant 1 : i32
    %2 = moore.add %0, %1 : i32
    moore.blocking_assign %a, %2 : i32
    %3 = moore.read %a : i32
    moore.blocking_assign %z, %3 : i32
    %4 = moore.read %a : i32
    %5 = moore.constant 1 : i32
    %6 = moore.add %4, %5 : i32
    moore.blocking_assign %a, %6 : i32
    %7 = moore.read %a : i32
    moore.blocking_assign %y, %7 : i32
}
// CHECK: moore.output
moore.output
}
