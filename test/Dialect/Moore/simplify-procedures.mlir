// RUN: circt-opt --moore-simplify-procedures %s | FileCheck %s

// CHECK-LABEL: moore.module @Foo()
moore.module @Foo() {
  %a = moore.variable : <i32>
  %u = moore.variable : <i32>
  %x = moore.variable : <i32>
  %y = moore.variable : <i32>
  %z = moore.variable : <i32>
  moore.procedure always_comb {
    // CHECK: %0 = moore.read %a : i32
    // CHECK: %local_a = moore.variable %0 : <i32>
    // CHECK: %1 = moore.constant 1 : i32
    %0 = moore.constant 1 : i32
    // CHECK: moore.blocking_assign %local_a, %1 : i32
    // CHECK: %2 = moore.read %local_a : i32
    // CHECK: moore.blocking_assign %a, %2 : i32
    moore.blocking_assign %a, %0 : i32
    // CHECK: %3 = moore.read %local_a : i32
    %1 = moore.read %a : i32
    // CHECK: moore.blocking_assign %x, %3 : i32
    moore.blocking_assign %x, %1 : i32
    // CHECK: %4 = moore.read %local_a : i32
    %2 = moore.read %a : i32
    // CHECK: %5 = moore.constant 1 : i32
    %3 = moore.constant 1 : i32
    // CHECK: %6 = moore.add %4, %5 : i32
    %4 = moore.add %2, %3 : i32
    // CHECK: moore.blocking_assign %local_a, %6 : i32
    // CHECK: %7 = moore.read %local_a : i32
    // CHECK: moore.blocking_assign %a, %7 : i32
    moore.blocking_assign %a, %4 : i32
    // CHECK: %8 = moore.read %local_a : i32
    %5 = moore.read %a : i32
    // CHECK: moore.blocking_assign %y, %8 : i32
    moore.blocking_assign %y, %5 : i32
    // CHECK: %9 = moore.read %local_a : i32
    %6 = moore.read %a : i32
    // CHECK: %10 = moore.constant 1 : i32
    %7 = moore.constant 1 : i32
    // CHECK: %11 = moore.add %9, %10 : i32
    %8 = moore.add %6, %7 : i32
    // CHECK: moore.blocking_assign %local_a, %11 : i32
    // CHECK: %12 = moore.read %local_a : i32
    // CHECK: moore.blocking_assign %a, %12 : i32
    moore.blocking_assign %a, %8 : i32
    // CHECK: %13 = moore.read %local_a : i32
    %9 = moore.read %a : i32
    // CHECK: moore.blocking_assign %z, %13 : i32
    moore.blocking_assign %z, %9 : i32
  }

  moore.procedure always_comb {
    //CHECK: %0 = moore.read %a : i32
    %0 = moore.read %a : i32
    //CHECK: %local_a = moore.variable %0 : <i32>
    //CHECK: %1 = moore.read %local_a : i32
    //CHECK: moore.blocking_assign %u, %1 : i32
    moore.blocking_assign %u, %0 : i32

  }
  // CHECK: moore.output
  moore.output
}
