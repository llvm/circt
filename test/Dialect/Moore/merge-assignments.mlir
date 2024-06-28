// RUN: circt-opt --moore-merge-assignments %s | FileCheck %s

// CHECK-LABEL: moore.module @Foo()
moore.module @Foo() {
  // CHECK: %a = moore.variable : <i32>
  // CHECK: %a_0 = moore.assigned_variable name "a" %0 : <i32>
  %a = moore.variable : <i32>

  // CHECK: %l = moore.net wire : <l1>
  %l = moore.net wire : <l1>

  // CHECK: %0 = moore.constant 32 : i32
  %0 = moore.constant 32 : i32
  moore.assign %a, %0 : i32
  %1 = moore.constant true : i1
  %2 = moore.conversion %1 : !moore.i1 -> !moore.l1
  moore.assign %l, %2 : l1
  moore.output
}

