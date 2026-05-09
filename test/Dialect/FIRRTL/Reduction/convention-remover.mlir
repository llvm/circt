// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include module-convention-remover --include extmodule-convention-remover | FileCheck %s

// Test removing convention attribute from regular module
firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.module @Foo
  // CHECK-SAME: () {
  // CHECK-NOT: attributes
  // CHECK-NOT: convention
  firrtl.module @Foo() attributes {convention = #firrtl<convention scalarized>} {
  }
}

// Test removing convention attribute from external module
firrtl.circuit "Bar" {
  // CHECK-LABEL: firrtl.extmodule @Bar
  // CHECK-SAME: ()
  // CHECK-NOT: attributes
  // CHECK-NOT: convention
  firrtl.extmodule @Bar() attributes {convention = #firrtl<convention scalarized>}
}
