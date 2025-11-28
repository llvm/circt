// RUN: circt-opt %s | FileCheck %s

hw.type_scope @__hw_typedecls {
  hw.typedecl @foo : i1
}

// https://github.com/llvm/circt/issues/5772
// CHECK-LABEL: @Issue5772
hw.module @Issue5772(in %arg0: !hw.typealias<@__hw_typedecls::@foo,i1>) {
  // CHECK: comb.concat %arg0 : !hw.typealias<@__hw_typedecls::@foo, i1>
  %0 = comb.concat %arg0 : !hw.typealias<@__hw_typedecls::@foo,i1>
}

// CHECK-LABEL: @NullaryConcat
hw.module @NullaryConcat() {
  // CHECK: comb.concat :
  comb.concat :
}
