// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=all}))' %s | FileCheck %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=all preserve-public-types=false}))' %s | FileCheck --check-prefix=NOT_PRESERVE_PUBLIC_TYPES %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=vec}))' %s | FileCheck --check-prefix=VEC %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=1d-vec}))' %s | FileCheck --check-prefix=1D_VEC %s

firrtl.circuit "TopLevel" {
  // CHECK-LABEL: firrtl.extmodule @External(in source_valid: !firrtl.uint<1>)
  // CHECK-LABEL: firrtl.module @TopLevel(in %source_valid: !firrtl.uint<1>, out %sink_valid: !firrtl.uint<1>)
  // NOT_PRESERVE_PUBLIC_TYPES-LABEL: firrtl.extmodule @External(in source_valid: !firrtl.uint<1>)
  // NOT_PRESERVE_PUBLIC_TYPES-LABEL: firrtl.module @TopLevel(in %source: !firrtl.bundle<valid: uint<1>>, out %sink: !firrtl.bundle<valid: uint<1>>)
  firrtl.extmodule @External(in source: !firrtl.bundle<valid: uint<1>>)
  firrtl.module @TopLevel(in %source: !firrtl.bundle<valid: uint<1>>,
                          out %sink: !firrtl.bundle<valid: uint<1>>) {
  }
  // CHECK: @Foo(in %a: !firrtl.bundle<a: vector<vector<uint<1>, 2>, 2>>)
  // VEC: @Foo(in %a_a: !firrtl.vector<vector<uint<1>, 2>, 2>)
  // 1D_VEC: @Foo(in %a_a_0: !firrtl.vector<uint<1>, 2>, in %a_a_1: !firrtl.vector<uint<1>, 2>)
  firrtl.module private @Foo(in %a: !firrtl.bundle<a: vector<vector<uint<1>, 2>, 2>>) {
  }
  // 1D_VEC: %a_0: !firrtl.uint<1>
  firrtl.module private @Bar(in %a: !firrtl.vector<uint<1>, 1>) {
  }
  // CHECK-LABEL: PublicModule
  // CHECK-NOT: firrtl.bundle
  // NOT_PRESERVE_PUBLIC_TYPES-LABEL: PublicModule
  // NOT_PRESERVE_PUBLIC_TYPES: firrtl.bundle
  firrtl.module public @PublicModule(in %source: !firrtl.bundle<valid: uint<1>>) {
  }
}
