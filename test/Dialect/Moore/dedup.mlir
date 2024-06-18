// RUN: circt-opt --moore-dedup %s | FileCheck %s

// CHECK-LABEL: moore.module @Foo()
moore.module @Foo() {
  %a = moore.net wire : <l4>
  %0 = moore.read %a : l4
  // CHECK: moore.instance "insA" @NestedA
  moore.instance "insA" @NestedA(a: %0: !moore.l4) -> ()
  %1 = moore.read %a : l4
  // CHECK: moore.instance "insB" @NestedA
  moore.instance "insB" @NestedA_0(a: %1: !moore.l4) -> ()
  moore.output
}
// CHECK: moore.module @NestedA
moore.module @NestedA(in %a : !moore.l4) {
  %a_0 = moore.net name "a" wire : <l4>
  moore.assign %a_0, %a : l4
  moore.output
}
// CHECK-NOT: moore.module @NestedA_0
moore.module @NestedA_0(in %a : !moore.l4) {
  %a_0 = moore.net name "a" wire : <l4>
  moore.assign %a_0, %a : l4
  moore.output
}
// CHECK-NOT: moore.module @NestedA_1
moore.module @NestedA_1(in %a : !moore.l4) {
  %a_0 = moore.net name "a" wire : <l4>
  moore.assign %a_0, %a : l4
  moore.output
}
