// RUN: circt-opt %s  | circt-opt | FileCheck %s

msft.module @Foo {} (%in0 : !esi.channel<i1>) -> (out: !esi.channel<i1>)

// CHECK-LABEL:  esi.pure_module @top {
// CHECK-NEXT:     %foo.out = msft.instance @foo @Foo(%foo.out)  : (!esi.channel<i1>) -> !esi.channel<i1>
esi.pure_module @top {
  %loopback = msft.instance @foo @Foo(%loopback) : (!esi.channel<i1>) -> (!esi.channel<i1>)
}
