// RUN: circt-opt %s  | circt-opt | FileCheck %s

msft.module @Foo {} (%clk: i1, %in0 : !esi.channel<i1>) -> (out: !esi.channel<i1>, a: i3)

// CHECK-LABEL:  esi.pure_module @top {
// CHECK-NEXT:     [[r0:%.+]] = esi.pure_module.input "clk" : i1
// CHECK-NEXT:     %foo.out, %foo.a = msft.instance @foo @Foo([[r0]], %foo.out)  : (i1, !esi.channel<i1>) -> (!esi.channel<i1>, i3)
// CHECK-NEXT:     esi.pure_module.output "a", %foo.a : i3
esi.pure_module @top {
  %clk = esi.pure_module.input "clk" : i1
  %loopback, %a = msft.instance @foo @Foo(%clk, %loopback) : (i1, !esi.channel<i1>) -> (!esi.channel<i1>, i3)
  esi.pure_module.output "a", %a : i3
}
