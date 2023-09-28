// RUN: circt-opt %s  | circt-opt | FileCheck %s
// RUN: circt-opt %s --lower-esi-to-physical | circt-opt | FileCheck %s --check-prefix=PHY

hw.module.extern @Foo (in %clk: i1, in %in0 : !esi.channel<i1>, out out: !esi.channel<i1>, out a: i3)

// CHECK-LABEL:  esi.pure_module @top attributes {esi.portFlattenStructs} {
// CHECK-NEXT:     [[r0:%.+]] = esi.pure_module.input "clk" : i1
// CHECK-NEXT:     %foo.out, %foo.a = hw.instance "foo" @Foo(clk: [[r0]]: i1, in0: %foo.out: !esi.channel<i1>) -> (out: !esi.channel<i1>, a: i3)
// CHECK-NEXT:     esi.pure_module.output "a", %foo.a : i3

// PHY-LABEL:    hw.module @top<FOO: i8, STR: none>(in %clk : i1, out a : i3) attributes {esi.portFlattenStructs}
// PHY-NEXT:       %foo.out, %foo.a = hw.instance "foo" @Foo(clk: %clk: i1, in0: %foo.out: !esi.channel<i1>) -> (out: !esi.channel<i1>, a: i3)
// PHY-NEXT:       hw.output %foo.a : i3

esi.pure_module @top attributes { esi.portFlattenStructs } {
  %clk = esi.pure_module.input "clk" : i1
  %loopback, %a = hw.instance "foo" @Foo(clk: %clk: i1, in0: %loopback: !esi.channel<i1>) -> (out: !esi.channel<i1>, a: i3)
  esi.pure_module.output "a", %a : i3
  esi.pure_module.param "FOO" : i8
  esi.pure_module.param "STR" : none
}
