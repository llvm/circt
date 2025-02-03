// RUN: circt-opt --split-input-file --kanagawa-convert-containers-to-hw %s | FileCheck %s

kanagawa.design @D {

// CHECK:  hw.module @D_MyB(in %in_foo : i1 {inputAttr}, out out_foo : i1 {outputAttr}) {
// CHECK:    hw.output %in_foo : i1
// CHECK:  }
// CHECK:  hw.module @D_AccessSibling(in %p_b_out_foo : i1, out p_b_in_foo : i1) {
// CHECK:    hw.output %p_b_out_foo : i1
// CHECK:  }
// CHECK:  hw.module @Parent() {
// CHECK:    %a.p_b_in_foo = hw.instance "a" @D_AccessSibling(p_b_out_foo: %b.out_foo: i1) -> (p_b_in_foo: i1)
// CHECK:    %b.out_foo = hw.instance "b" @D_MyB(in_foo: %a.p_b_in_foo: i1) -> (out_foo: i1)
// CHECK:    hw.output
// CHECK:  }

kanagawa.container "MyB" sym @B {
  // Test different port names vs. symbol names
  %in = kanagawa.port.input "in_foo" sym @in : i1 {"inputAttr"}
  %out = kanagawa.port.output "out_foo" sym @out : i1 {"outputAttr"}

  // Loopback.
  %v = kanagawa.port.read %in : !kanagawa.portref<in i1>
  kanagawa.port.write %out, %v : !kanagawa.portref<out i1>
}

kanagawa.container sym @AccessSibling {
  %p_b_out = kanagawa.port.input "p_b_out_foo" sym @p_b_out : i1
  %p_b_in = kanagawa.port.output "p_b_in_foo" sym @p_b_in : i1
  kanagawa.port.write %p_b_in, %p_b_out.val : !kanagawa.portref<out i1>
  %p_b_out.val = kanagawa.port.read %p_b_out : !kanagawa.portref<in i1>
}
kanagawa.container sym @Parent top_level {
  %a = kanagawa.container.instance @a, <@D::@AccessSibling>
  %a.p_b_out.ref = kanagawa.get_port %a, @p_b_out : !kanagawa.scoperef<@D::@AccessSibling> -> !kanagawa.portref<in i1>
  %b.out.ref.val = kanagawa.port.read %b.out.ref : !kanagawa.portref<out i1>
  kanagawa.port.write %a.p_b_out.ref, %b.out.ref.val : !kanagawa.portref<in i1>
  %a.p_b_in.ref = kanagawa.get_port %a, @p_b_in : !kanagawa.scoperef<@D::@AccessSibling> -> !kanagawa.portref<out i1>
  %a.p_b_in.ref.val = kanagawa.port.read %a.p_b_in.ref : !kanagawa.portref<out i1>
  kanagawa.port.write %b.in.ref, %a.p_b_in.ref.val : !kanagawa.portref<in i1>
  %b = kanagawa.container.instance @b, <@D::@B>
  %b.out.ref = kanagawa.get_port %b, @out : !kanagawa.scoperef<@D::@B> -> !kanagawa.portref<out i1>
  %b.in.ref = kanagawa.get_port %b, @in : !kanagawa.scoperef<@D::@B> -> !kanagawa.portref<in i1>
}

}

// -----

// Test that we can instantiate and get ports of a container from a hw.module.

kanagawa.design @D {

// CHECK:  hw.module @D_C(in %in_foo : i1, out out_foo : i1) {
// CHECK:    hw.output %in_foo : i1
// CHECK:  }
// CHECK:  hw.module @Top() {
// CHECK:    %c.out_foo = hw.instance "c" @D_C(in_foo: %c.out_foo: i1) -> (out_foo: i1)
// CHECK:    hw.output
// CHECK:  }

kanagawa.container sym @C {
  %in = kanagawa.port.input "in_foo" sym @in : i1
  %out = kanagawa.port.output "out_foo" sym @out : i1
  %v = kanagawa.port.read %in : !kanagawa.portref<in i1>
  kanagawa.port.write %out, %v : !kanagawa.portref<out i1>
}

}

hw.module @Top() {
  %c = kanagawa.container.instance @c, <@D::@C>
  %in = kanagawa.get_port %c, @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
  %out = kanagawa.get_port %c, @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
  %v = kanagawa.port.read %out : !kanagawa.portref<out i1>
  kanagawa.port.write %in, %v : !kanagawa.portref<in i1>
}

// -----

// Test that we can also move non-kanagawa ops

kanagawa.design @D {

// CHECK-LABEL:   hw.module @D_Inst(out out : i1) {
// CHECK:           %[[VAL_0:.*]] = hw.constant true
// CHECK:           hw.output %[[VAL_0]] : i1
// CHECK:         }

// CHECK-LABEL:   hw.module @D_Top() {
// CHECK:           %[[VAL_0:.*]] = hw.instance "myInst" @D_Inst() -> (out: i1)
// CHECK:           %[[VAL_1:.*]] = hw.constant true
// CHECK:           %[[VAL_2:.*]] = comb.and %[[VAL_1]], %[[VAL_0]] : i1
// CHECK:           hw.output
// CHECK:         }

kanagawa.container sym @Inst {
  %out = kanagawa.port.output "out" sym @out : i1
  %true = hw.constant 1 : i1
  kanagawa.port.write %out, %true : !kanagawa.portref<out i1>
}
kanagawa.container sym @Top {
  %myInst = kanagawa.container.instance @myInst, <@D::@Inst>
  %true = hw.constant 1 : i1
  %out.ref = kanagawa.get_port %myInst, @out : !kanagawa.scoperef<@D::@Inst> -> !kanagawa.portref<out i1>
  %out.v = kanagawa.port.read %out.ref : !kanagawa.portref<out i1>
  %blake = comb.and %true, %out.v : i1
}

}

// -----

// Test that we can unique duplicate port names.

kanagawa.design @D {

// CHECK:   hw.module @D_Top(in %clk : i1, in %clk_0 : i1, out out : i1, out out_0 : i1) {
// CHECK:     hw.output %clk, %clk_0 : i1, i1
kanagawa.container sym @Top {
  %clk1 = kanagawa.port.input "clk" sym @clk1 : i1
  %clk2 = kanagawa.port.input "clk" sym @clk2 : i1
  %out1 = kanagawa.port.output "out" sym @out1 : i1
  %out2 = kanagawa.port.output "out" sym @out2 : i1

  %v1 = kanagawa.port.read %clk1 : !kanagawa.portref<in i1>
  %v2 = kanagawa.port.read %clk2 : !kanagawa.portref<in i1>
  kanagawa.port.write %out1, %v1 : !kanagawa.portref<out i1>
  kanagawa.port.write %out2, %v2 : !kanagawa.portref<out i1>
}
}

// -----

// Test that we can de-alias module names.

// CHECK:  hw.module @D_Foo_0() {
// CHECK:    hw.output
// CHECK:  }
// CHECK:  hw.module @Foo_0() {
// CHECK:    hw.output
// CHECK:  }
// CHECK:  hw.module.extern @D_Foo(in %theExternModule : i1)
// CHECK:  hw.module.extern @Foo(in %theExternModule : i1)

kanagawa.design @D {

kanagawa.container "Foo" sym @A {
}

kanagawa.container "Foo" sym @B top_level {
}
}

hw.module.extern @D_Foo(in %theExternModule : i1)
hw.module.extern @Foo(in %theExternModule : i1)

// -----

// Test that containers with names that alias with the design op are not
// de-aliased.

// CHECK: hw.module @D

kanagawa.design @D {
  kanagawa.container "D" sym @D top_level {
  }
}
