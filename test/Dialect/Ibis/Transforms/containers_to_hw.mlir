// RUN: circt-opt --split-input-file --ibis-convert-containers-to-hw %s | FileCheck %s

ibis.design @D {

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

ibis.container "MyB" sym @B {
  %this = ibis.this <@D::@B>
  // Test different port names vs. symbol names
  %in = ibis.port.input "in_foo" sym @in : i1 {"inputAttr"}
  %out = ibis.port.output "out_foo" sym @out : i1 {"outputAttr"}

  // Loopback.
  %v = ibis.port.read %in : !ibis.portref<in i1>
  ibis.port.write %out, %v : !ibis.portref<out i1>
}

ibis.container sym @AccessSibling {
  %this = ibis.this <@D::@AccessSibling>
  %p_b_out = ibis.port.input "p_b_out_foo" sym @p_b_out : i1
  %p_b_in = ibis.port.output "p_b_in_foo" sym @p_b_in : i1
  ibis.port.write %p_b_in, %p_b_out.val : !ibis.portref<out i1>
  %p_b_out.val = ibis.port.read %p_b_out : !ibis.portref<in i1>
}
ibis.container sym @Parent top_level {
  %this = ibis.this <@D::@Parent>
  %a = ibis.container.instance @a, <@D::@AccessSibling>
  %a.p_b_out.ref = ibis.get_port %a, @p_b_out : !ibis.scoperef<@D::@AccessSibling> -> !ibis.portref<in i1>
  %b.out.ref.val = ibis.port.read %b.out.ref : !ibis.portref<out i1>
  ibis.port.write %a.p_b_out.ref, %b.out.ref.val : !ibis.portref<in i1>
  %a.p_b_in.ref = ibis.get_port %a, @p_b_in : !ibis.scoperef<@D::@AccessSibling> -> !ibis.portref<out i1>
  %a.p_b_in.ref.val = ibis.port.read %a.p_b_in.ref : !ibis.portref<out i1>
  ibis.port.write %b.in.ref, %a.p_b_in.ref.val : !ibis.portref<in i1>
  %b = ibis.container.instance @b, <@D::@B>
  %b.out.ref = ibis.get_port %b, @out : !ibis.scoperef<@D::@B> -> !ibis.portref<out i1>
  %b.in.ref = ibis.get_port %b, @in : !ibis.scoperef<@D::@B> -> !ibis.portref<in i1>
}

}

// -----

// Test that we can instantiate and get ports of a container from a hw.module.

ibis.design @D {

// CHECK:  hw.module @D_C(in %in_foo : i1, out out_foo : i1) {
// CHECK:    hw.output %in_foo : i1
// CHECK:  }
// CHECK:  hw.module @Top() {
// CHECK:    %c.out_foo = hw.instance "c" @D_C(in_foo: %c.out_foo: i1) -> (out_foo: i1)
// CHECK:    hw.output
// CHECK:  }

ibis.container sym @C {
  %this = ibis.this <@D::@C>
  %in = ibis.port.input "in_foo" sym @in : i1
  %out = ibis.port.output "out_foo" sym @out : i1
  %v = ibis.port.read %in : !ibis.portref<in i1>
  ibis.port.write %out, %v : !ibis.portref<out i1>
}

}

hw.module @Top() {
  %c = ibis.container.instance @c, <@D::@C>
  %in = ibis.get_port %c, @in : !ibis.scoperef<@D::@C> -> !ibis.portref<in i1>
  %out = ibis.get_port %c, @out : !ibis.scoperef<@D::@C> -> !ibis.portref<out i1>
  %v = ibis.port.read %out : !ibis.portref<out i1>
  ibis.port.write %in, %v : !ibis.portref<in i1>
}

// -----

// Test that we can also move non-ibis ops

ibis.design @D {

// CHECK-LABEL:   hw.module @D_Inst(out out : i1) {
// CHECK:           %[[VAL_0:.*]] = hw.constant true
// CHECK:           hw.output %[[VAL_0]] : i1
// CHECK:         }

// CHECK-LABEL:   hw.module @D_Top() {
// CHECK:           %[[VAL_0:.*]] = hw.instance "myInst" @D_Inst() -> (out: i1)
// CHECK:           %[[VAL_1:.*]] = hw.constant true
// CHECK:           %[[VAL_2:.*]] = comb.and bin %[[VAL_1]], %[[VAL_0]] : i1
// CHECK:           hw.output
// CHECK:         }

ibis.container sym @Inst {
  %this = ibis.this <@D::@Inst>
  %out = ibis.port.output "out" sym @out : i1
  %true = hw.constant 1 : i1
  ibis.port.write %out, %true : !ibis.portref<out i1>
}
ibis.container sym @Top {
  %this = ibis.this <@D::@Top>
  %myInst = ibis.container.instance @myInst, <@D::@Inst>
  %true = hw.constant 1 : i1
  %out.ref = ibis.get_port %myInst, @out : !ibis.scoperef<@D::@Inst> -> !ibis.portref<out i1>
  %out.v = ibis.port.read %out.ref : !ibis.portref<out i1>
  %blake = comb.and bin %true, %out.v : i1
}

}

// -----

// Test that we can unique duplicate port names.

ibis.design @D {

// CHECK:   hw.module @D_Top(in %clk : i1, in %clk_0 : i1, out out : i1, out out_0 : i1) {
// CHECK:     hw.output %clk, %clk_0 : i1, i1
ibis.container sym @Top {
  %this = ibis.this <@D::@Top>
  %clk1 = ibis.port.input "clk" sym @clk1 : i1
  %clk2 = ibis.port.input "clk" sym @clk2 : i1
  %out1 = ibis.port.output "out" sym @out1 : i1
  %out2 = ibis.port.output "out" sym @out2 : i1

  %v1 = ibis.port.read %clk1 : !ibis.portref<in i1>
  %v2 = ibis.port.read %clk2 : !ibis.portref<in i1>
  ibis.port.write %out1, %v1 : !ibis.portref<out i1>
  ibis.port.write %out2, %v2 : !ibis.portref<out i1>
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

ibis.design @D {

ibis.container "Foo" sym @A {
  %this = ibis.this <@D::@A>
}

ibis.container "Foo" sym @B top_level {
  %this = ibis.this <@D::@B>
}
}

hw.module.extern @D_Foo(in %theExternModule : i1)
hw.module.extern @Foo(in %theExternModule : i1)
