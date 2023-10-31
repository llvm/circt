// RUN: circt-opt --split-input-file --ibis-convert-containers-to-hw %s | FileCheck %s


// CHECK:  hw.module @B(in %in : i1 {inputAttr}, out out : i1 {outputAttr}) {
// CHECK:    hw.output %in : i1
// CHECK:  }
// CHECK:  hw.module @AccessSibling(in %p_b_out : i1, out p_b_in : i1) {
// CHECK:    hw.output %p_b_out : i1
// CHECK:  }
// CHECK:  hw.module @Parent() {
// CHECK:    %a.p_b_in = hw.instance "a" @AccessSibling(p_b_out: %b.out: i1) -> (p_b_in: i1)
// CHECK:    %b.out = hw.instance "b" @B(in: %a.p_b_in: i1) -> (out: i1)
// CHECK:    hw.output
// CHECK:  }

ibis.container @B {
  %this = ibis.this @B 
  %in = ibis.port.input @in : i1 {"inputAttr"}
  %out = ibis.port.output @out : i1 {"outputAttr"}

  // Loopback.
  %v = ibis.port.read %in : !ibis.portref<in i1>
  ibis.port.write %out, %v : !ibis.portref<out i1>
}

ibis.container @AccessSibling {
  %this = ibis.this @AccessSibling 
  %p_b_out = ibis.port.input @p_b_out : i1
  %p_b_in = ibis.port.output @p_b_in : i1
  ibis.port.write %p_b_in, %p_b_out.val : !ibis.portref<out i1>
  %p_b_out.val = ibis.port.read %p_b_out : !ibis.portref<in i1>
}
ibis.container @Parent {
  %this = ibis.this @Parent 
  %a = ibis.container.instance @a, @AccessSibling 
  %a.p_b_out.ref = ibis.get_port %a, @p_b_out : !ibis.scoperef<@AccessSibling> -> !ibis.portref<in i1>
  %b.out.ref.val = ibis.port.read %b.out.ref : !ibis.portref<out i1>
  ibis.port.write %a.p_b_out.ref, %b.out.ref.val : !ibis.portref<in i1>
  %a.p_b_in.ref = ibis.get_port %a, @p_b_in : !ibis.scoperef<@AccessSibling> -> !ibis.portref<out i1>
  %a.p_b_in.ref.val = ibis.port.read %a.p_b_in.ref : !ibis.portref<out i1>
  ibis.port.write %b.in.ref, %a.p_b_in.ref.val : !ibis.portref<in i1>
  %b = ibis.container.instance @b, @B 
  %b.out.ref = ibis.get_port %b, @out : !ibis.scoperef<@B> -> !ibis.portref<out i1>
  %b.in.ref = ibis.get_port %b, @in : !ibis.scoperef<@B> -> !ibis.portref<in i1>
}

// -----

// Test that we can instantiate and get ports of a container from a hw.module.

// CHECK:  hw.module @C(in %in : i1, out out : i1) {
// CHECK:    hw.output %in : i1
// CHECK:  }
// CHECK:  hw.module @Top() {
// CHECK:    %c.out = hw.instance "c" @C(in: %c.out: i1) -> (out: i1)
// CHECK:    hw.output
// CHECK:  }

ibis.container @C {
  %this = ibis.this @C
  %in = ibis.port.input @in : i1
  %out = ibis.port.output @out : i1
  %v = ibis.port.read %in : !ibis.portref<in i1>
  ibis.port.write %out, %v : !ibis.portref<out i1>
}

hw.module @Top() {
  %c = ibis.container.instance @c, @C
  %in = ibis.get_port %c, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
  %out = ibis.get_port %c, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
  %v = ibis.port.read %out : !ibis.portref<out i1>
  ibis.port.write %in, %v : !ibis.portref<in i1>
}

// -----

// Test that we can also move non-ibis ops

// CHECK-LABEL:   hw.module @Inst(out out : i1) {
// CHECK:           %[[VAL_0:.*]] = hw.constant true
// CHECK:           hw.output %[[VAL_0]] : i1
// CHECK:         }

// CHECK-LABEL:   hw.module @Top() {
// CHECK:           %[[VAL_0:.*]] = hw.instance "myInst" @Inst() -> (out: i1)
// CHECK:           %[[VAL_1:.*]] = hw.constant true
// CHECK:           %[[VAL_2:.*]] = comb.and bin %[[VAL_1]], %[[VAL_0]] : i1
// CHECK:           hw.output
// CHECK:         }

ibis.container @Inst {
  %this = ibis.this @Inst
  %out = ibis.port.output @out : i1
  %true = hw.constant 1 : i1
  ibis.port.write %out, %true : !ibis.portref<out i1>
}
ibis.container @Top {
  %this = ibis.this @Top
  %myInst = ibis.container.instance @myInst, @Inst
  %true = hw.constant 1 : i1
  %out.ref = ibis.get_port %myInst, @out : !ibis.scoperef<@Inst> -> !ibis.portref<out i1>
  %out.v = ibis.port.read %out.ref : !ibis.portref<out i1>
  %blake = comb.and bin %true, %out.v : i1
}
