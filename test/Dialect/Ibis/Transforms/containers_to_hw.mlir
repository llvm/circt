// RUN: circt-opt --split-input-file --ibis-convert-containers-to-hw %s | FileCheck %s


// CHECK:  hw.module @B(%in: i1) -> (out: i1) {
// CHECK:    hw.output %in : i1
// CHECK:  }
// CHECK:  hw.module @AccessSibling(%p_b_out: i1) -> (p_b_in: i1) {
// CHECK:    hw.output %p_b_out : i1
// CHECK:  }
// CHECK:  hw.module @Parent() {
// CHECK:    %a.p_b_in = hw.instance "a" @AccessSibling(p_b_out: %b.out: i1) -> (p_b_in: i1)
// CHECK:    %b.out = hw.instance "b" @B(in: %a.p_b_in: i1) -> (out: i1)
// CHECK:    hw.output
// CHECK:  }

ibis.container @B {
  %this = ibis.this @B 
  %in = ibis.port.input @in : i1
  %out = ibis.port.output @out : i1

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

// Test that we can instantiate and get ports of a container from a hw.module.

// CHECK:  hw.module @C(%in: i1) -> (out: i1) {
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

hw.module @Top() -> () {
  %c = ibis.container.instance @c, @C
  %in = ibis.get_port %c, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
  %out = ibis.get_port %c, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
  %v = ibis.port.read %out : !ibis.portref<out i1>
  ibis.port.write %in, %v : !ibis.portref<in i1>
}
