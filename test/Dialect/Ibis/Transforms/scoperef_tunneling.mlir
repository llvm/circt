// RUN: circt-opt --split-input-file --ibis-tunneling %s | FileCheck %s

ibis.design @D {

ibis.container @C {
  %this = ibis.this <@C>
  %in = ibis.port.input "in" sym @in : i1
  %out = ibis.port.output "out" sym @out : i1
}

// CHECK-LABEL:   ibis.container @AccessChild {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@AccessChild>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c, <@C>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = ibis.get_port %[[VAL_1]], @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECK-NEXT:         }

ibis.container @AccessChild {
  %this = ibis.this <@AccessChild>
  %c = ibis.container.instance @c, <@C>
  %c_ref = ibis.path [
    #ibis.step<child , @c : !ibis.scoperef<@C>>
  ]
  %c_in = ibis.get_port %c_ref, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
  %c_out = ibis.get_port %c_ref, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
}

}

// -----
// CHECK-LABEL: // -----

ibis.design @D {
ibis.container @C {
  %this = ibis.this <@C>
  %in = ibis.port.input "in" sym @in : i1
  %out = ibis.port.output "out" sym @out : i1
}

// CHECK-LABEL:   ibis.container @AccessSibling {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@AccessSibling>
// CHECK:           %[[VAL_1:.*]] = ibis.port.input "[[VAL_1]]" sym @[[VAL_1]] : !ibis.portref<out i1>
// CHECK:           %[[VAL_2:.*]] = ibis.port.read %[[VAL_1]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_3:.*]] = ibis.port.input "[[VAL_3]]" sym @[[VAL_3]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_4:.*]] = ibis.port.read %[[VAL_3]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:         }
ibis.container @AccessSibling {
  %this = ibis.this <@AccessSibling>
  %sibling = ibis.path [
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<child , @b : !ibis.scoperef<@C>>
  ]
  %c_in = ibis.get_port %sibling, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
  %c_out = ibis.get_port %sibling, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
}

// CHECK-LABEL:   ibis.container @Parent {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@Parent>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @a, <@AccessSibling>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @p_b_out.rd : !ibis.scoperef<@AccessSibling> -> !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_3:.*]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_4:.*]] = ibis.get_port %[[VAL_1]], @p_b_in.wr : !ibis.scoperef<@AccessSibling> -> !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_5:.*]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           %[[VAL_6:.*]] = ibis.container.instance @b, <@C>
// CHECK:           %[[VAL_3]] = ibis.get_port %[[VAL_6]], @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_5]] = ibis.get_port %[[VAL_6]], @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECK:         }
ibis.container @Parent {
  %this = ibis.this <@Parent>
  %a = ibis.container.instance @a, <@AccessSibling>
  %b = ibis.container.instance @b, <@C>
}

}

// -----
// CHECK-LABEL: // -----

// "Full"/recursive case.
// C1 child -> P1 parent -> P2 parent -> C2 child -> C3 child

ibis.design @D {
// CHECK-LABEL:   ibis.container @C1 {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@C1>
// CHECK:           %[[VAL_1:.*]] = ibis.port.input "p_p_c2_c3_out.rd" sym @p_p_c2_c3_out.rd : !ibis.portref<out i1>
// CHECK:           %[[VAL_2:.*]] = ibis.port.read %[[VAL_1]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_3:.*]] = ibis.port.input "p_p_c2_c3_in.wr" sym @p_p_c2_c3_in.wr : !ibis.portref<in i1>
// CHECK:           %[[VAL_4:.*]] = ibis.port.read %[[VAL_3]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:         }
ibis.container @C1 {
  %this = ibis.this <@C1>
  %c3 = ibis.path [
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<child , @c2 : !ibis.scoperef>,
    #ibis.step<child , @c3 : !ibis.scoperef<@C>>
  ]
  %c_in = ibis.get_port %c3, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
  %c_out = ibis.get_port %c3, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
}

// CHECK-LABEL:   ibis.container @C2 {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@C2>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c3, <@C>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = ibis.get_port %[[VAL_1]], @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECK:           %[[VAL_4:.*]] = ibis.port.output "p_p_c2_c3_out.rd" sym @p_p_c2_c3_out.rd : !ibis.portref<out i1>
// CHECK:           %[[VAL_5:.*]] = ibis.port.output "p_p_c2_c3_in.wr" sym @p_p_c2_c3_in.wr : !ibis.portref<in i1>
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_2]] : !ibis.portref<out !ibis.portref<out i1>>
// CHECK:           ibis.port.write %[[VAL_5]], %[[VAL_3]] : !ibis.portref<out !ibis.portref<in i1>>
// CHECK:         }
ibis.container @C2 {
  %this = ibis.this <@C2>
  %c3 = ibis.container.instance @c3, <@C>
}

ibis.container @C {
  %this = ibis.this <@C>
  %in = ibis.port.input "in" sym @in : i1
  %out = ibis.port.output "out" sym @out : i1
}

// CHECK-LABEL:   ibis.container @P1 {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@P1>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c1, <@C1>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @p_p_c2_c3_out.rd : !ibis.scoperef<@C1> -> !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_3:.*]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_4:.*]] = ibis.get_port %[[VAL_1]], @p_p_c2_c3_in.wr : !ibis.scoperef<@C1> -> !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_5:.*]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           %[[VAL_6:.*]] = ibis.port.input "p_p_c2_c3_out.rd" sym @p_p_c2_c3_out.rd : !ibis.portref<out i1>
// CHECK:           %[[VAL_3]] = ibis.port.read %[[VAL_6]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_7:.*]] = ibis.port.input "p_p_c2_c3_in.wr" sym @p_p_c2_c3_in.wr : !ibis.portref<in i1>
// CHECK:           %[[VAL_5]] = ibis.port.read %[[VAL_7]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:         }
ibis.container @P1 {
  %this = ibis.this <@P1>
  %c1 = ibis.container.instance @c1, <@C1>
}

// CHECK-LABEL:   ibis.container @P2 {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@P2>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @p1, <@P1>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @p_p_c2_c3_out.rd : !ibis.scoperef<@P1> -> !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_3:.*]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_4:.*]] = ibis.get_port %[[VAL_1]], @p_p_c2_c3_in.wr : !ibis.scoperef<@P1> -> !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_5:.*]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           %[[VAL_6:.*]] = ibis.container.instance @c2, <@C2>
// CHECK:           %[[VAL_7:.*]] = ibis.get_port %[[VAL_6]], @p_p_c2_c3_in.wr : !ibis.scoperef<@C2> -> !ibis.portref<out !ibis.portref<in i1>>
// CHECK:           %[[VAL_5]] = ibis.port.read %[[VAL_7]] : !ibis.portref<out !ibis.portref<in i1>>
// CHECK:           %[[VAL_8:.*]] = ibis.get_port %[[VAL_6]], @p_p_c2_c3_out.rd : !ibis.scoperef<@C2> -> !ibis.portref<out !ibis.portref<out i1>>
// CHECK:           %[[VAL_3]] = ibis.port.read %[[VAL_8]] : !ibis.portref<out !ibis.portref<out i1>>
// CHECK:         }
ibis.container @P2 {
  %this = ibis.this <@P2>
  %p1 = ibis.container.instance @p1, <@P1>
  %c2 = ibis.container.instance @c2, <@C2>
}

}

// -----
// CHECK-LABEL: // -----

ibis.design @D {
// CHECK-LABEL:   ibis.container @AccessParent {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@AccessParent>
// CHECK:           %[[VAL_1:.*]] = ibis.port.input "p_out.wr" sym @p_out.wr : !ibis.portref<in i1>
// CHECK:           %[[VAL_2:.*]] = ibis.port.read %[[VAL_1]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           %[[VAL_3:.*]] = ibis.port.input "p_in.rd" sym @p_in.rd : !ibis.portref<out i1>
// CHECK:           %[[VAL_4:.*]] = ibis.port.read %[[VAL_3]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:         }
ibis.container @AccessParent {
  %this = ibis.this <@AccessParent>
  %p = ibis.path [
    #ibis.step<parent : !ibis.scoperef<@Parent>>
  ]

  // get_port states the intended usage of the port. Hence we should be able to
  // request a parent input port as an output port (readable), and vice versa.
  // This will insert wires in the target container to facilitate the direction
  // flip.
  %p_in_ref = ibis.get_port %p, @in : !ibis.scoperef<@Parent> -> !ibis.portref<out i1>
  %p_out_ref = ibis.get_port %p, @out : !ibis.scoperef<@Parent> -> !ibis.portref<in i1>
}

// CHECK-LABEL:   ibis.container @Parent {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@Parent>
// CHECK:           %[[VAL_1:.*]] = ibis.port.input "in" sym @in : i1
// CHECK:           %[[VAL_2:.*]] = ibis.port.read %[[VAL_1]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_3:.*]] = ibis.wire.output @in.rd, %[[VAL_2]] : i1
// CHECK:           %[[VAL_4:.*]] = ibis.port.output "out" sym @out : i1
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = ibis.wire.input @out.wr : i1
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_6]] : !ibis.portref<out i1>
// CHECK:           %[[VAL_7:.*]] = ibis.container.instance @c, <@AccessParent>
// CHECK:           %[[VAL_8:.*]] = ibis.get_port %[[VAL_7]], @p_out.wr : !ibis.scoperef<@AccessParent> -> !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           ibis.port.write %[[VAL_8]], %[[VAL_5]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           %[[VAL_9:.*]] = ibis.get_port %[[VAL_7]], @p_in.rd : !ibis.scoperef<@AccessParent> -> !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           ibis.port.write %[[VAL_9]], %[[VAL_3]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:         }
ibis.container @Parent {
  %this = ibis.this <@Parent>
  %in = ibis.port.input "in" sym @in : i1
  %out = ibis.port.output "out" sym @out : i1
  %c = ibis.container.instance @c, <@AccessParent>
}

}

// -----
// CHECK-LABEL: // -----

ibis.design @D {

ibis.container @C {
  %this = ibis.this <@C>
  %in = ibis.port.input "in" sym @in : i1
  %out = ibis.port.output "out" sym @out : i1
}

// TODO: support this case. Too hard to support now. Problem is that hw.module
// cannot live within an ibis.design, but we need to run this pass on the
// ibis.design op. I don't think it's critical that we support this case currently.

// CHECKx-LABEL:   hw.module @AccessChildFromHW() {
// CHECKx:           %[[VAL_0:.*]] = ibis.container.instance @c, <@C>
// CHECKx:           %[[VAL_1:.*]] = ibis.get_port %[[VAL_0]], @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECKx:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_0]], @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECKx:           hw.output
// CHECKx:         }

}

hw.module @AccessChildFromHW() {
  %c = ibis.container.instance @c, <@C>
  %c_ref = ibis.path [
    #ibis.step<child , @c : !ibis.scoperef<@C>>
  ]
  %c_in = ibis.get_port %c_ref, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
  %c_out = ibis.get_port %c_ref, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
}


// -----
// CHECK-LABEL: // -----

ibis.design @D {
// The ultimate tunneling test - by having 3 intermediate levels for both up-
// and downwards tunneling, we are sure test all the possible combinations of
// tunneling.

ibis.container @D_up {
  %this = ibis.this <@D_up>
  %d = ibis.path [
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<child, @a_down : !ibis.scoperef> : !ibis.scoperef,
    #ibis.step<child, @b : !ibis.scoperef> : !ibis.scoperef,
    #ibis.step<child, @c : !ibis.scoperef> : !ibis.scoperef,
    #ibis.step<child, @d : !ibis.scoperef<@D_down>> : !ibis.scoperef<@D_down>]
  // Write an input port
  %clk_ref = ibis.get_port %d, @clk_in : !ibis.scoperef<@D_down> -> !ibis.portref<in i1>
  %clk = hw.constant 1 : i1
  ibis.port.write %clk_ref, %clk : !ibis.portref<in i1>

  // Read an input port
  %clk_ref_2 = ibis.get_port %d, @clk_in : !ibis.scoperef<@D_down> -> !ibis.portref<out i1>
  %clk_in_val = ibis.port.read %clk_ref_2 : !ibis.portref<out i1>

  // Read an output port
  %clk_out_ref = ibis.get_port %d, @clk_out : !ibis.scoperef<@D_down> -> !ibis.portref<out i1>
  %clk_out_val = ibis.port.read %clk_out_ref : !ibis.portref<out i1>
}
ibis.container @C_up {
  %this = ibis.this <@C_up>
  %d = ibis.container.instance @d, <@D_up>
}
ibis.container @B_up {
  %this = ibis.this <@B_up>
  %c = ibis.container.instance @c, <@C_up>
  
}

ibis.container @A_up {
  %this = ibis.this <@A_up>
  %b = ibis.container.instance @b, <@B_up>
}

ibis.container @Top {
  %this = ibis.this <@Top>
  %a_down = ibis.container.instance @a_down, <@A_down>
  %a_up = ibis.container.instance @a_up, <@A_up>
}
ibis.container @A_down {
  %this = ibis.this <@A_down>
  %b = ibis.container.instance @b, <@B_down>
}
ibis.container @B_down {
  %this = ibis.this <@B_down>
  %c = ibis.container.instance @c, <@C_down>
}
ibis.container @C_down {
  %this = ibis.this <@C_down>
  %d = ibis.container.instance @d, <@D_down>
}
ibis.container @D_down {
  %this = ibis.this <@D_down>
  %clk = ibis.port.input "clk_in" sym @clk_in : i1
  %clk_out = ibis.port.output "clk_out" sym @clk_out : i1
  %clk.val = ibis.port.read %clk : !ibis.portref<in i1>
  ibis.port.write %clk_out, %clk.val : !ibis.portref<out i1>
}
}
