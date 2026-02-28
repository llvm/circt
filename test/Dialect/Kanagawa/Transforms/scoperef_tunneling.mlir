// RUN: circt-opt --split-input-file --kanagawa-tunneling %s | FileCheck %s

kanagawa.design @D {

kanagawa.container sym @C {
  %in = kanagawa.port.input "in" sym @in : i1
  %out = kanagawa.port.output "out" sym @out : i1
}

// CHECK-LABEL:   kanagawa.container sym @AccessChild {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @c, <@D::@C>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.get_port %[[VAL_1]], @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
// CHECK-NEXT:         }

kanagawa.container sym @AccessChild {
  %c = kanagawa.container.instance @c, <@D::@C>
  %c_ref = kanagawa.path [
    #kanagawa.step<child , @c : !kanagawa.scoperef<@D::@C>>
  ]
  %c_in = kanagawa.get_port %c_ref, @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
  %c_out = kanagawa.get_port %c_ref, @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
}

}

// -----
// CHECK-LABEL: // -----

kanagawa.design @D {
kanagawa.container sym @C {
  %in = kanagawa.port.input "in" sym @in : i1
  %out = kanagawa.port.output "out" sym @out : i1
}

// CHECK-LABEL:   kanagawa.container sym @AccessSibling {
// CHECK:           %[[VAL_1:.*]] = kanagawa.port.input "[[VAL_1]]" sym @[[VAL_1]] : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_2:.*]] = kanagawa.port.read %[[VAL_1]] : !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.input "[[VAL_3]]" sym @[[VAL_3]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_4:.*]] = kanagawa.port.read %[[VAL_3]] : !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:         }
kanagawa.container sym @AccessSibling {
  %sibling = kanagawa.path [
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<child , @b : !kanagawa.scoperef<@D::@C>>
  ]
  %c_in = kanagawa.get_port %sibling, @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
  %c_out = kanagawa.get_port %sibling, @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
}

// CHECK-LABEL:   kanagawa.container sym @Parent {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @a, <@D::@AccessSibling>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @p_b_out.rd : !kanagawa.scoperef<@D::@AccessSibling> -> !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:           kanagawa.port.write %[[VAL_2]], %[[VAL_3:.*]] : !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:           %[[VAL_4:.*]] = kanagawa.get_port %[[VAL_1]], @p_b_in.wr : !kanagawa.scoperef<@D::@AccessSibling> -> !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:           kanagawa.port.write %[[VAL_4]], %[[VAL_5:.*]] : !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:           %[[VAL_6:.*]] = kanagawa.container.instance @b, <@D::@C>
// CHECK:           %[[VAL_3]] = kanagawa.get_port %[[VAL_6]], @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_5]] = kanagawa.get_port %[[VAL_6]], @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
// CHECK:         }
kanagawa.container sym @Parent {
  %a = kanagawa.container.instance @a, <@D::@AccessSibling>
  %b = kanagawa.container.instance @b, <@D::@C>
}

}

// -----
// CHECK-LABEL: // -----

// "Full"/recursive case.
// C1 child -> P1 parent -> P2 parent -> C2 child -> C3 child

kanagawa.design @D {
// CHECK-LABEL:   kanagawa.container sym @C1 {
// CHECK:           %[[VAL_1:.*]] = kanagawa.port.input "p_p_c2_c3_out.rd" sym @p_p_c2_c3_out.rd : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_2:.*]] = kanagawa.port.read %[[VAL_1]] : !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.input "p_p_c2_c3_in.wr" sym @p_p_c2_c3_in.wr : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_4:.*]] = kanagawa.port.read %[[VAL_3]] : !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:         }
kanagawa.container sym @C1 {
  %c3 = kanagawa.path [
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<child , @c2 : !kanagawa.scoperef>,
    #kanagawa.step<child , @c3 : !kanagawa.scoperef<@D::@C>>
  ]
  %c_in = kanagawa.get_port %c3, @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
  %c_out = kanagawa.get_port %c3, @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
}

// CHECK-LABEL:   kanagawa.container sym @C2 {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @c3, <@D::@C>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.get_port %[[VAL_1]], @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
// CHECK:           %[[VAL_4:.*]] = kanagawa.port.output "p_p_c2_c3_out.rd" sym @p_p_c2_c3_out.rd : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_5:.*]] = kanagawa.port.output "p_p_c2_c3_in.wr" sym @p_p_c2_c3_in.wr : !kanagawa.portref<in i1>
// CHECK:           kanagawa.port.write %[[VAL_4]], %[[VAL_2]] : !kanagawa.portref<out !kanagawa.portref<out i1>>
// CHECK:           kanagawa.port.write %[[VAL_5]], %[[VAL_3]] : !kanagawa.portref<out !kanagawa.portref<in i1>>
// CHECK:         }
kanagawa.container sym @C2 {
  %c3 = kanagawa.container.instance @c3, <@D::@C>
}

kanagawa.container sym @C {
  %in = kanagawa.port.input "in" sym @in : i1
  %out = kanagawa.port.output "out" sym @out : i1
}

// CHECK-LABEL:   kanagawa.container sym @P1 {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @c1, <@D::@C1>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @p_p_c2_c3_out.rd : !kanagawa.scoperef<@D::@C1> -> !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:           kanagawa.port.write %[[VAL_2]], %[[VAL_3:.*]] : !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:           %[[VAL_4:.*]] = kanagawa.get_port %[[VAL_1]], @p_p_c2_c3_in.wr : !kanagawa.scoperef<@D::@C1> -> !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:           kanagawa.port.write %[[VAL_4]], %[[VAL_5:.*]] : !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:           %[[VAL_6:.*]] = kanagawa.port.input "p_p_c2_c3_out.rd" sym @p_p_c2_c3_out.rd : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_3]] = kanagawa.port.read %[[VAL_6]] : !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:           %[[VAL_7:.*]] = kanagawa.port.input "p_p_c2_c3_in.wr" sym @p_p_c2_c3_in.wr : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_5]] = kanagawa.port.read %[[VAL_7]] : !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:         }
kanagawa.container sym @P1 {
  %c1 = kanagawa.container.instance @c1, <@D::@C1>
}

// CHECK-LABEL:   kanagawa.container sym @P2 {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @p1, <@D::@P1>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @p_p_c2_c3_out.rd : !kanagawa.scoperef<@D::@P1> -> !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:           kanagawa.port.write %[[VAL_2]], %[[VAL_3:.*]] : !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:           %[[VAL_4:.*]] = kanagawa.get_port %[[VAL_1]], @p_p_c2_c3_in.wr : !kanagawa.scoperef<@D::@P1> -> !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:           kanagawa.port.write %[[VAL_4]], %[[VAL_5:.*]] : !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:           %[[VAL_6:.*]] = kanagawa.container.instance @c2, <@D::@C2>
// CHECK:           %[[VAL_7:.*]] = kanagawa.get_port %[[VAL_6]], @p_p_c2_c3_in.wr : !kanagawa.scoperef<@D::@C2> -> !kanagawa.portref<out !kanagawa.portref<in i1>>
// CHECK:           %[[VAL_5]] = kanagawa.port.read %[[VAL_7]] : !kanagawa.portref<out !kanagawa.portref<in i1>>
// CHECK:           %[[VAL_8:.*]] = kanagawa.get_port %[[VAL_6]], @p_p_c2_c3_out.rd : !kanagawa.scoperef<@D::@C2> -> !kanagawa.portref<out !kanagawa.portref<out i1>>
// CHECK:           %[[VAL_3]] = kanagawa.port.read %[[VAL_8]] : !kanagawa.portref<out !kanagawa.portref<out i1>>
// CHECK:         }
kanagawa.container sym @P2 {
  %p1 = kanagawa.container.instance @p1, <@D::@P1>
  %c2 = kanagawa.container.instance @c2, <@D::@C2>
}

}

// -----
// CHECK-LABEL: // -----

kanagawa.design @D {
// CHECK-LABEL:   kanagawa.container sym @AccessParent {
// CHECK:           %[[VAL_1:.*]] = kanagawa.port.input "p_out.wr" sym @p_out.wr : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_2:.*]] = kanagawa.port.read %[[VAL_1]] : !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.input "p_in.rd" sym @p_in.rd : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_4:.*]] = kanagawa.port.read %[[VAL_3]] : !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:         }
kanagawa.container sym @AccessParent {
  %p = kanagawa.path [
    #kanagawa.step<parent : !kanagawa.scoperef<@D::@Parent>>
  ]

  // get_port states the intended usage of the port. Hence we should be able to
  // request a parent input port as an output port (readable), and vice versa.
  // This will insert wires in the target container to facilitate the direction
  // flip.
  %p_in_ref = kanagawa.get_port %p, @in : !kanagawa.scoperef<@D::@Parent> -> !kanagawa.portref<out i1>
  %p_out_ref = kanagawa.get_port %p, @out : !kanagawa.scoperef<@D::@Parent> -> !kanagawa.portref<in i1>
}

// CHECK-LABEL:   kanagawa.container sym @Parent {
// CHECK:           %[[VAL_1:.*]] = kanagawa.port.input "in" sym @in : i1
// CHECK:           %[[VAL_2:.*]] = kanagawa.port.read %[[VAL_1]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.wire.output @in.rd, %[[VAL_2]] : i1
// CHECK:           %[[VAL_4:.*]] = kanagawa.port.output "out" sym @out : i1
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = kanagawa.wire.input @out.wr : i1
// CHECK:           kanagawa.port.write %[[VAL_4]], %[[VAL_6]] : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_7:.*]] = kanagawa.container.instance @c, <@D::@AccessParent>
// CHECK:           %[[VAL_8:.*]] = kanagawa.get_port %[[VAL_7]], @p_out.wr : !kanagawa.scoperef<@D::@AccessParent> -> !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:           kanagawa.port.write %[[VAL_8]], %[[VAL_5]] : !kanagawa.portref<in !kanagawa.portref<in i1>>
// CHECK:           %[[VAL_9:.*]] = kanagawa.get_port %[[VAL_7]], @p_in.rd : !kanagawa.scoperef<@D::@AccessParent> -> !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:           kanagawa.port.write %[[VAL_9]], %[[VAL_3]] : !kanagawa.portref<in !kanagawa.portref<out i1>>
// CHECK:         }
kanagawa.container sym @Parent {
  %in = kanagawa.port.input "in" sym @in : i1
  %out = kanagawa.port.output "out" sym @out : i1
  %c = kanagawa.container.instance @c, <@D::@AccessParent>
}

}

// -----
// CHECK-LABEL: // -----

kanagawa.design @D {

kanagawa.container sym @C {
  %in = kanagawa.port.input "in" sym @in : i1
  %out = kanagawa.port.output "out" sym @out : i1
}

// TODO: support this case. Too hard to support now. Problem is that hw.module
// cannot live within an kanagawa.design, but we need to run this pass on the
// kanagawa.design op. I don't think it's critical that we support this case currently.

// COM: CHECK-LABEL:   hw.module @AccessChildFromHW() {
// COM: CHECK:           %[[VAL_0:.*]] = kanagawa.container.instance @c, <@D::@C>
// COM: CHECK:           %[[VAL_1:.*]] = kanagawa.get_port %[[VAL_0]], @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
// COM: CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_0]], @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
// COM: CHECK:           hw.output
// COM: CHECK:         }

}

hw.module @AccessChildFromHW() {
  %c = kanagawa.container.instance @c, <@D::@C>
  %c_ref = kanagawa.path [
    #kanagawa.step<child , @c : !kanagawa.scoperef<@D::@C>>
  ]
  %c_in = kanagawa.get_port %c_ref, @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
  %c_out = kanagawa.get_port %c_ref, @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
}


// -----
// CHECK-LABEL: // -----

kanagawa.design @D {
// The ultimate tunneling test - by having 3 intermediate levels for both up-
// and downwards tunneling, we are sure test all the possible combinations of
// tunneling.

kanagawa.container sym @D_up {
  %d = kanagawa.path [
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<child, @a_down : !kanagawa.scoperef> : !kanagawa.scoperef,
    #kanagawa.step<child, @b : !kanagawa.scoperef> : !kanagawa.scoperef,
    #kanagawa.step<child, @c : !kanagawa.scoperef> : !kanagawa.scoperef,
    #kanagawa.step<child, @d : !kanagawa.scoperef<@D::@D_down>> : !kanagawa.scoperef<@D::@D_down>]
  // Write an input port
  %clk_ref = kanagawa.get_port %d, @clk_in : !kanagawa.scoperef<@D::@D_down> -> !kanagawa.portref<in i1>
  %clk = hw.constant 1 : i1
  kanagawa.port.write %clk_ref, %clk : !kanagawa.portref<in i1>

  // Read an input port
  %clk_ref_2 = kanagawa.get_port %d, @clk_in : !kanagawa.scoperef<@D::@D_down> -> !kanagawa.portref<out i1>
  %clk_in_val = kanagawa.port.read %clk_ref_2 : !kanagawa.portref<out i1>

  // Read an output port
  %clk_out_ref = kanagawa.get_port %d, @clk_out : !kanagawa.scoperef<@D::@D_down> -> !kanagawa.portref<out i1>
  %clk_out_val = kanagawa.port.read %clk_out_ref : !kanagawa.portref<out i1>
}
kanagawa.container sym @C_up {
  %d = kanagawa.container.instance @d, <@D::@D_up>
}
kanagawa.container sym @B_up {
  %c = kanagawa.container.instance @c, <@D::@C_up>
  
}

kanagawa.container sym @A_up {
  %b = kanagawa.container.instance @b, <@D::@B_up>
}

kanagawa.container sym @Top {
  %a_down = kanagawa.container.instance @a_down, <@D::@A_down>
  %a_up = kanagawa.container.instance @a_up, <@D::@A_up>
}
kanagawa.container sym @A_down {
  %b = kanagawa.container.instance @b, <@D::@B_down>
}
kanagawa.container sym @B_down {
  %c = kanagawa.container.instance @c, <@D::@C_down>
}
kanagawa.container sym @C_down {
  %d = kanagawa.container.instance @d, <@D::@D_down>
}
kanagawa.container sym @D_down {
  %clk = kanagawa.port.input "clk_in" sym @clk_in : i1
  %clk_out = kanagawa.port.output "clk_out" sym @clk_out : i1
  %clk.val = kanagawa.port.read %clk : !kanagawa.portref<in i1>
  kanagawa.port.write %clk_out, %clk.val : !kanagawa.portref<out i1>
}
}
