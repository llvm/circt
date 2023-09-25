// RUN: circt-opt --split-input-file --ibis-tunneling %s | FileCheck %s

ibis.container @C {
  %this = ibis.this @C
  %in = ibis.port.input @in : i1
%out = ibis.port.output @out : i1
}

// CHECK-LABEL:   ibis.container @AccessChild {
// CHECK:           %[[VAL_0:.*]] = ibis.this @AccessChild
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c, @C
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = ibis.get_port %[[VAL_1]], @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECK-NEXT:         }

ibis.container @AccessChild {
  %this = ibis.this @AccessChild
  %c = ibis.container.instance @c, @C
  %c_ref = ibis.path [
    #ibis.step<child , @c : !ibis.scoperef<@C>>
  ]
  %c_in = ibis.get_port %c_ref, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
  %c_out = ibis.get_port %c_ref, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
}

// -----

ibis.container @C {
  %this = ibis.this @C
  %in = ibis.port.input @in : i1
  %out = ibis.port.output @out : i1
}

// CHECK-LABEL:   ibis.container @AccessSibling {
// CHECK:           %[[VAL_0:.*]] = ibis.this @AccessSibling
// CHECK:           %[[VAL_1:.*]] = ibis.port.input @p_b_out : !ibis.portref<out i1>
// CHECK:           %[[VAL_2:.*]] = ibis.port.read %[[VAL_1]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_3:.*]] = ibis.port.input @p_b_in : !ibis.portref<in i1>
// CHECK:           %[[VAL_4:.*]] = ibis.port.read %[[VAL_3]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:         }
ibis.container @AccessSibling {
  %this = ibis.this @AccessSibling
  %sibling = ibis.path [
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<child , @b : !ibis.scoperef<@C>>
  ]
  %c_in = ibis.get_port %sibling, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
  %c_out = ibis.get_port %sibling, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
}

// CHECK-LABEL:   ibis.container @Parent {
// CHECK:           %[[VAL_0:.*]] = ibis.this @Parent
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @a, @AccessSibling
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @p_b_out : !ibis.scoperef<@AccessSibling> -> !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_3:.*]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_4:.*]] = ibis.get_port %[[VAL_1]], @p_b_in : !ibis.scoperef<@AccessSibling> -> !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_5:.*]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           %[[VAL_6:.*]] = ibis.container.instance @b, @C
// CHECK:           %[[VAL_3]] = ibis.get_port %[[VAL_6]], @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_5]] = ibis.get_port %[[VAL_6]], @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECK:         }
ibis.container @Parent {
  %this = ibis.this @Parent
  %a = ibis.container.instance @a, @AccessSibling
  %b = ibis.container.instance @b, @C
}

// -----

// "Full"/recursive case.
// C1 child -> P1 parent -> P2 parent -> C2 child -> C3 child

// CHECK-LABEL:   ibis.container @C1 {
// CHECK:           %[[VAL_0:.*]] = ibis.this @C1
// CHECK:           %[[VAL_1:.*]] = ibis.port.input @p_p_c2_c3_out : !ibis.portref<out i1>
// CHECK:           %[[VAL_2:.*]] = ibis.port.read %[[VAL_1]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_3:.*]] = ibis.port.input @p_p_c2_c3_in : !ibis.portref<in i1>
// CHECK:           %[[VAL_4:.*]] = ibis.port.read %[[VAL_3]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:         }
ibis.container @C1 {
  %this = ibis.this @C1
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
// CHECK:           %[[VAL_0:.*]] = ibis.this @C2
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c3, @C
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = ibis.get_port %[[VAL_1]], @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECK:           %[[VAL_4:.*]] = ibis.port.output @p_p_c2_c3_out : !ibis.portref<out i1>
// CHECK:           %[[VAL_5:.*]] = ibis.port.output @p_p_c2_c3_in : !ibis.portref<in i1>
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_2]] : !ibis.portref<out !ibis.portref<out i1>>
// CHECK:           ibis.port.write %[[VAL_5]], %[[VAL_3]] : !ibis.portref<out !ibis.portref<in i1>>
// CHECK:         }
ibis.container @C2 {
  %this = ibis.this @C2
  %c3 = ibis.container.instance @c3, @C
}

ibis.container @C {
  %this = ibis.this @C
  %in = ibis.port.input @in : i1
  %out = ibis.port.output @out : i1
}

// CHECK-LABEL:   ibis.container @P1 {
// CHECK:           %[[VAL_0:.*]] = ibis.this @P1
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c1, @C1
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @p_p_c2_c3_out : !ibis.scoperef<@C1> -> !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_3:.*]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_4:.*]] = ibis.get_port %[[VAL_1]], @p_p_c2_c3_in : !ibis.scoperef<@C1> -> !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_5:.*]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           %[[VAL_6:.*]] = ibis.port.input @p_p_c2_c3_out : !ibis.portref<out i1>
// CHECK:           %[[VAL_3]] = ibis.port.read %[[VAL_6]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_7:.*]] = ibis.port.input @p_p_c2_c3_in : !ibis.portref<in i1>
// CHECK:           %[[VAL_5]] = ibis.port.read %[[VAL_7]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:         }
ibis.container @P1 {
  %this = ibis.this @P1
  %c1 = ibis.container.instance @c1, @C1
}

// CHECK-LABEL:   ibis.container @P2 {
// CHECK:           %[[VAL_0:.*]] = ibis.this @P2
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @p1, @P1
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @p_p_c2_c3_out : !ibis.scoperef<@P1> -> !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_3:.*]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           %[[VAL_4:.*]] = ibis.get_port %[[VAL_1]], @p_p_c2_c3_in : !ibis.scoperef<@P1> -> !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_5:.*]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           %[[VAL_6:.*]] = ibis.container.instance @c2, @C2
// CHECK:           %[[VAL_7:.*]] = ibis.get_port %[[VAL_6]], @p_p_c2_c3_in : !ibis.scoperef<@C2> -> !ibis.portref<out !ibis.portref<in i1>>
// CHECK:           %[[VAL_5]] = ibis.port.read %[[VAL_7]] : !ibis.portref<out !ibis.portref<in i1>>
// CHECK:           %[[VAL_8:.*]] = ibis.get_port %[[VAL_6]], @p_p_c2_c3_out : !ibis.scoperef<@C2> -> !ibis.portref<out !ibis.portref<out i1>>
// CHECK:           %[[VAL_3]] = ibis.port.read %[[VAL_8]] : !ibis.portref<out !ibis.portref<out i1>>
// CHECK:         }
ibis.container @P2 {
  %this = ibis.this @P2
  %p1 = ibis.container.instance @p1, @P1
  %c2 = ibis.container.instance @c2, @C2
}

// -----

// CHECK-LABEL:   ibis.container @AccessParent {
// CHECK:           %[[VAL_0:.*]] = ibis.this @AccessParent
// CHECK:           %[[VAL_1:.*]] = ibis.port.input @p_out : !ibis.portref<in i1>
// CHECK:           %[[VAL_2:.*]] = ibis.port.read %[[VAL_1]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           %[[VAL_3:.*]] = ibis.port.input @p_in : !ibis.portref<out i1>
// CHECK:           %[[VAL_4:.*]] = ibis.port.read %[[VAL_3]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:         }
ibis.container @AccessParent {
  %this = ibis.this @AccessParent
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
// CHECK:           %[[VAL_0:.*]] = ibis.this @Parent
// CHECK:           %[[VAL_1:.*]] = ibis.port.input @in : i1
// CHECK:           %[[VAL_2:.*]] = ibis.port.read %[[VAL_1]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_3:.*]] = ibis.wire.output @in.rd, %[[VAL_2]] : i1
// CHECK:           %[[VAL_4:.*]] = ibis.port.output @out : i1
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = ibis.wire.input @out.wr : i1
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_6]] : !ibis.portref<out i1>
// CHECK:           %[[VAL_7:.*]] = ibis.container.instance @c, @AccessParent
// CHECK:           %[[VAL_8:.*]] = ibis.get_port %[[VAL_7]], @p_out : !ibis.scoperef<@AccessParent> -> !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           ibis.port.write %[[VAL_8]], %[[VAL_5]] : !ibis.portref<in !ibis.portref<in i1>>
// CHECK:           %[[VAL_9:.*]] = ibis.get_port %[[VAL_7]], @p_in : !ibis.scoperef<@AccessParent> -> !ibis.portref<in !ibis.portref<out i1>>
// CHECK:           ibis.port.write %[[VAL_9]], %[[VAL_3]] : !ibis.portref<in !ibis.portref<out i1>>
// CHECK:         }
ibis.container @Parent {
  %this = ibis.this @Parent
  %in = ibis.port.input @in : i1
  %out = ibis.port.output @out : i1
  %c = ibis.container.instance @c, @AccessParent
}

// -----

ibis.container @C {
  %this = ibis.this @C
  %in = ibis.port.input @in : i1
  %out = ibis.port.output @out : i1
}

// CHECK-LABEL:   hw.module @AccessChildFromHW() {
// CHECK:           %[[VAL_0:.*]] = ibis.container.instance @c, @C
// CHECK:           %[[VAL_1:.*]] = ibis.get_port %[[VAL_0]], @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_0]], @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECK:           hw.output
// CHECK:         }

hw.module @AccessChildFromHW() -> () {
  %c = ibis.container.instance @c, @C
  %c_ref = ibis.path [
    #ibis.step<child , @c : !ibis.scoperef<@C>>
  ]
  %c_in = ibis.get_port %c_ref, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
  %c_out = ibis.get_port %c_ref, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
}
