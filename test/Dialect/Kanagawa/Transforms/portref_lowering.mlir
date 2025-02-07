// RUN: circt-opt --split-input-file --kanagawa-lower-portrefs %s | FileCheck %s

kanagawa.design @D {

kanagawa.container sym @C {
  %in = kanagawa.port.input "in" sym @in : i1
  %out = kanagawa.port.output "out" sym @out : i1
}

// CHECK-LABEL:   kanagawa.container sym @AccessSibling {
// CHECK:           %[[VAL_1:.*]] = kanagawa.port.input "p_b_out" sym @p_b_out : i1
// CHECK:           %[[VAL_2:.*]] = kanagawa.port.output "p_b_in" sym @p_b_in : i1
// CHECK:           kanagawa.port.write %[[VAL_2]], %[[VAL_3:.*]] : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_3]] = kanagawa.port.read %[[VAL_1]] : !kanagawa.portref<in i1>
// CHECK:         }
kanagawa.container sym @AccessSibling {
  %p_b_out = kanagawa.port.input "p_b_out" sym @p_b_out : !kanagawa.portref<out i1>
  %p_b_out_val = kanagawa.port.read %p_b_out : !kanagawa.portref<in !kanagawa.portref<out i1>>
  %p_b_in = kanagawa.port.input "p_b_in" sym @p_b_in : !kanagawa.portref<in i1>
  %p_b_in_val = kanagawa.port.read %p_b_in : !kanagawa.portref<in !kanagawa.portref<in i1>>

  // Loopback to ensure that value replacement is performed.
  %v = kanagawa.port.read %p_b_out_val : !kanagawa.portref<out i1>
  kanagawa.port.write %p_b_in_val, %v : !kanagawa.portref<in i1>
}

// CHECK-LABEL:   kanagawa.container sym @Parent {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @a, <@D::@AccessSibling>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @p_b_out : !kanagawa.scoperef<@D::@AccessSibling> -> !kanagawa.portref<in i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.read %[[VAL_4:.*]] : !kanagawa.portref<out i1>
// CHECK:           kanagawa.port.write %[[VAL_2]], %[[VAL_3]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_5:.*]] = kanagawa.get_port %[[VAL_1]], @p_b_in : !kanagawa.scoperef<@D::@AccessSibling> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_6:.*]] = kanagawa.port.read %[[VAL_5]] : !kanagawa.portref<out i1>
// CHECK:           kanagawa.port.write %[[VAL_7:.*]], %[[VAL_6]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_8:.*]] = kanagawa.container.instance @b, <@D::@C>
// CHECK:           %[[VAL_4]] = kanagawa.get_port %[[VAL_8]], @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_7]] = kanagawa.get_port %[[VAL_8]], @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
// CHECK:         }
kanagawa.container sym @Parent {
  %a = kanagawa.container.instance @a, <@D::@AccessSibling> 
  %a.p_b_out = kanagawa.get_port %a, @p_b_out : !kanagawa.scoperef<@D::@AccessSibling> -> !kanagawa.portref<in !kanagawa.portref<out i1>>
  kanagawa.port.write %a.p_b_out, %b.out : !kanagawa.portref<in !kanagawa.portref<out i1>>
  %a.p_b_in = kanagawa.get_port %a, @p_b_in : !kanagawa.scoperef<@D::@AccessSibling> -> !kanagawa.portref<in !kanagawa.portref<in i1>>
  kanagawa.port.write %a.p_b_in, %b.in : !kanagawa.portref<in !kanagawa.portref<in i1>>
  %b = kanagawa.container.instance @b, <@D::@C> 
  %b.out = kanagawa.get_port %b, @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
  %b.in = kanagawa.get_port %b, @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
}

}

// -----

kanagawa.design @D {


// CHECK-LABEL:   kanagawa.container sym @ParentPortAccess {
// CHECK:           %[[VAL_1:.*]] = kanagawa.port.output "p_in" sym @p_in : i1
// CHECK:           %[[VAL_2:.*]] = kanagawa.port.input "p_out" sym @p_out : i1
// CHECK:         }
kanagawa.container sym @ParentPortAccess {
  %p_in = kanagawa.port.input "p_in" sym @p_in : !kanagawa.portref<in i1>
  %p_in_val = kanagawa.port.read %p_in : !kanagawa.portref<in !kanagawa.portref<in i1>>
  %p_out = kanagawa.port.input "p_out" sym @p_out : !kanagawa.portref<out i1>
  %p_out_val = kanagawa.port.read %p_out : !kanagawa.portref<in !kanagawa.portref<out i1>>
}

// CHECK-LABEL:   kanagawa.container sym @Parent {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @c, <@D::@ParentPortAccess>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @p_in : !kanagawa.scoperef<@D::@ParentPortAccess> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.read %[[VAL_2]] : !kanagawa.portref<out i1>
// CHECK:           kanagawa.port.write %[[VAL_4:.*]], %[[VAL_3]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_5:.*]] = kanagawa.get_port %[[VAL_1]], @p_out : !kanagawa.scoperef<@D::@ParentPortAccess> -> !kanagawa.portref<in i1>
// CHECK:           %[[VAL_6:.*]] = kanagawa.port.read %[[VAL_7:.*]] : !kanagawa.portref<out i1>
// CHECK:           kanagawa.port.write %[[VAL_5]], %[[VAL_6]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_4]] = kanagawa.port.input "in" sym @in : i1
// CHECK:           %[[VAL_7]] = kanagawa.port.output "out" sym @out : i1
// CHECK:         }
kanagawa.container sym @Parent {
  %c = kanagawa.container.instance @c, <@D::@ParentPortAccess> 
  %c.p_in = kanagawa.get_port %c, @p_in : !kanagawa.scoperef<@D::@ParentPortAccess> -> !kanagawa.portref<in !kanagawa.portref<in i1>>
  kanagawa.port.write %c.p_in, %in : !kanagawa.portref<in !kanagawa.portref<in i1>>
  %c.p_out = kanagawa.get_port %c, @p_out : !kanagawa.scoperef<@D::@ParentPortAccess> -> !kanagawa.portref<in !kanagawa.portref<out i1>>
  kanagawa.port.write %c.p_out, %out : !kanagawa.portref<in !kanagawa.portref<out i1>>
  %in = kanagawa.port.input "in" sym @in : i1
  %out = kanagawa.port.output "out" sym @out : i1
}

}

// -----

kanagawa.design @D {


// C1 child -> P1 parent -> P2 parent -> C2 child -> C3 child

// CHECK-LABEL:   kanagawa.container sym @C1 {
// CHECK:           %[[VAL_1:.*]] = kanagawa.port.output "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : i1
// CHECK:           kanagawa.port.write %[[VAL_1]], %[[VAL_2:.*]] : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.input "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : i1
// CHECK:           %[[VAL_2]] = kanagawa.port.read %[[VAL_3]] : !kanagawa.portref<in i1>
// CHECK:         }
kanagawa.container sym @C1 {
  %parent_parent_c2_c3_in = kanagawa.port.input "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : !kanagawa.portref<in i1>
  %parent_parent_c2_c3_out = kanagawa.port.input "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : !kanagawa.portref<out i1>

  // Assignment drivers - unwrap the ports and roundtrip read-write.
  %parent_b_in_unwrapped = kanagawa.port.read %parent_parent_c2_c3_in : !kanagawa.portref<in !kanagawa.portref<in i1>>
  %parent_b_out_unwrapped = kanagawa.port.read %parent_parent_c2_c3_out : !kanagawa.portref<in !kanagawa.portref<out i1>>
  %parent_b_out_value = kanagawa.port.read %parent_b_out_unwrapped : !kanagawa.portref<out i1>
  kanagawa.port.write %parent_b_in_unwrapped, %parent_b_out_value : !kanagawa.portref<in i1>
}

// CHECK-LABEL:   kanagawa.container sym @C2 {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @c3, <@D::@C>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.get_port %[[VAL_1]], @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_4:.*]] = kanagawa.port.input "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : i1
// CHECK:           %[[VAL_5:.*]] = kanagawa.port.read %[[VAL_4]] : !kanagawa.portref<in i1>
// CHECK:           kanagawa.port.write %[[VAL_2]], %[[VAL_5]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_6:.*]] = kanagawa.port.output "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : i1
// CHECK:           %[[VAL_7:.*]] = kanagawa.port.read %[[VAL_3]] : !kanagawa.portref<out i1>
// CHECK:           kanagawa.port.write %[[VAL_6]], %[[VAL_7]] : !kanagawa.portref<out i1>
// CHECK:         }
kanagawa.container sym @C2 {
  %c3 = kanagawa.container.instance @c3, <@D::@C> 
  %c3.in = kanagawa.get_port %c3, @in : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<in i1>
  %c3.out = kanagawa.get_port %c3, @out : !kanagawa.scoperef<@D::@C> -> !kanagawa.portref<out i1>
  %parent_parent_c2_c3_in = kanagawa.port.output "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : !kanagawa.portref<in i1>
  %parent_parent_c2_c3_out = kanagawa.port.output "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : !kanagawa.portref<out i1>
  kanagawa.port.write %parent_parent_c2_c3_in, %c3.in : !kanagawa.portref<out !kanagawa.portref<in i1>>
  kanagawa.port.write %parent_parent_c2_c3_out, %c3.out : !kanagawa.portref<out !kanagawa.portref<out i1>>
}
kanagawa.container sym @C {
  %in = kanagawa.port.input "in" sym @in : i1
  %out = kanagawa.port.output "out" sym @out : i1
}

// CHECK-LABEL:   kanagawa.container sym @P1 {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @c1, <@D::@C1>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @parent_parent_c2_c3_in : !kanagawa.scoperef<@D::@C1> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.read %[[VAL_2]] : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_4:.*]] = kanagawa.get_port %[[VAL_1]], @parent_parent_c2_c3_out : !kanagawa.scoperef<@D::@C1> -> !kanagawa.portref<in i1>
// CHECK:           %[[VAL_5:.*]] = kanagawa.port.read %[[VAL_6:.*]] : !kanagawa.portref<in i1>
// CHECK:           kanagawa.port.write %[[VAL_4]], %[[VAL_5]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_7:.*]] = kanagawa.port.output "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : i1
// CHECK:           kanagawa.port.write %[[VAL_7]], %[[VAL_3]] : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_6]] = kanagawa.port.input "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : i1
// CHECK:         }
kanagawa.container sym @P1 {
  %c1 = kanagawa.container.instance @c1, <@D::@C1> 
  %c1.parent_parent_c2_c3_in = kanagawa.get_port %c1, @parent_parent_c2_c3_in : !kanagawa.scoperef<@D::@C1> -> !kanagawa.portref<in !kanagawa.portref<in i1>>
  kanagawa.port.write %c1.parent_parent_c2_c3_in, %0 : !kanagawa.portref<in !kanagawa.portref<in i1>>
  %c1.parent_parent_c2_c3_out = kanagawa.get_port %c1, @parent_parent_c2_c3_out : !kanagawa.scoperef<@D::@C1> -> !kanagawa.portref<in !kanagawa.portref<out i1>>
  kanagawa.port.write %c1.parent_parent_c2_c3_out, %1 : !kanagawa.portref<in !kanagawa.portref<out i1>>
  %parent_parent_c2_c3_in = kanagawa.port.input "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : !kanagawa.portref<in i1>
  %0 = kanagawa.port.read %parent_parent_c2_c3_in : !kanagawa.portref<in !kanagawa.portref<in i1>>
  %parent_parent_c2_c3_out = kanagawa.port.input "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : !kanagawa.portref<out i1>
  %1 = kanagawa.port.read %parent_parent_c2_c3_out : !kanagawa.portref<in !kanagawa.portref<out i1>>
}

// CHECK-LABEL:   kanagawa.container sym @P2 {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @p1, <@D::@P1>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @parent_parent_c2_c3_in : !kanagawa.scoperef<@D::@P1> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.read %[[VAL_2]] : !kanagawa.portref<out i1>
// CHECK:           kanagawa.port.write %[[VAL_4:.*]], %[[VAL_3]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_5:.*]] = kanagawa.get_port %[[VAL_1]], @parent_parent_c2_c3_out : !kanagawa.scoperef<@D::@P1> -> !kanagawa.portref<in i1>
// CHECK:           %[[VAL_6:.*]] = kanagawa.port.read %[[VAL_7:.*]] : !kanagawa.portref<out i1>
// CHECK:           kanagawa.port.write %[[VAL_5]], %[[VAL_6]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_8:.*]] = kanagawa.container.instance @c2, <@D::@C2>
// CHECK:           %[[VAL_7]] = kanagawa.get_port %[[VAL_8]], @parent_parent_c2_c3_out : !kanagawa.scoperef<@D::@C2> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_4]] = kanagawa.port.input "parent_parent_c2_c3_in_fw" sym @parent_parent_c2_c3_in_fw : i1
// CHECK:           %[[VAL_9:.*]] = kanagawa.port.read %[[VAL_4]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_10:.*]] = kanagawa.get_port %[[VAL_8]], @parent_parent_c2_c3_in : !kanagawa.scoperef<@D::@C2> -> !kanagawa.portref<in i1>
// CHECK:           kanagawa.port.write %[[VAL_10]], %[[VAL_9]] : !kanagawa.portref<in i1>
// CHECK:         }
kanagawa.container sym @P2 {
  %p1 = kanagawa.container.instance @p1, <@D::@P1> 
  %p1.parent_parent_c2_c3_in = kanagawa.get_port %p1, @parent_parent_c2_c3_in : !kanagawa.scoperef<@D::@P1> -> !kanagawa.portref<in !kanagawa.portref<in i1>>
  kanagawa.port.write %p1.parent_parent_c2_c3_in, %1 : !kanagawa.portref<in !kanagawa.portref<in i1>>
  %p1.parent_parent_c2_c3_out = kanagawa.get_port %p1, @parent_parent_c2_c3_out : !kanagawa.scoperef<@D::@P1> -> !kanagawa.portref<in !kanagawa.portref<out i1>>
  kanagawa.port.write %p1.parent_parent_c2_c3_out, %0 : !kanagawa.portref<in !kanagawa.portref<out i1>>
  %c2 = kanagawa.container.instance @c2, <@D::@C2> 
  %c2.parent_parent_c2_c3_out = kanagawa.get_port %c2, @parent_parent_c2_c3_out : !kanagawa.scoperef<@D::@C2> -> !kanagawa.portref<out !kanagawa.portref<out i1>>
  %0 = kanagawa.port.read %c2.parent_parent_c2_c3_out : !kanagawa.portref<out !kanagawa.portref<out i1>>
  %c2.parent_parent_c2_c3_in = kanagawa.get_port %c2, @parent_parent_c2_c3_in : !kanagawa.scoperef<@D::@C2> -> !kanagawa.portref<out !kanagawa.portref<in i1>>
  %1 = kanagawa.port.read %c2.parent_parent_c2_c3_in : !kanagawa.portref<out !kanagawa.portref<in i1>>
}

}

// -----

kanagawa.design @D {


// CHECK-LABEL:   kanagawa.container sym @AccessParent {
// CHECK:           %[[VAL_1:.*]] = kanagawa.port.output "p_out" sym @p_out : i1
// CHECK:           %[[VAL_2:.*]] = kanagawa.port.input "p_in" sym @p_in : i1
// CHECK:         }
kanagawa.container sym @AccessParent {
  %p_out = kanagawa.port.input "p_out" sym @p_out : !kanagawa.portref<in i1>
  %p_out.val = kanagawa.port.read %p_out : !kanagawa.portref<in !kanagawa.portref<in i1>>
  %p_in = kanagawa.port.input "p_in" sym @p_in : !kanagawa.portref<out i1>
  %p_in.val = kanagawa.port.read %p_in : !kanagawa.portref<in !kanagawa.portref<out i1>>
}

// CHECK-LABEL:   kanagawa.container sym @Parent {
// CHECK:           %[[VAL_1:.*]] = kanagawa.port.input "in" sym @in : i1
// CHECK:           %[[VAL_2:.*]] = kanagawa.port.read %[[VAL_1]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.wire.output @in.rd, %[[VAL_2]] : i1
// CHECK:           %[[VAL_4:.*]] = kanagawa.port.output "out" sym @out : i1
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = kanagawa.wire.input @out.wr : i1
// CHECK:           kanagawa.port.write %[[VAL_4]], %[[VAL_6]] : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_7:.*]] = kanagawa.container.instance @c, <@D::@AccessParent>
// CHECK:           %[[VAL_8:.*]] = kanagawa.get_port %[[VAL_7]], @p_out : !kanagawa.scoperef<@D::@AccessParent> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_9:.*]] = kanagawa.port.read %[[VAL_8]] : !kanagawa.portref<out i1>
// CHECK:           kanagawa.port.write %[[VAL_5]], %[[VAL_9]] : !kanagawa.portref<in i1>
// CHECK:           %[[VAL_10:.*]] = kanagawa.get_port %[[VAL_7]], @p_in : !kanagawa.scoperef<@D::@AccessParent> -> !kanagawa.portref<in i1>
// CHECK:           %[[VAL_11:.*]] = kanagawa.port.read %[[VAL_3]] : !kanagawa.portref<out i1>
// CHECK:           kanagawa.port.write %[[VAL_10]], %[[VAL_11]] : !kanagawa.portref<in i1>
// CHECK:         }
kanagawa.container sym @Parent {
  %in = kanagawa.port.input "in" sym @in : i1
  %in.val = kanagawa.port.read %in : !kanagawa.portref<in i1>
  %in.rd = kanagawa.wire.output @in.rd, %in.val : i1
  %out = kanagawa.port.output "out" sym @out : i1
  %out.wr, %out.wr.out = kanagawa.wire.input @out.wr : i1
  kanagawa.port.write %out, %out.wr.out : !kanagawa.portref<out i1>
  %c = kanagawa.container.instance @c, <@D::@AccessParent> 
  %c.p_out.ref = kanagawa.get_port %c, @p_out : !kanagawa.scoperef<@D::@AccessParent> -> !kanagawa.portref<in !kanagawa.portref<in i1>>
  kanagawa.port.write %c.p_out.ref, %out.wr : !kanagawa.portref<in !kanagawa.portref<in i1>>
  %c.p_in.ref = kanagawa.get_port %c, @p_in : !kanagawa.scoperef<@D::@AccessParent> -> !kanagawa.portref<in !kanagawa.portref<out i1>>
  kanagawa.port.write %c.p_in.ref, %in.rd : !kanagawa.portref<in !kanagawa.portref<out i1>>
}

}
