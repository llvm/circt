// RUN: circt-opt --split-input-file --ibis-lower-portrefs %s | FileCheck %s

ibis.design @D {

ibis.container @C {
  %this = ibis.this <@C> 
  %in = ibis.port.input "in" sym @in : i1
  %out = ibis.port.output "out" sym @out : i1
}

// CHECK-LABEL:   ibis.container @AccessSibling {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@AccessSibling>
// CHECK:           %[[VAL_1:.*]] = ibis.port.input "p_b_out" sym @p_b_out : i1
// CHECK:           %[[VAL_2:.*]] = ibis.port.output "p_b_in" sym @p_b_in : i1
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_3:.*]] : !ibis.portref<out i1>
// CHECK:           %[[VAL_3]] = ibis.port.read %[[VAL_1]] : !ibis.portref<in i1>
// CHECK:         }
ibis.container @AccessSibling {
  %this = ibis.this <@AccessSibling> 
  %p_b_out = ibis.port.input "p_b_out" sym @p_b_out : !ibis.portref<out i1>
  %p_b_out_val = ibis.port.read %p_b_out : !ibis.portref<in !ibis.portref<out i1>>
  %p_b_in = ibis.port.input "p_b_in" sym @p_b_in : !ibis.portref<in i1>
  %p_b_in_val = ibis.port.read %p_b_in : !ibis.portref<in !ibis.portref<in i1>>

  // Loopback to ensure that value replacement is performed.
  %v = ibis.port.read %p_b_out_val : !ibis.portref<out i1>
  ibis.port.write %p_b_in_val, %v : !ibis.portref<in i1>
}

// CHECK-LABEL:   ibis.container @Parent {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@Parent>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @a, <@AccessSibling>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @p_b_out : !ibis.scoperef<@AccessSibling> -> !ibis.portref<in i1>
// CHECK:           %[[VAL_3:.*]] = ibis.port.read %[[VAL_4:.*]] : !ibis.portref<out i1>
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_3]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_5:.*]] = ibis.get_port %[[VAL_1]], @p_b_in : !ibis.scoperef<@AccessSibling> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_6:.*]] = ibis.port.read %[[VAL_5]] : !ibis.portref<out i1>
// CHECK:           ibis.port.write %[[VAL_7:.*]], %[[VAL_6]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_8:.*]] = ibis.container.instance @b, <@C>
// CHECK:           %[[VAL_4]] = ibis.get_port %[[VAL_8]], @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_7]] = ibis.get_port %[[VAL_8]], @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECK:         }
ibis.container @Parent {
  %this = ibis.this <@Parent> 
  %a = ibis.container.instance @a, <@AccessSibling> 
  %a.p_b_out = ibis.get_port %a, @p_b_out : !ibis.scoperef<@AccessSibling> -> !ibis.portref<in !ibis.portref<out i1>>
  ibis.port.write %a.p_b_out, %b.out : !ibis.portref<in !ibis.portref<out i1>>
  %a.p_b_in = ibis.get_port %a, @p_b_in : !ibis.scoperef<@AccessSibling> -> !ibis.portref<in !ibis.portref<in i1>>
  ibis.port.write %a.p_b_in, %b.in : !ibis.portref<in !ibis.portref<in i1>>
  %b = ibis.container.instance @b, <@C> 
  %b.out = ibis.get_port %b, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
  %b.in = ibis.get_port %b, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
}

}

// -----

ibis.design @D {


// CHECK-LABEL:   ibis.container @ParentPortAccess {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@ParentPortAccess>
// CHECK:           %[[VAL_1:.*]] = ibis.port.output "p_in" sym @p_in : i1
// CHECK:           %[[VAL_2:.*]] = ibis.port.input "p_out" sym @p_out : i1
// CHECK:         }
ibis.container @ParentPortAccess {
  %this = ibis.this <@ParentPortAccess> 
  %p_in = ibis.port.input "p_in" sym @p_in : !ibis.portref<in i1>
  %p_in_val = ibis.port.read %p_in : !ibis.portref<in !ibis.portref<in i1>>
  %p_out = ibis.port.input "p_out" sym @p_out : !ibis.portref<out i1>
  %p_out_val = ibis.port.read %p_out : !ibis.portref<in !ibis.portref<out i1>>
}

// CHECK-LABEL:   ibis.container @Parent {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@Parent>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c, <@ParentPortAccess>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @p_in : !ibis.scoperef<@ParentPortAccess> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = ibis.port.read %[[VAL_2]] : !ibis.portref<out i1>
// CHECK:           ibis.port.write %[[VAL_4:.*]], %[[VAL_3]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_5:.*]] = ibis.get_port %[[VAL_1]], @p_out : !ibis.scoperef<@ParentPortAccess> -> !ibis.portref<in i1>
// CHECK:           %[[VAL_6:.*]] = ibis.port.read %[[VAL_7:.*]] : !ibis.portref<out i1>
// CHECK:           ibis.port.write %[[VAL_5]], %[[VAL_6]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_4]] = ibis.port.input "in" sym @in : i1
// CHECK:           %[[VAL_7]] = ibis.port.output "out" sym @out : i1
// CHECK:         }
ibis.container @Parent {
  %this = ibis.this <@Parent> 
  %c = ibis.container.instance @c, <@ParentPortAccess> 
  %c.p_in = ibis.get_port %c, @p_in : !ibis.scoperef<@ParentPortAccess> -> !ibis.portref<in !ibis.portref<in i1>>
  ibis.port.write %c.p_in, %in : !ibis.portref<in !ibis.portref<in i1>>
  %c.p_out = ibis.get_port %c, @p_out : !ibis.scoperef<@ParentPortAccess> -> !ibis.portref<in !ibis.portref<out i1>>
  ibis.port.write %c.p_out, %out : !ibis.portref<in !ibis.portref<out i1>>
  %in = ibis.port.input "in" sym @in : i1
  %out = ibis.port.output "out" sym @out : i1
}

}

// -----

ibis.design @D {


// C1 child -> P1 parent -> P2 parent -> C2 child -> C3 child

// CHECK-LABEL:   ibis.container @C1 {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@C1>
// CHECK:           %[[VAL_1:.*]] = ibis.port.output "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : i1
// CHECK:           ibis.port.write %[[VAL_1]], %[[VAL_2:.*]] : !ibis.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = ibis.port.input "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : i1
// CHECK:           %[[VAL_2]] = ibis.port.read %[[VAL_3]] : !ibis.portref<in i1>
// CHECK:         }
ibis.container @C1 {
  %this = ibis.this <@C1> 
  %parent_parent_c2_c3_in = ibis.port.input "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : !ibis.portref<in i1>
  %parent_parent_c2_c3_out = ibis.port.input "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : !ibis.portref<out i1>

  // Assignment drivers - unwrap the ports and roundtrip read-write.
  %parent_b_in_unwrapped = ibis.port.read %parent_parent_c2_c3_in : !ibis.portref<in !ibis.portref<in i1>>
  %parent_b_out_unwrapped = ibis.port.read %parent_parent_c2_c3_out : !ibis.portref<in !ibis.portref<out i1>>
  %parent_b_out_value = ibis.port.read %parent_b_out_unwrapped : !ibis.portref<out i1>
  ibis.port.write %parent_b_in_unwrapped, %parent_b_out_value : !ibis.portref<in i1>
}

// CHECK-LABEL:   ibis.container @C2 {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@C2>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c3, <@C>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECK:           %[[VAL_3:.*]] = ibis.get_port %[[VAL_1]], @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_4:.*]] = ibis.port.input "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : i1
// CHECK:           %[[VAL_5:.*]] = ibis.port.read %[[VAL_4]] : !ibis.portref<in i1>
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_5]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_6:.*]] = ibis.port.output "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : i1
// CHECK:           %[[VAL_7:.*]] = ibis.port.read %[[VAL_3]] : !ibis.portref<out i1>
// CHECK:           ibis.port.write %[[VAL_6]], %[[VAL_7]] : !ibis.portref<out i1>
// CHECK:         }
ibis.container @C2 {
  %this = ibis.this <@C2> 
  %c3 = ibis.container.instance @c3, <@C> 
  %c3.in = ibis.get_port %c3, @in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
  %c3.out = ibis.get_port %c3, @out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
  %parent_parent_c2_c3_in = ibis.port.output "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : !ibis.portref<in i1>
  %parent_parent_c2_c3_out = ibis.port.output "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : !ibis.portref<out i1>
  ibis.port.write %parent_parent_c2_c3_in, %c3.in : !ibis.portref<out !ibis.portref<in i1>>
  ibis.port.write %parent_parent_c2_c3_out, %c3.out : !ibis.portref<out !ibis.portref<out i1>>
}
ibis.container @C {
  %this = ibis.this <@C> 
  %in = ibis.port.input "in" sym @in : i1
  %out = ibis.port.output "out" sym @out : i1
}

// CHECK-LABEL:   ibis.container @P1 {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@P1>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c1, <@C1>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @parent_parent_c2_c3_in : !ibis.scoperef<@C1> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = ibis.port.read %[[VAL_2]] : !ibis.portref<out i1>
// CHECK:           %[[VAL_4:.*]] = ibis.get_port %[[VAL_1]], @parent_parent_c2_c3_out : !ibis.scoperef<@C1> -> !ibis.portref<in i1>
// CHECK:           %[[VAL_5:.*]] = ibis.port.read %[[VAL_6:.*]] : !ibis.portref<in i1>
// CHECK:           ibis.port.write %[[VAL_4]], %[[VAL_5]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_7:.*]] = ibis.port.output "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : i1
// CHECK:           ibis.port.write %[[VAL_7]], %[[VAL_3]] : !ibis.portref<out i1>
// CHECK:           %[[VAL_6]] = ibis.port.input "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : i1
// CHECK:         }
ibis.container @P1 {
  %this = ibis.this <@P1> 
  %c1 = ibis.container.instance @c1, <@C1> 
  %c1.parent_parent_c2_c3_in = ibis.get_port %c1, @parent_parent_c2_c3_in : !ibis.scoperef<@C1> -> !ibis.portref<in !ibis.portref<in i1>>
  ibis.port.write %c1.parent_parent_c2_c3_in, %0 : !ibis.portref<in !ibis.portref<in i1>>
  %c1.parent_parent_c2_c3_out = ibis.get_port %c1, @parent_parent_c2_c3_out : !ibis.scoperef<@C1> -> !ibis.portref<in !ibis.portref<out i1>>
  ibis.port.write %c1.parent_parent_c2_c3_out, %1 : !ibis.portref<in !ibis.portref<out i1>>
  %parent_parent_c2_c3_in = ibis.port.input "parent_parent_c2_c3_in" sym @parent_parent_c2_c3_in : !ibis.portref<in i1>
  %0 = ibis.port.read %parent_parent_c2_c3_in : !ibis.portref<in !ibis.portref<in i1>>
  %parent_parent_c2_c3_out = ibis.port.input "parent_parent_c2_c3_out" sym @parent_parent_c2_c3_out : !ibis.portref<out i1>
  %1 = ibis.port.read %parent_parent_c2_c3_out : !ibis.portref<in !ibis.portref<out i1>>
}

// CHECK-LABEL:   ibis.container @P2 {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@P2>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @p1, <@P1>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @parent_parent_c2_c3_in : !ibis.scoperef<@P1> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = ibis.port.read %[[VAL_2]] : !ibis.portref<out i1>
// CHECK:           ibis.port.write %[[VAL_4:.*]], %[[VAL_3]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_5:.*]] = ibis.get_port %[[VAL_1]], @parent_parent_c2_c3_out : !ibis.scoperef<@P1> -> !ibis.portref<in i1>
// CHECK:           %[[VAL_6:.*]] = ibis.port.read %[[VAL_7:.*]] : !ibis.portref<out i1>
// CHECK:           ibis.port.write %[[VAL_5]], %[[VAL_6]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_8:.*]] = ibis.container.instance @c2, <@C2>
// CHECK:           %[[VAL_7]] = ibis.get_port %[[VAL_8]], @parent_parent_c2_c3_out : !ibis.scoperef<@C2> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_4]] = ibis.port.input "parent_parent_c2_c3_in_fw" sym @parent_parent_c2_c3_in_fw : i1
// CHECK:           %[[VAL_9:.*]] = ibis.port.read %[[VAL_4]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_10:.*]] = ibis.get_port %[[VAL_8]], @parent_parent_c2_c3_in : !ibis.scoperef<@C2> -> !ibis.portref<in i1>
// CHECK:           ibis.port.write %[[VAL_10]], %[[VAL_9]] : !ibis.portref<in i1>
// CHECK:         }
ibis.container @P2 {
  %this = ibis.this <@P2> 
  %p1 = ibis.container.instance @p1, <@P1> 
  %p1.parent_parent_c2_c3_in = ibis.get_port %p1, @parent_parent_c2_c3_in : !ibis.scoperef<@P1> -> !ibis.portref<in !ibis.portref<in i1>>
  ibis.port.write %p1.parent_parent_c2_c3_in, %1 : !ibis.portref<in !ibis.portref<in i1>>
  %p1.parent_parent_c2_c3_out = ibis.get_port %p1, @parent_parent_c2_c3_out : !ibis.scoperef<@P1> -> !ibis.portref<in !ibis.portref<out i1>>
  ibis.port.write %p1.parent_parent_c2_c3_out, %0 : !ibis.portref<in !ibis.portref<out i1>>
  %c2 = ibis.container.instance @c2, <@C2> 
  %c2.parent_parent_c2_c3_out = ibis.get_port %c2, @parent_parent_c2_c3_out : !ibis.scoperef<@C2> -> !ibis.portref<out !ibis.portref<out i1>>
  %0 = ibis.port.read %c2.parent_parent_c2_c3_out : !ibis.portref<out !ibis.portref<out i1>>
  %c2.parent_parent_c2_c3_in = ibis.get_port %c2, @parent_parent_c2_c3_in : !ibis.scoperef<@C2> -> !ibis.portref<out !ibis.portref<in i1>>
  %1 = ibis.port.read %c2.parent_parent_c2_c3_in : !ibis.portref<out !ibis.portref<in i1>>
}

}

// -----

ibis.design @D {


// CHECK-LABEL:   ibis.container @AccessParent {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@AccessParent>
// CHECK:           %[[VAL_1:.*]] = ibis.port.output "p_out" sym @p_out : i1
// CHECK:           %[[VAL_2:.*]] = ibis.port.input "p_in" sym @p_in : i1
// CHECK:         }
ibis.container @AccessParent {
  %this = ibis.this <@AccessParent> 
  %p_out = ibis.port.input "p_out" sym @p_out : !ibis.portref<in i1>
  %p_out.val = ibis.port.read %p_out : !ibis.portref<in !ibis.portref<in i1>>
  %p_in = ibis.port.input "p_in" sym @p_in : !ibis.portref<out i1>
  %p_in.val = ibis.port.read %p_in : !ibis.portref<in !ibis.portref<out i1>>
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
// CHECK:           %[[VAL_8:.*]] = ibis.get_port %[[VAL_7]], @p_out : !ibis.scoperef<@AccessParent> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_9:.*]] = ibis.port.read %[[VAL_8]] : !ibis.portref<out i1>
// CHECK:           ibis.port.write %[[VAL_5]], %[[VAL_9]] : !ibis.portref<in i1>
// CHECK:           %[[VAL_10:.*]] = ibis.get_port %[[VAL_7]], @p_in : !ibis.scoperef<@AccessParent> -> !ibis.portref<in i1>
// CHECK:           %[[VAL_11:.*]] = ibis.port.read %[[VAL_3]] : !ibis.portref<out i1>
// CHECK:           ibis.port.write %[[VAL_10]], %[[VAL_11]] : !ibis.portref<in i1>
// CHECK:         }
ibis.container @Parent {
  %this = ibis.this <@Parent> 
  %in = ibis.port.input "in" sym @in : i1
  %in.val = ibis.port.read %in : !ibis.portref<in i1>
  %in.rd = ibis.wire.output @in.rd, %in.val : i1
  %out = ibis.port.output "out" sym @out : i1
  %out.wr, %out.wr.out = ibis.wire.input @out.wr : i1
  ibis.port.write %out, %out.wr.out : !ibis.portref<out i1>
  %c = ibis.container.instance @c, <@AccessParent> 
  %c.p_out.ref = ibis.get_port %c, @p_out : !ibis.scoperef<@AccessParent> -> !ibis.portref<in !ibis.portref<in i1>>
  ibis.port.write %c.p_out.ref, %out.wr : !ibis.portref<in !ibis.portref<in i1>>
  %c.p_in.ref = ibis.get_port %c, @p_in : !ibis.scoperef<@AccessParent> -> !ibis.portref<in !ibis.portref<out i1>>
  ibis.port.write %c.p_in.ref, %in.rd : !ibis.portref<in !ibis.portref<out i1>>
}

}
