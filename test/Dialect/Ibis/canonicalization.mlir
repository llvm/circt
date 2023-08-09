// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL:   ibis.class @C {
// CHECK:           %[[VAL_0:.*]] = ibis.this @C
// CHECK:           %[[VAL_1:.*]] = ibis.instance @a, @A
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @A_in : !ibis.scoperef<@A> -> !ibis.portref<i1>
// CHECK:         }

ibis.class @A {
  %this = ibis.this @A
  ibis.port.input @A_in : i1
}

ibis.class @C {
  %this = ibis.this @C
  %a = ibis.instance @a, @A
  %a_parent = ibis.get_parent_of_ref %a : !ibis.scoperef<@A> -> !ibis.scoperef
  %a_child = ibis.get_instance_in_ref @a : @A in %a_parent : !ibis.scoperef
  %a_in = ibis.get_port %a_child, @A_in : !ibis.scoperef<@A> -> !ibis.portref<i1>
}
