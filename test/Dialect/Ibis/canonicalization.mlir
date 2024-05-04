// RUN: circt-opt --allow-unregistered-dialect --split-input-file --canonicalize --cse %s | FileCheck %s

ibis.design @foo {

// CHECK-LABEL:   ibis.container @GetPortOnThis {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@foo::@GetPortOnThis>
// CHECK:           %[[VAL_1:.*]] = ibis.port.input "in" sym @in : i1
// CHECK:           "foo.user"(%[[VAL_1]]) : (!ibis.portref<in i1>) -> ()
// CHECK:         }
ibis.container @GetPortOnThis {
  %this = ibis.this <@foo::@GetPortOnThis>
  %p = ibis.port.input "in" sym @in : i1
  %p2 = ibis.get_port %this, @in : !ibis.scoperef<@foo::@GetPortOnThis> -> !ibis.portref<in i1>
  "foo.user"(%p2) : (!ibis.portref<in i1>) -> ()
}

}

// -----

ibis.design @foo {

ibis.container @C {
  %this = ibis.this <@foo::@C>
}

// CHECK-LABEL:   ibis.container @AccessChild {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@foo::@AccessChild>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c, <@foo::@C
// CHECK:           "foo.user"(%[[VAL_1]]) : (!ibis.scoperef<@foo::@C>) -> ()
// CHECK:         }
ibis.container @AccessChild {
  %this = ibis.this <@foo::@AccessChild>
  %c = ibis.container.instance @c, <@foo::@C>
  %c_ref = ibis.path [
    #ibis.step<child , @c : !ibis.scoperef<@foo::@C>>
  ]
  "foo.user"(%c_ref) : (!ibis.scoperef<@foo::@C>) -> ()
}
}
