// RUN: circt-opt --allow-unregistered-dialect --split-input-file --canonicalize --cse %s | FileCheck %s

ibis.design @foo {

// CHECK-LABEL:   ibis.container @GetPortOnThis {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@GetPortOnThis>
// CHECK:           %[[VAL_1:.*]] = ibis.port.input "in" sym @in : i1
// CHECK:           "foo.user"(%[[VAL_1]]) : (!ibis.portref<in i1>) -> ()
// CHECK:         }
ibis.container @GetPortOnThis {
  %this = ibis.this <@GetPortOnThis>
  %p = ibis.port.input "in" sym @in : i1
  %p2 = ibis.get_port %this, @in : !ibis.scoperef<@GetPortOnThis> -> !ibis.portref<in i1>
  "foo.user"(%p2) : (!ibis.portref<in i1>) -> ()
}

}

// -----

ibis.design @foo {

ibis.container @C {
  %this = ibis.this <@C>
}

// CHECK-LABEL:   ibis.container @AccessChild {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@AccessChild>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @c, <@C
// CHECK:           "foo.user"(%[[VAL_1]]) : (!ibis.scoperef<@C>) -> ()
// CHECK:         }
ibis.container @AccessChild {
  %this = ibis.this <@AccessChild>
  %c = ibis.container.instance @c, <@C>
  %c_ref = ibis.path [
    #ibis.step<child , @c : !ibis.scoperef<@C>>
  ]
  "foo.user"(%c_ref) : (!ibis.scoperef<@C>) -> ()
}
}
