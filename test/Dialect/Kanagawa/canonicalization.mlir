// RUN: circt-opt --allow-unregistered-dialect --split-input-file --canonicalize --cse %s | FileCheck %s

kanagawa.design @foo {

// CHECK-LABEL:   kanagawa.container sym @GetPortOnThis {
// CHECK:           %[[VAL_0:.*]] = kanagawa.this <@foo::@GetPortOnThis>
// CHECK:           %[[VAL_1:.*]] = kanagawa.port.input "in" sym @in : i1
// CHECK:           "foo.user"(%[[VAL_1]]) : (!kanagawa.portref<in i1>) -> ()
// CHECK:         }
kanagawa.container sym @GetPortOnThis {
  %this = kanagawa.this <@foo::@GetPortOnThis>
  %p = kanagawa.port.input "in" sym @in : i1
  %p2 = kanagawa.get_port %this, @in : !kanagawa.scoperef<@foo::@GetPortOnThis> -> !kanagawa.portref<in i1>
  "foo.user"(%p2) : (!kanagawa.portref<in i1>) -> ()
}

}

// -----

kanagawa.design @foo {

kanagawa.container sym @C {
  %this = kanagawa.this <@foo::@C>
}

// CHECK-LABEL:   kanagawa.container sym @AccessChild {
// CHECK:           %[[VAL_0:.*]] = kanagawa.this <@foo::@AccessChild>
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @c, <@foo::@C
// CHECK:           "foo.user"(%[[VAL_1]]) : (!kanagawa.scoperef<@foo::@C>) -> ()
// CHECK:         }
kanagawa.container sym @AccessChild {
  %this = kanagawa.this <@foo::@AccessChild>
  %c = kanagawa.container.instance @c, <@foo::@C>
  %c_ref = kanagawa.path [
    #kanagawa.step<child , @c : !kanagawa.scoperef<@foo::@C>>
  ]
  "foo.user"(%c_ref) : (!kanagawa.scoperef<@foo::@C>) -> ()
}
}
