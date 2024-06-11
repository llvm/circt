// RUN: circt-opt --split-input-file --allow-unregistered-dialect --ibis-tunneling --verify-diagnostics %s

ibis.design @foo {
ibis.container sym @Parent {
  %this = ibis.this <@foo::@Parent>
  %in = ibis.port.input "in" sym @in : i1
}

ibis.container sym @Orphan {
  %this = ibis.this <@foo::@Orphan>
  // expected-error @+2 {{'ibis.path' op cannot tunnel up from "Orphan" because it has no uses}}
  // expected-error @+1 {{failed to legalize operation 'ibis.path' that was explicitly marked illegal}}
  %parent = ibis.path [
    #ibis.step<parent : !ibis.scoperef<@foo::@Parent>>
  ]

  %p = ibis.get_port %parent, @in : !ibis.scoperef<@foo::@Parent> -> !ibis.portref<in i1>
}
}
// -----

ibis.design @foo {
ibis.container sym @Parent {
  %this = ibis.this <@foo::@Parent>
  %mc = ibis.container.instance @mc, <@foo::@MissingChild>
}

ibis.container sym @Child {
  %this = ibis.this <@foo::@Child>
  %in = ibis.port.input "in" sym @in : i1
}

ibis.container sym @MissingChild {
  %this = ibis.this <@foo::@MissingChild>
  // expected-error @+2 {{'ibis.path' op expected an instance named @c in @Parent but found none}}
  // expected-error @+1 {{failed to legalize operation 'ibis.path' that was explicitly marked illegal}}
  %parent = ibis.path [
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<child , @c : !ibis.scoperef<@foo::@Child>>
  ]
  %p = ibis.get_port %parent, @in : !ibis.scoperef<@foo::@Child> -> !ibis.portref<in i1>
}
}
