// RUN: circt-opt --split-input-file --allow-unregistered-dialect --ibis-tunneling --verify-diagnostics %s

ibis.design @foo {
ibis.container @Parent {
  %this = ibis.this <@Parent>
  %in = ibis.port.input "in" sym @in : i1
}

ibis.container @Orphan {
  %this = ibis.this <@Orphan>
  // expected-error @+2 {{'ibis.path' op cannot tunnel up from "Orphan" because it has no uses}}
  // expected-error @+1 {{failed to legalize operation 'ibis.path' that was explicitly marked illegal}}
  %parent = ibis.path [
    #ibis.step<parent : !ibis.scoperef<@Parent>>
  ]

  %p = ibis.get_port %parent, @in : !ibis.scoperef<@Parent> -> !ibis.portref<in i1>
}
}
// -----

ibis.design @foo {
ibis.container @Parent {
  %this = ibis.this <@Parent>
  %mc = ibis.container.instance @mc, <@MissingChild>
}

ibis.container @Child {
  %this = ibis.this <@Child>
  %in = ibis.port.input "in" sym @in : i1
}

ibis.container @MissingChild {
  %this = ibis.this <@MissingChild>
  // expected-error @+2 {{'ibis.path' op expected an instance named @c in @Parent but found none}}
  // expected-error @+1 {{failed to legalize operation 'ibis.path' that was explicitly marked illegal}}
  %parent = ibis.path [
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<child , @c : !ibis.scoperef<@Child>>
  ]
  %p = ibis.get_port %parent, @in : !ibis.scoperef<@Child> -> !ibis.portref<in i1>
}
}
