// RUN: circt-opt --split-input-file --allow-unregistered-dialect --kanagawa-tunneling --verify-diagnostics %s

kanagawa.design @foo {
kanagawa.container sym @Parent {
  %in = kanagawa.port.input "in" sym @in : i1
}

kanagawa.container sym @Orphan {
  // expected-error @+2 {{'kanagawa.path' op cannot tunnel up from "Orphan" because it has no uses}}
  // expected-error @+1 {{failed to legalize operation 'kanagawa.path' that was explicitly marked illegal}}
  %parent = kanagawa.path [
    #kanagawa.step<parent : !kanagawa.scoperef<@foo::@Parent>>
  ]

  %p = kanagawa.get_port %parent, @in : !kanagawa.scoperef<@foo::@Parent> -> !kanagawa.portref<in i1>
}
}
// -----

kanagawa.design @foo {
kanagawa.container sym @Parent {
  %mc = kanagawa.container.instance @mc, <@foo::@MissingChild>
}

kanagawa.container sym @Child {
  %in = kanagawa.port.input "in" sym @in : i1
}

kanagawa.container sym @MissingChild {
  // expected-error @+2 {{'kanagawa.path' op expected an instance named @c in @Parent but found none}}
  // expected-error @+1 {{failed to legalize operation 'kanagawa.path' that was explicitly marked illegal}}
  %parent = kanagawa.path [
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<child , @c : !kanagawa.scoperef<@foo::@Child>>
  ]
  %p = kanagawa.get_port %parent, @in : !kanagawa.scoperef<@foo::@Child> -> !kanagawa.portref<in i1>
}
}
