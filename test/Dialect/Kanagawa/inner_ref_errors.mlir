// RUN: circt-opt --hw-verify-irn --split-input-file --verify-diagnostics %s

kanagawa.design @foo {
kanagawa.class sym @InvalidGetVar2 {
  kanagawa.var @var : memref<i32>
  kanagawa.method @foo()  {
    %parent = kanagawa.path [
      #kanagawa.step<parent : !kanagawa.scoperef<@foo::@InvalidGetVar2>>
    ]
    // expected-error @+1 {{'kanagawa.get_var' op dereferenced type ('memref<i1>') must match variable type ('memref<i32>')}}
    %var = kanagawa.get_var %parent, @var : !kanagawa.scoperef<@foo::@InvalidGetVar2> -> memref<i1>
    kanagawa.return
  }
}
}

// -----

kanagawa.design @foo {
kanagawa.class sym @InvalidGetVar {
  kanagawa.var @var : memref<i32>
  kanagawa.method @foo()  {
    %parent = kanagawa.path [
      #kanagawa.step<parent : !kanagawa.scoperef<@foo::@InvalidGetVar>>
    ]
    // expected-error @+1 {{'kanagawa.get_var' op result #0 must be memref of any type values, but got 'i32'}}
    %var = kanagawa.get_var %parent, @var : !kanagawa.scoperef<@foo::@InvalidGetVar> -> i32
    kanagawa.return
  }
}
}


// -----

kanagawa.design @foo {
kanagawa.class sym @PathStepNonExistingChild {
  // expected-error @+1 {{'kanagawa.path' op kanagawa.step scoperef symbol '@A' does not exist}}
  %p = kanagawa.path [#kanagawa.step<child , @a : !kanagawa.scoperef<@foo::@A>>]
}
}

// -----

kanagawa.design @foo {
kanagawa.class sym @PathStepNonExistingParent {
  // expected-error @+1 {{'kanagawa.path' op last kanagawa.step in path must specify a symbol for the scoperef}}
  %p = kanagawa.path [#kanagawa.step<parent : !kanagawa.scoperef>]
}
}
