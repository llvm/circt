// RUN: circt-opt --hw-verify-irn --split-input-file --verify-diagnostics %s

kanagawa.design @foo {
kanagawa.class sym @MissingPort {
  %this = kanagawa.this <@foo::@MissingPort>
  // expected-error @+1 {{'kanagawa.get_port' op port '@C_in' does not exist in @MissingPort}}
  %c_in = kanagawa.get_port %this, @C_in : !kanagawa.scoperef<@foo::@MissingPort> -> !kanagawa.portref<in i1>
}
}

// -----

kanagawa.design @foo {
kanagawa.class sym @InvalidGetVar2 {
  %this = kanagawa.this <@foo::@InvalidGetVar2>
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
  %this = kanagawa.this <@foo::@InvalidGetVar>
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
kanagawa.class sym @PortTypeMismatch {
  %this = kanagawa.this <@foo::@PortTypeMismatch>
  kanagawa.port.input "in" sym @in : i1
  // expected-error @+1 {{'kanagawa.get_port' op symbol '@in' refers to a port of type 'i1', but this op has type 'i2'}}
  %c_in = kanagawa.get_port %this, @in : !kanagawa.scoperef<@foo::@PortTypeMismatch> -> !kanagawa.portref<in i2>
}
}

// -----

kanagawa.design @foo {
kanagawa.class sym @PathStepNonExistingChild {
  %this = kanagawa.this <@foo::@PathStepNonExistingChild>
  // expected-error @+1 {{'kanagawa.path' op kanagawa.step scoperef symbol '@A' does not exist}}
  %p = kanagawa.path [#kanagawa.step<child , @a : !kanagawa.scoperef<@foo::@A>>]
}
}

// -----

kanagawa.design @foo {
kanagawa.class sym @PathStepNonExistingParent {
  %this = kanagawa.this <@foo::@PathStepNonExistingParent>
  // expected-error @+1 {{'kanagawa.path' op last kanagawa.step in path must specify a symbol for the scoperef}}
  %p = kanagawa.path [#kanagawa.step<parent : !kanagawa.scoperef>]
}
}
