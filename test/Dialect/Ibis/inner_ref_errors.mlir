// RUN: circt-opt --hw-verify-irn --split-input-file --verify-diagnostics %s

ibis.design @foo {
ibis.class sym @MissingPort {
  %this = ibis.this <@foo::@MissingPort>
  // expected-error @+1 {{'ibis.get_port' op port '@C_in' does not exist in @MissingPort}}
  %c_in = ibis.get_port %this, @C_in : !ibis.scoperef<@foo::@MissingPort> -> !ibis.portref<in i1>
}
}

// -----

ibis.design @foo {
ibis.class sym @InvalidGetVar2 {
  %this = ibis.this <@foo::@InvalidGetVar2>
  ibis.var @var : memref<i32>
  ibis.method @foo()  {
    %parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@foo::@InvalidGetVar2>>
    ]
    // expected-error @+1 {{'ibis.get_var' op dereferenced type ('memref<i1>') must match variable type ('memref<i32>')}}
    %var = ibis.get_var %parent, @var : !ibis.scoperef<@foo::@InvalidGetVar2> -> memref<i1>
    ibis.return
  }
}
}

// -----

ibis.design @foo {
ibis.class sym @InvalidGetVar {
  %this = ibis.this <@foo::@InvalidGetVar>
  ibis.var @var : memref<i32>
  ibis.method @foo()  {
    %parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@foo::@InvalidGetVar>>
    ]
    // expected-error @+1 {{'ibis.get_var' op result #0 must be memref of any type values, but got 'i32'}}
    %var = ibis.get_var %parent, @var : !ibis.scoperef<@foo::@InvalidGetVar> -> i32
    ibis.return
  }
}
}

// -----

ibis.design @foo {
ibis.class sym @PortTypeMismatch {
  %this = ibis.this <@foo::@PortTypeMismatch>
  ibis.port.input "in" sym @in : i1
  // expected-error @+1 {{'ibis.get_port' op symbol '@in' refers to a port of type 'i1', but this op has type 'i2'}}
  %c_in = ibis.get_port %this, @in : !ibis.scoperef<@foo::@PortTypeMismatch> -> !ibis.portref<in i2>
}
}

// -----

ibis.design @foo {
ibis.class sym @PathStepNonExistingChild {
  %this = ibis.this <@foo::@PathStepNonExistingChild>
  // expected-error @+1 {{'ibis.path' op ibis.step scoperef symbol '@A' does not exist}}
  %p = ibis.path [#ibis.step<child , @a : !ibis.scoperef<@foo::@A>>]
}
}

// -----

ibis.design @foo {
ibis.class sym @PathStepNonExistingParent {
  %this = ibis.this <@foo::@PathStepNonExistingParent>
  // expected-error @+1 {{'ibis.path' op last ibis.step in path must specify a symbol for the scoperef}}
  %p = ibis.path [#ibis.step<parent : !ibis.scoperef>]
}
}
