// RUN: circt-opt --split-input-file --verify-diagnostics %s

ibis.class @C {
  %this = ibis.this @C
  ibis.method @typeMismatch1() -> ui32 {
    // expected-error @+1 {{must return a value}}
    ibis.return
  }
}

// -----

ibis.class @C {
  %this = ibis.this @C
  ibis.method @typeMismatch2() {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{cannot return a value from a function with no result type}}
    ibis.return %c : i8
  }
}

// -----
ibis.class @C {
  %this = ibis.this @C
  ibis.method @typeMismatch3() -> ui32 {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{return type ('i8') must match function return type ('ui32')}}
    ibis.return %c : i8
  }
}

// -----

ibis.class @MissingPort {
  %this = ibis.this @MissingPort
  // expected-error @+1 {{'ibis.get_port' op port '@C_in' does not exist in "MissingPort"}}
  %c_in = ibis.get_port %this, @C_in : !ibis.scoperef<@MissingPort> -> !ibis.portref<in i1>
}

// -----

ibis.class @PortTypeMismatch {
  %this = ibis.this @PortTypeMismatch
  ibis.port.input @in : i1
  // expected-error @+1 {{'ibis.get_port' op symbol '@in' refers to a port of type 'i1', but this op has type 'i2'}}
  %c_in = ibis.get_port %this, @in : !ibis.scoperef<@PortTypeMismatch> -> !ibis.portref<in i2>
}

// -----

// expected-error @+1 {{'ibis.class' op must contain only one 'ibis.this' operation}}
ibis.class @MultipleThis {
  %this = ibis.this @MultipleThis
  %this2 = ibis.this @MultipleThis
}

// -----

// expected-error @+1 {{'ibis.container' op must contain a 'ibis.this' operation}}
ibis.container @NoThis {
}

// -----

ibis.class @PathStepParentWithInstanceName {
  %this = ibis.this @PathStepParentWithInstanceName
  // expected-error @+1 {{ibis.step 'parent' may not specify an instance name}}
  %p = ibis.path [#ibis.step<parent , @a : !ibis.scoperef>]
}

// -----

ibis.class @PathStepInvalidType {
  %this = ibis.this @PathStepParentWithInstanceName
  // expected-error @+1 {{ibis.step type must be an !ibis.scoperef type}}
  %p = ibis.path [#ibis.step<parent : i1>]
}

// -----

ibis.class @PathStepNonExistingChild {
  %this = ibis.this @PathStepNonExistingChild
  // expected-error @+1 {{'ibis.path' op ibis.step scoperef symbol '@A' does not exist}}
  %p = ibis.path [#ibis.step<child , @a : !ibis.scoperef<@A>>]
}

// -----

ibis.class @PathStepNonExistingChild {
  %this = ibis.this @PathStepNonExistingChild
  // expected-error @+1 {{'ibis.path' op last ibis.step in path must specify a symbol for the scoperef}}
  %p = ibis.path [#ibis.step<parent : !ibis.scoperef>]
}

// -----

ibis.class @PathStepChildMissingSymbol {
  %this = ibis.this @PathStepNonExistingChild
  // expected-error @+1 {{ibis.step 'child' must specify an instance name}}
  %p = ibis.path [#ibis.step<child : !ibis.scoperef<@A>>]
}
