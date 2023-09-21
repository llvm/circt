// RUN: circt-opt --split-input-file --verify-diagnostics %s

ibis.class @C {
  %this = ibis.this @C
  ibis.method @typeMismatch1() -> (ui32, i32) {
    // expected-error @+1 {{'ibis.return' op must have the same number of operands as the method has results}}
    ibis.return
  }
}

// -----
ibis.class @C {
  %this = ibis.this @C
  ibis.method @typeMismatch3() -> ui32 {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{'ibis.return' op operand type ('i8') must match function return type ('ui32')}}
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

// -----

ibis.class @InvalidVar {
  %this = ibis.this @C
  // expected-error @+1 {{'ibis.var' op attribute 'type' failed to satisfy constraint: any memref type}}
  ibis.var @var : i32
}

// -----

ibis.class @InvalidGetVar {
  %this = ibis.this @InvalidGetVar
  ibis.var @var : memref<i32>
  ibis.method @foo()  {
    %parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@InvalidGetVar>>
    ]
    // expected-error @+1 {{'ibis.get_var' op result #0 must be memref of any type values, but got 'i32'}}
    %var = ibis.get_var %parent, @var : !ibis.scoperef<@InvalidGetVar> -> i32
    ibis.return
  }
}

// -----

ibis.class @InvalidGetVar2 {
  %this = ibis.this @InvalidGetVar2
  ibis.var @var : memref<i32>
  ibis.method @foo()  {
    %parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@InvalidGetVar2>>
    ]
    // expected-error @+1 {{'ibis.get_var' op dereferenced type ('memref<i1>') must match variable type ('memref<i32>')}}
    %var = ibis.get_var %parent, @var : !ibis.scoperef<@InvalidGetVar2> -> memref<i1>
    ibis.return
  }
}

// -----

ibis.class @InvalidReturn {
  %this = ibis.this @InvalidReturn
  ibis.method @foo() {
    %c = hw.constant 1 : i32
    // expected-error @+1 {{'ibis.sblock.return' op number of operands must match number of block outputs}}
    %ret = ibis.sblock() -> i32 {
    }
    ibis.return
  }
}
