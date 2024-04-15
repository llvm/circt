// RUN: circt-opt --split-input-file --verify-diagnostics %s

ibis.design @foo {
ibis.class @C {
  %this = ibis.this <@C>
  ibis.method @typeMismatch1() -> (ui32, i32) {
    // expected-error @+1 {{'ibis.return' op must have the same number of operands as the method has results}}
    ibis.return
  }
}
}

// -----
ibis.design @foo {
ibis.class @C {
  %this = ibis.this <@C>
  ibis.method @typeMismatch3() -> ui32 {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{'ibis.return' op operand type ('i8') must match function return type ('ui32')}}
    ibis.return %c : i8
  }
}
}

// -----

ibis.design @foo {
// expected-error @+1 {{'ibis.class' op must contain only one 'ibis.this' operation}}
ibis.class @MultipleThis {
  %this = ibis.this <@MultipleThis>
  %this2 = ibis.this <@MultipleThis>
}
}

// -----

ibis.design @foo {
// expected-error @+1 {{'ibis.container' op must contain a 'ibis.this' operation}}
ibis.container @NoThis {
}
}

// -----

ibis.design @foo {
ibis.class @PathStepParentWithInstanceName {
  %this = ibis.this <@PathStepParentWithInstanceName>
  // expected-error @+1 {{ibis.step 'parent' may not specify an instance name}}
  %p = ibis.path [#ibis.step<parent , @a : !ibis.scoperef>]
}
}

// -----

ibis.design @foo {
ibis.class @PathStepInvalidType {
  %this = ibis.this <@PathStepParentWithInstanceName>
  // expected-error @+1 {{ibis.step type must be an !ibis.scoperef type}}
  %p = ibis.path [#ibis.step<parent : i1>]
}
}

// -----

ibis.design @foo {
ibis.class @PathStepChildMissingSymbol {
  %this = ibis.this <@PathStepNonExistingChild>
  // expected-error @+1 {{ibis.step 'child' must specify an instance name}}
  %p = ibis.path [#ibis.step<child : !ibis.scoperef<@A>>]
}
}

// -----

ibis.design @foo {
ibis.class @InvalidVar {
  %this = ibis.this <@C>
  // expected-error @+1 {{'ibis.var' op attribute 'type' failed to satisfy constraint: any memref type}}
  ibis.var @var : i32
}
}

// -----

ibis.design @foo {
ibis.class @InvalidReturn {
  %this = ibis.this <@InvalidReturn>
  ibis.method @foo() {
    %c = hw.constant 1 : i32
    // expected-error @+1 {{'ibis.sblock.return' op number of operands must match number of block outputs}}
    %ret = ibis.sblock() -> i32 {
    }
    ibis.return
  }
}
}
