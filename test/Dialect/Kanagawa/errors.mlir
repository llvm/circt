// RUN: circt-opt --split-input-file --verify-diagnostics %s

kanagawa.design @foo {
kanagawa.class sym @C {
  %this = kanagawa.this <@foo::@C>
  kanagawa.method @typeMismatch1() -> (ui32, i32) {
    // expected-error @+1 {{'kanagawa.return' op must have the same number of operands as the method has results}}
    kanagawa.return
  }
}
}

// -----
kanagawa.design @foo {
kanagawa.class sym @C {
  %this = kanagawa.this <@foo::@C>
  kanagawa.method @typeMismatch3() -> ui32 {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{'kanagawa.return' op operand type ('i8') must match function return type ('ui32')}}
    kanagawa.return %c : i8
  }
}
}

// -----

kanagawa.design @foo {
// expected-error @+1 {{'kanagawa.class' op must contain only one 'kanagawa.this' operation}}
kanagawa.class sym @MultipleThis {
  %this = kanagawa.this <@foo::@MultipleThis>
  %this2 = kanagawa.this <@foo::@MultipleThis>
}
}

// -----

kanagawa.design @foo {
// expected-error @+1 {{'kanagawa.container' op must contain a 'kanagawa.this' operation}}
kanagawa.container sym @NoThis {
}
}

// -----

kanagawa.design @foo {
kanagawa.class sym @PathStepParentWithInstanceName {
  %this = kanagawa.this <@foo::@PathStepParentWithInstanceName>
  // expected-error @+1 {{kanagawa.step 'parent' may not specify an instance name}}
  %p = kanagawa.path [#kanagawa.step<parent , @a : !kanagawa.scoperef>]
}
}

// -----

kanagawa.design @foo {
kanagawa.class sym @PathStepInvalidType {
  %this = kanagawa.this <@foo::@PathStepParentWithInstanceName>
  // expected-error @+1 {{kanagawa.step type must be an !kanagawa.scoperef type}}
  %p = kanagawa.path [#kanagawa.step<parent : i1>]
}
}

// -----

kanagawa.design @foo {
kanagawa.class sym @PathStepChildMissingSymbol {
  %this = kanagawa.this <@foo::@PathStepNonExistingChild>
  // expected-error @+1 {{kanagawa.step 'child' must specify an instance name}}
  %p = kanagawa.path [#kanagawa.step<child : !kanagawa.scoperef<@foo::@A>>]
}
}

// -----

kanagawa.design @foo {
kanagawa.class sym @InvalidVar {
  %this = kanagawa.this <@foo::@C>
  // expected-error @+1 {{'kanagawa.var' op attribute 'type' failed to satisfy constraint: any memref type}}
  kanagawa.var @var : i32
}
}

// -----

kanagawa.design @foo {
kanagawa.class sym @InvalidReturn {
  %this = kanagawa.this <@foo::@InvalidReturn>
  kanagawa.method @foo() {
    %c = hw.constant 1 : i32
    // expected-error @+1 {{'kanagawa.sblock.return' op number of operands must match number of block outputs}}
    %ret = kanagawa.sblock() -> i32 {
    }
    kanagawa.return
  }
}
}
