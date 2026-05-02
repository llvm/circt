// RUN: circt-opt -om-elaborate-object='test=true' %s -verify-diagnostics -split-input-file

om.class @AssertFalse() {
  %false = om.constant false
  // expected-error @below {{OM property assertion failed: condition must be true}}
  om.property_assert %false, "condition must be true" : i1
  om.class.fields
}

// -----

// Multiple assertions
om.class @MultipleAsserts() {
  %false = om.constant false
  // expected-error @below {{OM property assertion failed: first assertion fails}}
  om.property_assert %false, "first assertion fails" : i1
  // expected-error @below {{OM property assertion failed: second assertion fails}}
  om.property_assert %false, "second assertion fails" : i1
  om.class.fields
}

// -----

// Multiple assertions in nested classes
om.class @WrapperWithAssert(%in: i1) -> (out: i1) {
  // expected-error @below {{OM property assertion failed: wrapper assertion fails}}
  om.property_assert %in, "wrapper assertion fails" : i1
  om.class.fields %in : i1
}

om.class @ParentWithNestedAsserts() {
  %false = om.constant false
  %obj = om.object @WrapperWithAssert(%false) : (i1) -> !om.class.type<@WrapperWithAssert>
  %result = om.object.field %obj["out"] : (!om.class.type<@WrapperWithAssert>) -> i1
  // expected-error @below {{OM property assertion failed: parent assertion fails}}
  om.property_assert %result, "parent assertion fails" : i1
  om.class.fields
}

// -----

// Complex expression resulting in false after elaboration
om.class @ComplexExpressionFalse() {
  %false = om.constant false
  %obj = om.object @BoolWrapper(%false) : (i1) -> !om.class.type<@BoolWrapper>
  %result = om.object.field %obj["out"] : (!om.class.type<@BoolWrapper>) -> i1
  // expected-error @below {{OM property assertion failed: complex expression is false}}
  om.property_assert %result, "complex expression is false" : i1
  om.class.fields
}

om.class @BoolWrapper(%in: i1) -> (out: i1) {
  om.class.fields %in : i1
}

// -----

// Cycle in dataflow (field access creates a cycle that can't be evaluated)
om.class @WrapperCycle(%val: !om.integer) -> (out: !om.integer) {
  om.class.fields %val : !om.integer
}

om.class @DataflowCycle() -> (result: !om.integer) {
  %obj = om.object @WrapperCycle(%feedback) : (!om.integer) -> !om.class.type<@WrapperCycle>
  // expected-error @below {{failed to evaluate om.object.field}}
  %feedback = om.object.field %obj["out"] : (!om.class.type<@WrapperCycle>) -> !om.integer
  om.class.fields %feedback : !om.integer
}
