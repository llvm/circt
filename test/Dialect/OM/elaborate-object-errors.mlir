// RUN: circt-opt -om-elaborate-object='all-public-classes=true' %s -verify-diagnostics -split-input-file

om.class @AssertFalse() {
  %false = om.constant false
  %message = om.constant "condition must be true" : !om.string
  // expected-error @below {{OM property assertion failed: condition must be true}}
  om.property_assert %false, %message : i1
  om.class.fields
}

// -----

// Multiple assertions
om.class @MultipleAsserts() {
  %false = om.constant false
  %firstMessage = om.constant "first assertion fails" : !om.string
  // expected-error @below {{OM property assertion failed: first assertion fails}}
  om.property_assert %false, %firstMessage : i1
  %secondMessage = om.constant "second assertion fails" : !om.string
  // expected-error @below {{OM property assertion failed: second assertion fails}}
  om.property_assert %false, %secondMessage : i1
  om.class.fields
}

// -----

// Multiple assertions in nested classes
om.class private @WrapperWithAssert(%in: i1) -> (out: i1) {
  %message = om.constant "wrapper assertion fails" : !om.string
  // expected-error @below {{OM property assertion failed: wrapper assertion fails}}
  om.property_assert %in, %message : i1
  om.class.fields %in : i1
}

om.class @ParentWithNestedAsserts() {
  %false = om.constant false
  %obj = om.object @WrapperWithAssert(%false) : (i1) -> !om.class.type<@WrapperWithAssert>
  %result = om.object.field %obj["out"] : (!om.class.type<@WrapperWithAssert>) -> i1
  %message = om.constant "parent assertion fails" : !om.string
  // expected-error @below {{OM property assertion failed: parent assertion fails}}
  om.property_assert %result, %message : i1
  om.class.fields
}

// -----

// Complex expression resulting in false after elaboration
om.class @ComplexExpressionFalse() {
  %false = om.constant false
  %obj = om.object @BoolWrapper(%false) : (i1) -> !om.class.type<@BoolWrapper>
  %result = om.object.field %obj["out"] : (!om.class.type<@BoolWrapper>) -> i1
  %message = om.constant "complex expression is false" : !om.string
  // expected-error @below {{OM property assertion failed: complex expression is false}}
  om.property_assert %result, %message : i1
  om.class.fields
}

om.class private @BoolWrapper(%in: i1) -> (out: i1) {
  om.class.fields %in : i1
}

// -----

// Cycle in dataflow (field access creates a cycle that can't be evaluated)
om.class private @WrapperCycle(%val: !om.integer) -> (out: !om.integer) {
  // FIXME: Currently the primary location points at om.field.class field op due
  //        to backward compatibility with the old evaluator implementation.
  // expected-error @below {{failed to evaluate om.object.field}}
  om.class.fields %val : !om.integer
}

om.class @DataflowCycle() -> (result: !om.integer) {
  %obj = om.object @WrapperCycle(%feedback) : (!om.integer) -> !om.class.type<@WrapperCycle>
  %feedback = om.object.field %obj["out"] : (!om.class.type<@WrapperCycle>) -> !om.integer
  om.class.fields %feedback : !om.integer
}

// -----

om.class @UnevaluatedMessage() {
  // expected-note @below {{unevaluated message operation is here}}
  %0 = om.unknown : !om.string
  %1 = om.constant false
  // expected-error @below {{OM property assertion failed, but no message is available as the message is unevaluated}}
  om.property_assert %1, %0 : i1
  om.class.fields
}

// -----

om.class @CompoundMessage() {
  %false = om.constant false
  %foo = om.constant "foo" : !om.string
  %space = om.constant " " : !om.string
  %bar = om.constant "bar" : !om.string
  %foobar = om.string.concat %foo, %space, %bar : !om.string
  // expected-error @below {{OM property assertion failed: foo bar}}
  om.property_assert %false, %foobar : i1
  om.class.fields
}
