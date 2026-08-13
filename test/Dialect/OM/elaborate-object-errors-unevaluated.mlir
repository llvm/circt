// RUN: circt-opt -om-elaborate-object='all-public-classes=true allow-unevaluated=true' %s -verify-diagnostics -split-input-file -allow-unregistered-dialect

om.class @UnevaluatedMessage() {
  %0 = om.unknown : !om.string
  %1 = om.constant false
  // expected-error @below {{OM property assertion failed: <unevaluated>}}
  om.property_assert %1, %0 : i1
  om.class.fields
}

// -----

om.class @BlockArgMessage(%msg: !om.string) {
  %false = om.constant false
  // expected-error @below {{OM property assertion failed: <unevaluated>}}
  om.property_assert %false, %msg : i1
  om.class.fields
}

// -----

om.class @NonConstantMessage() {
  %false = om.constant false
  %msg = om.constant #hw.param.verbatim<"P"> : !om.string
  // expected-error @below {{OM property assertion failed, but no message is available because the message is not a constant string}}
  om.property_assert %false, %msg : i1
  om.class.fields
}
