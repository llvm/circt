// RUN: circt-opt %s -verify-diagnostics -split-input-file

om.class @Class() {
  // expected-error @+1 {{'om.object' op result type ("Bar") does not match referred to class ("Foo")}}
  %0 = om.object @Foo() : () -> !om.class.type<@Bar>
}

// -----

om.class @Class() {
  // expected-error @+1 {{'om.object' op refers to non-existant class ("NonExistant")}}
  %0 = om.object @NonExistant() : () -> !om.class.type<@NonExistant>
}

// -----

// expected-note @+1 {{formal parameters:}}
om.class @Class1(%param : i1) {}

om.class @Class2() {
  // expected-error @+2 {{'om.object' op actual parameter list doesn't match formal parameter list}}
  // expected-note @+1 {{actual parameters:}}
  %0 = om.object @Class1() : () -> !om.class.type<@Class1>
}

// -----

om.class @Class1(%param : i1) {}

om.class @Class2(%param : i2) {
  // expected-error @+1 {{'om.object' op actual parameter type ('i2') doesn't match formal parameter type ('i1')}}
  %1 = om.object @Class1(%param) : (i2) -> !om.class.type<@Class1>
}
