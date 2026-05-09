// RUN: circt-opt -om-elaborate-object %s -verify-diagnostics
// expected-error @unknown {{either 'test' or 'target-class' must be specified}}
module {
  om.class @SomeClass() {
    om.class.fields
  }
}
