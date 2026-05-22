// RUN: circt-opt -om-elaborate-object %s -verify-diagnostics
// expected-error @unknown {{exactly one of 'target-class' or 'all-public-classes' must be specified}}
module {
  om.class @SomeClass() {
    om.class.fields
  }
}
