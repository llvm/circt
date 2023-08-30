// RUN: circt-opt -firrtl-lower-classes %s -verify-diagnostics

firrtl.circuit "Component" {
  firrtl.module @Component() {}
  // expected-error @+1{{failed to legalize operation 'om.class' that was explicitly marked illegal}}
  firrtl.class @Map(in %s1: !firrtl.map<list<string>, string>) {}
}
