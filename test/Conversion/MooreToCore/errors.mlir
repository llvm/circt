// RUN: circt-opt %s --convert-moore-to-core --split-input-file --verify-diagnostics

func.func @dynamicArrayVariable() {
  // expected-error @below {{failed to legalize operation 'moore.variable'}}
  %var = moore.variable : <!moore.open_uarray<i32>>
  return
}

// -----

// expected-error @below {{port '"e"' has unsupported type '!moore.event' that cannot be converted to hardware type}}
// expected-error @below {{failed to legalize}}
moore.module @UnsupportedInputPortType(in %e : !moore.event) {
  moore.output
}

// -----

// expected-error @below {{port '"data"' has unsupported type '!moore.event' that cannot be converted to hardware type}}
// expected-error @below {{failed to legalize}}
moore.module @MixedPortsWithUnsupported(in %valid : !moore.l1, in %data : !moore.event, out out : !moore.l1) {
  moore.output %valid : !moore.l1
}

// -----

moore.class.classdecl @ClassWithString {
  moore.class.propertydecl @text : !moore.string
}

func.func @classNewWithString() {
  // expected-error @below {{class struct has member types with no data layout}}
  // expected-error @below {{failed to legalize operation 'moore.class.new'}}
  %h = moore.class.new : <@ClassWithString>
  return
}

// -----
