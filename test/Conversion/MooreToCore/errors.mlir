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

func.func @ExtractRefOnNet(%arg0: !moore.net<i32>) {
  // expected-error @below {{op extracts from a net which is not currently supported}}
  // expected-error @below {{failed to legalize}}
  %0 = moore.extract_ref %arg0 from 1 : net<i32> -> net<i2>
  return
}

// -----

func.func @DynExtractRefOnNet(%arg0: !moore.net<i32>, %arg1: !moore.i32) {
  // expected-error @below {{op extracts from a net which is not currently supported}}
  // expected-error @below {{failed to legalize}}
  %0 = moore.dyn_extract_ref %arg0 from %arg1 : net<i32>, i32 -> net<i1>
  return
}

// -----

func.func @DynExtractRefStructNonIntegerField(%j: !moore.ref<struct<{d: array<4 x l4>}>>, %idx: !moore.l4) -> (!moore.ref<l4>) {
  // expected-error @below {{failed to legalize operation 'moore.dyn_extract_ref'}}
  %0 = moore.dyn_extract_ref %j from %idx : ref<struct<{d: array<4 x l4>}>>, !moore.l4 -> ref<l4>
  return %0 : !moore.ref<l4>
}

// -----

func.func @StructExtractRefOnNet(%arg0: !moore.net<struct<{a: i32}>>) {
  // expected-error @below {{op extracts from a net which is not currently supported}}
  // expected-error @below {{failed to legalize}}
  %0 = moore.struct_extract_ref %arg0, "a" : net<struct<{a: i32}>> -> net<i32>
  return
}

// -----

func.func @UnionExtractRefOnNet(%arg0: !moore.net<union<{a: i32}>>) {
  // expected-error @below {{op extracts from a net which is not currently supported}}
  // expected-error @below {{failed to legalize}}
  %0 = moore.union_extract_ref %arg0, "a" : net<union<{a: i32}>> -> net<i32>
  return
}
