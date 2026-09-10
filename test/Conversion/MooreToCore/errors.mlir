// RUN: circt-opt %s --convert-moore-to-core --split-input-file --verify-diagnostics

func.func @invalidType() {
  // expected-error @below {{failed to legalize operation 'moore.variable'}}
  %var = moore.variable : <!moore.assoc_array<i32, i32>>

  return
}

// -----

func.func @dynamicArrayVariable() {
  // expected-error @below {{failed to legalize operation 'moore.variable'}}
  %var = moore.variable : <!moore.open_uarray<i32>>
  return
}

// -----

func.func @unsupportedOpenArrayCastReftoOpen(%arg0: !moore.ref<i1>) {
  // expected-error @below {{unsupported DPI open-array conversion from '!moore.ref<i1>' to '!moore.open_uarray<i8>'}}
  // expected-error @below {{failed to legalize operation 'moore.conversion'}}
  %0 = moore.conversion %arg0 : !moore.ref<i1> -> !moore.open_uarray<i8>
  return
}

// -----

func.func @unsupportedOpenArrayCastOpenToFixed(%arg0: !moore.open_uarray<i8>) {
  // expected-error @below {{unsupported DPI open-array conversion from '!moore.open_uarray<i8>' to '!moore.uarray<8 x i8>'}}
  // expected-error @below {{failed to legalize operation 'moore.conversion'}}
  %1 = moore.conversion %arg0 : !moore.open_uarray<i8> -> !moore.uarray<8 x i8>
  return
}

// -----

func.func @unsupportedOpenArrayCastPackedToUnpacked(%arg0: !moore.array<8 x i8>) {
  // expected-error @below {{unsupported DPI open-array conversion from '!moore.array<8 x i8>' to '!moore.open_uarray<i8>'}}
  // expected-error @below {{failed to legalize operation 'moore.conversion'}}
  %2 = moore.conversion %arg0 : !moore.array<8 x i8> -> !moore.open_uarray<i8>
  return
}

// -----

func.func @unsupportedOpenArrayCastUnpackedToPacked(%arg0: !moore.uarray<8 x i8>) {
  // expected-error @below {{unsupported DPI open-array conversion from '!moore.uarray<8 x i8>' to '!moore.open_array<i8>'}}
  // expected-error @below {{failed to legalize operation 'moore.conversion'}}
  %3 = moore.conversion %arg0 : !moore.uarray<8 x i8> -> !moore.open_array<i8>
  return
}

// -----

// expected-error @below {{port '"queue_port"' has unsupported type '!moore.assoc_array<i32, string>' that cannot be converted to hardware type}}
// expected-error @below {{failed to legalize}}
moore.module @UnsupportedInputPortType(in %queue_port : !moore.assoc_array<i32, string>) {
  moore.output
}

// -----

// expected-error @below {{port '"data"' has unsupported type '!moore.assoc_array<i32, string>' that cannot be converted to hardware type}}
// expected-error @below {{failed to legalize}}
moore.module @MixedPortsWithUnsupported(in %valid : !moore.l1, in %data : !moore.assoc_array<i32, string>, out out : !moore.l1) {
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
