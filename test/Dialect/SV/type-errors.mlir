// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{sv.net element type must have a packed base type, but got '!hw.string'}}
func.func @netString(%arg0: !sv.net<!hw.string>) {
  return
}

// -----

// expected-error @+1 {{sv.net element type must have a packed base type, but got '!hw.struct<a: !hw.uarray<8xi8>>'}}
func.func @netUnpackedStructMember(%arg0: !sv.net<!hw.struct<a: !hw.uarray<8xi8>>>) {
  return
}

// -----

// expected-error @+1 {{sv.net element type must have a packed base type, but got '!hw.array<2xuarray<4xi8>>'}}
func.func @netUnpackedArrayElement(%arg0: !sv.net<!hw.array<2x!hw.uarray<4xi8>>>) {
  return
}

// -----

// expected-error @+1 {{sv.net element type may not be itself an sv.net or sv.var handle}}
func.func @netOfNet(%arg0: !sv.net<!sv.net<i1>>) {
  return
}

// -----

// expected-error @+1 {{sv.net element type may not be itself an sv.net or sv.var handle}}
func.func @netOfVar(%arg0: !sv.net<!sv.var<i1>>) {
  return
}

// -----

// expected-error @+1 {{sv.var element type may not be itself an sv.var or sv.net handle}}
func.func @varOfVar(%arg0: !sv.var<!sv.var<i1>>) {
  return
}

// -----

// expected-error @+1 {{sv.var element type may not be itself an sv.var or sv.net handle}}
func.func @varOfNet(%arg0: !sv.var<!sv.net<i1>>) {
  return
}

// -----

// expected-error @+1 {{sv.var element type must be a valid value type, but got 'index'}}
func.func @varOfIndex(%arg0: !sv.var<index>) {
  return
}
