// RUN: circt-opt %s --verify-diagnostics --split-input-file

om.class @Mismatch(%items: !om.collection<!om.integer>, %value: !om.string) {
  // expected-error @+1 {{element type '!om.string' does not match collection element type '!om.integer'}}
  om.collection_push %items, %value : !om.collection<!om.integer>, !om.string
  om.class.fields
}

// -----

om.class @InitialMismatch(%value: !om.string) {
  // expected-error @+1 {{element type '!om.string' does not match collection element type '!om.integer'}}
  %items = "om.collection"(%value) : (!om.string) -> !om.collection<!om.integer>
  om.class.fields
}

// -----

om.class @ListIsNotCollection(%items: !om.list<!om.integer>, %value: !om.integer) {
  // expected-error @+1 {{operand #0 must be}}
  "om.collection_push"(%items, %value) : (!om.list<!om.integer>, !om.integer) -> ()
  om.class.fields
}

// -----

// expected-error @+1 {{expects parent op 'om.class'}}
%items = om.collection : !om.collection<!om.integer>

// -----

om.class @ConstantHandle() -> (items: !om.collection<!om.integer>) {
  // expected-error @+1 {{collection handles cannot be constants}}
  %items = om.constant "not a handle" : !om.collection<!om.integer>
  om.class.fields %items : !om.collection<!om.integer>
}
