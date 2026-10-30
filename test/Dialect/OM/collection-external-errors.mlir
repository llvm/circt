// RUN: circt-opt %s --om-elaborate-object='target-class=Top' --verify-diagnostics --split-input-file

om.class.extern @External(%items: !om.collection<!om.integer>) {}
om.class @Top() {
  %items = om.collection : !om.collection<!om.integer>
  // expected-error @+1 {{cannot evaluate external constructor with collection capabilities}}
  %writer = om.object @External(%items) : (!om.collection<!om.integer>) -> !om.class.type<@External>
  om.class.fields
}

// -----

om.class @Owner(%items: !om.collection<!om.integer>) -> (items: !om.collection<!om.integer>) {
  om.class.fields %items : !om.collection<!om.integer>
}
om.class.extern @External(%owner: !om.class.type<@Owner>) {}
om.class @Top() {
  %items = om.collection : !om.collection<!om.integer>
  %owner = om.object @Owner(%items) : (!om.collection<!om.integer>) -> !om.class.type<@Owner>
  // expected-error @+1 {{cannot evaluate external constructor with collection capabilities}}
  %writer = om.object @External(%owner) : (!om.class.type<@Owner>) -> !om.class.type<@External>
  om.class.fields
}

// -----

om.class.extern @External(%value: !om.any) {}
om.class @Top(%opaque: !om.any) {
  // expected-error @+1 {{cannot evaluate external constructor with collection capabilities}}
  %writer = om.object @External(%opaque) : (!om.any) -> !om.class.type<@External>
  om.class.fields
}
