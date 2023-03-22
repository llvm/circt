// RUN: circt-opt %s -split-input-file -verify-diagnostics
// We do not validate per-field symbols if the type doesn't implement FieldIDTTypeInterface.
// This also means per-fields into, e.g., non-base types are similarly trusted.
// This is more conservative, but put under test so change here is not missed.

firrtl.circuit "CombMemInvalidSym" {
  firrtl.module @CombMemInvalidSym() {
    // Symbol on fieldID 10 sub-element, should be error.
    %mem = chirrtl.combmem sym [<@x,10,public>] : !chirrtl.cmemory<bundle<a: uint<1>>, 1>
  }
}

// -----

firrtl.circuit "SeqMemInvalidSym" {
  firrtl.module @SeqMemInvalidSym() {
    // Symbol on fieldID 10 sub-element, should be error.
    %mem = chirrtl.seqmem sym [<@x,10,public>] Undefined : !chirrtl.cmemory<bundle<a: uint<1>>, 1>
  }
}
