// RUN: circt-opt %s --verify-diagnostics --split-input-file

firrtl.circuit "Mismatch" {
  firrtl.module @Mismatch(in %items: !firrtl.mutable_collection<integer>, in %value: !firrtl.string) {
    // expected-error @+1 {{element type '!firrtl.string' does not match collection element type '!firrtl.integer'}}
    firrtl.prop.push %items, %value : !firrtl.mutable_collection<integer>, !firrtl.string
  }
}

// -----

firrtl.circuit "OutputPort" {
  firrtl.module @OutputPort(out %items: !firrtl.mutable_collection<integer>) {
    %one = firrtl.integer 1
    // expected-error @+1 {{collection has sink flow, expected source or duplex flow}}
    firrtl.prop.push %items, %one : !firrtl.mutable_collection<integer>, !firrtl.integer
  }
}

// -----

firrtl.circuit "Constant" {
  // expected-error @+1 {{mutable collections cannot be const}}
  firrtl.module @Constant(in %items: !firrtl.const.mutable_collection<integer>) {}
}
