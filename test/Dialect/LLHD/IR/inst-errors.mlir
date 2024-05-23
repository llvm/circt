//RUN: circt-opt %s -split-input-file -verify-diagnostics

// Testing Objectives:
// * inst can only be used in hw.module
// * inst must always refer to a valid proc (match symbol name, input and output operands)
// * syntax: no inputs and outputs, one input zero outputs, zero inputs one output, multiple inputs and outputs
// * check that number of inputs and number of outputs are verified separately

llhd.proc @empty_proc() -> () {
  llhd.halt
}

llhd.proc @fail() -> () {
  // expected-error @+1 {{expects parent op 'hw.module'}}
  llhd.inst "empty" @empty_proc() -> () : () -> ()
  llhd.halt
}

// -----

llhd.proc @operand_count_mismatch() -> () {
  llhd.halt
}

hw.module @caller(inout %arg : i32) {
  // expected-error @+1 {{incorrect number of inputs for proc instantiation}}
  llhd.inst "mismatch" @operand_count_mismatch(%arg) -> () : (!hw.inout<i32>) -> ()
}

// -----

hw.module @caller() {
  // expected-error @+1 {{does not reference a valid llhd.proc}}
  llhd.inst "does_not_exist" @does_not_exist() -> () : () -> ()
}

// -----

// expected-error @below {{region #0 ('body') failed to verify constraint: region with at least 1 blocks}}
llhd.proc @empty() -> () {}

// -----

// expected-error @below {{empty block: expect at least a terminator}}
llhd.proc @empty(%a: !hw.inout<i1>) -> () {}
