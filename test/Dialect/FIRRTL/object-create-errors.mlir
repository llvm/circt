// RUN: circt-opt %s -verify-diagnostics -split-input-file

firrtl.circuit "MissingOperand" {
  firrtl.extclass @Class(in str: !firrtl.string)
  firrtl.module @MissingOperand() {
    // expected-error @below {{missing operand for input port "str"}}
    %obj = firrtl.object.create() : () -> !firrtl.class<@Class(in str: !firrtl.string)>
  }
}

// -----

firrtl.circuit "ExtraOperand" {
  firrtl.extclass @Class()
  firrtl.module @ExtraOperand(in %str: !firrtl.string) {
    // expected-error @below {{expected 0 input operands, but got 1}}
    %obj = firrtl.object.create(%str) : (!firrtl.string) -> !firrtl.class<@Class()>
  }
}

// -----

firrtl.circuit "WrongType" {
  firrtl.extclass @Class(in str: !firrtl.string)
  firrtl.module @WrongType(in %int: !firrtl.integer) {
    // expected-error @below {{input port "str" expects '!firrtl.string', but operand 0 has type '!firrtl.integer'}}
    %obj = firrtl.object.create(%int) : (!firrtl.integer) -> !firrtl.class<@Class(in str: !firrtl.string)>
  }
}

// -----

firrtl.circuit "Unknown" {
  firrtl.module @Unknown() {
    // expected-error @below {{references unknown class @Missing}}
    %obj = firrtl.object.create() : () -> !firrtl.class<@Missing()>
  }
}

// -----

firrtl.circuit "ClassMismatch" {
  firrtl.extclass @Class(out result: !firrtl.string)
  firrtl.module @ClassMismatch() {
    // expected-error @below {{has wrong number of ports, got 0, expected 1}}
    %obj = firrtl.object.create() : () -> !firrtl.class<@Class()>
  }
}

// -----

// Inputs of value-like objects are not assignment destinations.
firrtl.circuit "InputFlow" {
  firrtl.extclass @Class(in str: !firrtl.string)
  firrtl.module @InputFlow(in %str: !firrtl.string) {
    // expected-note @below {{the destination was defined here}}
    %obj = firrtl.object.create(%str) : (!firrtl.string) -> !firrtl.class<@Class(in str: !firrtl.string)>
    %in = firrtl.object.subfield %obj[str] : !firrtl.class<@Class(in str: !firrtl.string)>
    // expected-error @below {{has no flow}}
    firrtl.propassign %in, %str : !firrtl.string
  }
}
