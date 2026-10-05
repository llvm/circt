// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "IntegerAndWrongType" {
  firrtl.module @IntegerAndWrongType(
      in %lhs: !firrtl.integer, in %rhs: !firrtl.string) {
    // expected-error @+1 {{'firrtl.integer.and' op operand #1 must be integer type}}
    %0 = "firrtl.integer.and"(%lhs, %rhs) : (!firrtl.integer, !firrtl.string) -> !firrtl.integer
  }
}

// -----

firrtl.circuit "IntegerOrWrongType" {
  firrtl.module @IntegerOrWrongType(
      in %lhs: !firrtl.uint<8>, in %rhs: !firrtl.integer) {
    // expected-error @+1 {{'firrtl.integer.or' op operand #0 must be integer type}}
    %0 = "firrtl.integer.or"(%lhs, %rhs) : (!firrtl.uint<8>, !firrtl.integer) -> !firrtl.integer
  }
}

// -----

firrtl.circuit "IntegerNotWrongType" {
  firrtl.module @IntegerNotWrongType(in %input: !firrtl.string) {
    // expected-error @+1 {{'firrtl.integer.not' op operand #0 must be integer type}}
    %0 = "firrtl.integer.not"(%input) : (!firrtl.string) -> !firrtl.integer
  }
}

// -----

firrtl.circuit "IntegerAndWrongArity" {
  firrtl.module @IntegerAndWrongArity(in %input: !firrtl.integer) {
    // expected-error @+1 {{'firrtl.integer.and' op expected 2 operands, but found 1}}
    %0 = "firrtl.integer.and"(%input) : (!firrtl.integer) -> !firrtl.integer
  }
}

// -----

firrtl.circuit "IntegerNotWrongArity" {
  firrtl.module @IntegerNotWrongArity(
      in %lhs: !firrtl.integer, in %rhs: !firrtl.integer) {
    // expected-error @+1 {{'firrtl.integer.not' op requires a single operand}}
    %0 = "firrtl.integer.not"(%lhs, %rhs) : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  }
}
