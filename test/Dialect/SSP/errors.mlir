// RUN: circt-opt %s -split-input-file -verify-diagnostics

ssp.instance "error1" of "Problem" {
  // expected-error @+1 {{references invalid source operation: @opX}}
  operation<> @op1(@opX)
}

// -----

ssp.instance "error2" of "Problem" {
  operator_type @opr
  // expected-error @+1 {{references invalid source operation: @opr}}
  operation<> @op1(@opr)
}

// -----

ssp.instance "error3" of "Problem" {
  %0 = operation<> @Op0()
  operation<> @Op1(%0)
  // expected-error @+1 {{Auxiliary dependence from @Op0 is interleaved with SSA operands}}
  operation<> @Op2(@Op0, %0)
}

// -----

// expected-error @+1 {{custom op 'ssp.instance' carries unknown shortform property: unknown}}
ssp.instance "error4" of "Problem" [unknown<>] {}

// -----

// expected-error @+2 {{custom op 'ssp.instance' expected integer value}}
// expected-error @+1 {{custom op 'ssp.instance' failed to parse InitiationInterval parameter 'value' which is to be a `unsigned`}}
ssp.instance "error5" of "Problem" [II<"not-an-integer">] {}
