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
