// RUN: circt-opt %s -split-input-file -verify-diagnostics

ssp.instance "error1" of "Problem" {
  // expected-error @+1 {{'ssp.operation' op references invalid source operation: @opX}}
  ssp.operation @op1(@opX)
}

// -----

ssp.instance "error2" of "Problem" {
  ssp.operator_type @opr
  // expected-error @+1 {{'ssp.operation' op references invalid source operation: @opr}}
  ssp.operation @op1(@opr)
}

// -----

ssp.instance "error3" of "Problem" {
  %0 = ssp.operation @Op0()
  ssp.operation @Op1(%0)
  // expected-error @+1 {{Auxiliary dependence from @Op0 is interleaved with SSA operands}}
  ssp.operation @Op2(@Op0, %0)
}
