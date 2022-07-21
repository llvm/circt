// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{must contain exactly one 'library' op and one 'graph' op}}
ssp.instance "error0a" of "Problem" {}

// -----

// expected-error @+1 {{must contain exactly one 'library' op and one 'graph' op}}
ssp.instance "error0b" of "Problem" {
  graph {}
  library {}
}

// -----

ssp.instance "error1" of "Problem" {
  library {}
  graph {
    // expected-error @+1 {{Auxiliary dependence references invalid source operation: @opX}}
    operation<> @op1(@opX)
  }
}

// -----

ssp.instance "error2" of "Problem" {
  library {
    operator_type @opr
  }
  graph {
    // expected-error @+1 {{Auxiliary dependence references invalid source operation: @opr}}
    operation<> @op1(@opr)
  }
}

// -----

ssp.instance "error3" of "Problem" {
  library {}
  graph {
    %0 = operation<> @Op0()
    operation<> @Op1(%0)
    // expected-error @+1 {{Auxiliary dependence from @Op0 is interleaved with SSA operands}}
    operation<> @Op2(@Op0, %0)
  }
}

// -----

// expected-error @+1 {{custom op 'ssp.instance' carries unknown shortform property: unknown}}
ssp.instance "error4" of "Problem" [unknown<>] {
  library {}
  graph {}
}

// -----

// expected-error @+2 {{custom op 'ssp.instance' expected integer value}}
// expected-error @+1 {{custom op 'ssp.instance' failed to parse InitiationInterval parameter 'value' which is to be a `unsigned`}}
ssp.instance "error5" of "Problem" [II<"not-an-integer">] {
  library {}
  graph {}
}

// -----

ssp.instance "error6" of "Problem" {
  library {}
  graph {
    // expected-error @+1 {{Linked operator type property references invalid operator type: @InvalidOpr}}
    operation<@InvalidOpr>()
  }
}
