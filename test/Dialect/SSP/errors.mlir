// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{must contain exactly one 'library' op and one 'graph' op}}
ssp.instance "error0a" of "Problem" {}

// -----

// expected-error @+1 {{must contain exactly one 'library' op and one 'graph' op}}
ssp.instance "error0b" of "Problem" {
  graph {}
  library {}
}
