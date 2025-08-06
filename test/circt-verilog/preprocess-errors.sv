// RUN: circt-verilog %s -E --verify-diagnostics
// REQUIRES: slang

// expected-error-re @below {{'unknown.sv': {{.+}}}}
`include "unknown.sv"
