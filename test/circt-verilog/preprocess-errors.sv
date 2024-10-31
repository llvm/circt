// RUN: circt-verilog %s -E --verify-diagnostics
// REQUIRES: slang

// expected-error @below {{'unknown.sv': No such file or directory}}
`include "unknown.sv"
