// RUN: circt-verilog %s -E --verify-diagnostics
// REQUIRES: slang

// expected-error @below {{could not find or open include file}}
`include "unknown.sv"
