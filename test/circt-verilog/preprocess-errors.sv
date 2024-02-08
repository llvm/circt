// RUN: circt-verilog %s -E --verify-diagnostics

// expected-error @below {{could not find or open include file}}
`include "unknown.sv"
