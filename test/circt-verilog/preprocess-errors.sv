// RUN: circt-translate %s --import-verilog --verify-diagnostics --split-input-file
// REQUIRES: slang
// UNSUPPORTED: valgrind

// expected-error-re @below {{'unknown.sv': {{.+}}}}
`include "unknown.sv"
