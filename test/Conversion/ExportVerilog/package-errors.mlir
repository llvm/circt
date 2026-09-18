// RUN: circt-opt %s -export-verilog -verify-diagnostics -o /dev/null

// A package cannot be emitted into more than one file.
// expected-error @+1 {{packages can only be emitted to a single file}}
sv.package @duplicated {}
emit.file "first.sv" { emit.ref @duplicated }
emit.file "second.sv" { emit.ref @duplicated }
