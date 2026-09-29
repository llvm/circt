// RUN: circt-opt %s -export-verilog -verify-diagnostics -o /dev/null

// A package cannot be emitted into more than one file.
// expected-error @+1 {{packages can only be emitted to a single file}}
sv.package @duplicated {}
emit.file "first.sv" { emit.ref @duplicated }
emit.file "second.sv" { emit.ref @duplicated }

// External package names cannot be legalized by renaming them.
// expected-error @+1 {{name "module" is not allowed in Verilog output}}
sv.package @module {} {extern}
