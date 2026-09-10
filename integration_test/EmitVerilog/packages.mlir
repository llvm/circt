// REQUIRES: verilator
// RUN: circt-opt %S/../../test/Conversion/ExportVerilog/packages.mlir -export-verilog -o /dev/null > %t.sv
// RUN: verilator --lint-only --top-module Consumer %t.sv
// RUN: verilator --lint-only --top-module StateMachine %t.sv
// RUN: circt-opt %S/../../test/Conversion/ExportVerilog/packages.mlir -export-split-verilog="dir-name=%t.split" -o /dev/null
// RUN: cd %t.split && verilator --lint-only --top-module Consumer -f filelist.f
// RUN: circt-opt %S/../../test/Conversion/ExportVerilog/package-names.mlir -export-verilog -o /dev/null > %t.names.sv
// RUN: verilator --lint-only --top-module Consumer %t.names.sv
// RUN: circt-opt %S/../../test/Conversion/ExportVerilog/package-files.mlir -export-split-verilog="dir-name=%t.files" -o /dev/null
// RUN: cd %t.files && verilator --lint-only --top-module Consumer -f filelist.f

// Compile the emitted output: textual checks alone cannot catch an invalid
// package scope, a mis-scoped enumeration member or a bad compilation order.
