// REQUIRES: verilator
// RUN: circt-opt %S/../../test/Conversion/ExportVerilog/packages.mlir -export-verilog -o /dev/null > %t.sv
// RUN: verilator --lint-only --top-module Consumer %t.sv
// RUN: circt-opt %S/../../test/Conversion/ExportVerilog/packages.mlir -export-split-verilog="dir-name=%t.split" -o /dev/null
// RUN: cd %t.split && verilator --lint-only --top-module Consumer -f filelist.f
// RUN: circt-opt %S/../../test/Conversion/ExportVerilog/package-files.mlir -export-split-verilog="dir-name=%t.files" -o /dev/null
// RUN: cd %t.files && verilator --lint-only --top-module Grouped -f filelist.f
