// RUN: split-file %s %t

// `--parse-only` only parses the input syntax and does not elaborate or map to
// IR. Instantiating an unknown module is only diagnosed during elaboration, so
// parsing alone succeeds.
// RUN: circt-verilog --parse-only %t/unknown.sv

// Constructs that only fail during IR conversion, such as a class with a
// constraint block, are also accepted since that stage never runs.
// RUN: circt-verilog --parse-only %t/constraint.sv

// Genuine syntax errors are still reported and cause a failure.
// RUN: not circt-verilog --parse-only %t/syntax-error.sv 2>&1 | FileCheck %t/syntax-error.sv

// `--import-only` mirrors the old `--parse-only`: it parses and elaborates the
// input and maps it to Moore IR, but runs no lowering passes.
// RUN: circt-verilog --import-only %t/import.sv | FileCheck %t/import.sv

// Elaboration errors still fire in `--import-only` mode.
// RUN: not circt-verilog --import-only %t/unknown.sv 2>&1 | FileCheck %t/unknown.sv --check-prefix=IMPORT

// REQUIRES: slang
// UNSUPPORTED: valgrind

//--- unknown.sv
// IMPORT: error: unknown module 'Unknown'
module Top;
  Unknown u();
endmodule

//--- constraint.sv
class C;
  rand int x;
  constraint c { x > 0; }
endclass

//--- syntax-error.sv
// CHECK: error: expected ';'
module Top
endmodule

//--- import.sv
// CHECK: moore.module @Top
module Top;
endmodule
