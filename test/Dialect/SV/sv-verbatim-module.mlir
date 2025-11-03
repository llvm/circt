// RUN: circt-opt %s | FileCheck %s

emit.file "test_header.vh" sym @test_header {
  emit.verbatim "`define TEST_MACRO 1'b1"
}

// CHECK-LABEL: sv.verbatim.module @VerbatimTestModule<WIDTH: i32 = 8>(in %clk : i1, in %data : !hw.int<#hw.param.decl.ref<"WIDTH">>, inout %bus : i8, out result : i1) attributes {additional_files = [@test_header], content = "module VerbatimTestModule #(\0A  parameter WIDTH = 8\0A) (\0A  input clk,\0A  input [WIDTH-1:0] data,\0A  inout [7:0] bus,\0A  output result\0A);\0A  `include \22test_header.vh\22\0A  assign result = |data & `TEST_MACRO;\0Aendmodule", output_file = #hw.output_file<"VerbatimTestModule.v">, verilogName = "VerbatimTestModule"}
sv.verbatim.module @VerbatimTestModule<WIDTH: i32 = 8>(in %clk : i1, in %data : !hw.int<#hw.param.decl.ref<"WIDTH">>, inout %bus : i8, out result : i1) {
  content = "module VerbatimTestModule #(\n  parameter WIDTH = 8\n) (\n  input clk,\n  input [WIDTH-1:0] data,\n  inout [7:0] bus,\n  output result\n);\n  `include \"test_header.vh\"\n  assign result = |data & `TEST_MACRO;\nendmodule",
  output_file = #hw.output_file<"VerbatimTestModule.v">,
  additional_files = [@test_header],
  verilogName = "VerbatimTestModule"
}
