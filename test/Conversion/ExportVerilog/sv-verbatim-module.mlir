// RUN: circt-opt -export-verilog %s | FileCheck %s

sv.verbatim.module @SimpleVerbatimModule(in %a : i1, out b : i1) {
  content = "module SimpleVerbatimModule(\n  input  a,\n  output b\n);\n  // Simple verbatim content\n  assign b = a;\nendmodule",
  output_file = #hw.output_file<"simple.v">
}

sv.verbatim.module @ParameterizedVerbatimModule<WIDTH: i32 = 8>(in %data_in : !hw.int<#hw.param.decl.ref<"WIDTH">>, out data_out : !hw.int<#hw.param.decl.ref<"WIDTH">>) {
  content = "module ParameterizedVerbatimModule #(\n  parameter WIDTH = 8\n) (\n  input  [WIDTH-1:0] data_in,\n  output [WIDTH-1:0] data_out\n);\n  // Parameterized verbatim content\n  assign data_out = data_in;\nendmodule",
  output_file = #hw.output_file<"param.v">
}

sv.verbatim.module @NoPortsVerbatimModule() {
  content = "module NoPortsVerbatimModule();\n  // No ports verbatim content\n  initial begin\n    $display(\"Hello from verbatim module\");\n  end\nendmodule",
  output_file = #hw.output_file<"no_ports.v">
}

hw.module @TestInstantiation(in %clk: i1, in %data: i1, out result: i1) {
  %simple_out = hw.instance "simple_inst" @SimpleVerbatimModule(a: %data: i1) -> (b: i1)

  %c0_i7 = hw.constant 0 : i7
  %data_extended = comb.concat %c0_i7, %data : i7, i1
  %param_out = hw.instance "param_inst" @ParameterizedVerbatimModule<WIDTH: i32 = 8>(data_in: %data_extended: i8) -> (data_out: i8)

  %param_bit = comb.extract %param_out from 0 : (i8) -> i1
  %result = comb.and %simple_out, %param_bit : i1
  hw.output %result : i1
}

// CHECK:      module TestInstantiation(
// CHECK-NEXT:   input  clk,
// CHECK-NEXT:          data,
// CHECK-NEXT:   output result
// CHECK-NEXT: );

// CHECK:      SimpleVerbatimModule simple_inst (
// CHECK-NEXT:   .a (data),
// CHECK-NEXT:   .b ({{.*}})
// CHECK-NEXT: );

// CHECK:      ParameterizedVerbatimModule param_inst (
// CHECK-NEXT:   .data_in  ({{.*}}),
// CHECK-NEXT:   .data_out ({{.*}})
// CHECK-NEXT: );

// CHECK:      module SimpleVerbatimModule(
// CHECK-NEXT:   input  a,
// CHECK-NEXT:   output b
// CHECK-NEXT: );
// CHECK-NEXT:   // Simple verbatim content
// CHECK-NEXT:   assign b = a;
// CHECK-NEXT: endmodule

// CHECK:      module ParameterizedVerbatimModule #(
// CHECK-NEXT:   parameter WIDTH = 8
// CHECK-NEXT: ) (
// CHECK-NEXT:   input  [WIDTH-1:0] data_in,
// CHECK-NEXT:   output [WIDTH-1:0] data_out
// CHECK-NEXT: );
// CHECK-NEXT:   // Parameterized verbatim content
// CHECK-NEXT:   assign data_out = data_in;
// CHECK-NEXT: endmodule

// CHECK:      module NoPortsVerbatimModule();
// CHECK-NEXT:   // No ports verbatim content
// CHECK-NEXT:   initial begin
// CHECK-NEXT:     $display("Hello from verbatim module");
// CHECK-NEXT:   end
// CHECK-NEXT: endmodule
