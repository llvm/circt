// RUN: circt-opt -export-verilog %s | FileCheck %s

// CHECK-LABEL: FILE "SimpleVerbatimModule.v"
// CHECK-LABEL: module SimpleVerbatimModule(); endmodule
// CHECK-NOT:   module
sv.verbatim.source @SimpleVerbatimModule.v attributes {
  content = "module SimpleVerbatimModule(); endmodule",
  output_file = #hw.output_file<"SimpleVerbatimModule.v">,
  verilogName = "SimpleVerbatimModule"
}

hw.module.extern @SimpleVerbatimModule(out out : i1) attributes {
  source = @SimpleVerbatimModule.v
}

// CHECK-LABEL: FILE "ParameterizedVerbatimModule.v"
// CHECK-LABEL: module ParameterizedVerbatimModule #(parameter WIDTH = 2)(); endmodule
sv.verbatim.source @ParameterizedVerbatimModule.v<WIDTH: i32> attributes {
  content = "module ParameterizedVerbatimModule #(parameter WIDTH = 2)(); endmodule",
  output_file = #hw.output_file<"ParameterizedVerbatimModule.v">,
  verilogName = "ParameterizedVerbatimModule"
}

sv.verbatim.module @ParameterizedVerbatimModule<WIDTH: i32 = 8>(out out: !hw.int<#hw.param.decl.ref<"WIDTH">>) attributes {
  source = @ParameterizedVerbatimModule.v
}

sv.verbatim.module @ParameterizedVerbatimModule_1<WIDTH: i32 = 32>(out out: !hw.int<#hw.param.decl.ref<"WIDTH">>) attributes {
  source = @ParameterizedVerbatimModule.v
}

// CHECK-NOT: FILE
