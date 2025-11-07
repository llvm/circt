// RUN: circt-opt -lower-firrtl-to-hw %s -verify-diagnostics | FileCheck %s

// Test that LowerToHW consumes the circt.VerbatimBlackBoxAnno annotation
// on a firrtl.extmodule and lowers to sv.verbatim.module.

firrtl.circuit "VerbatimBlackBoxTest" {

  // CHECK: sv.verbatim.module @SimpleVerbatimBlackBox(
  // CHECK-SAME: content = {{.*}}module SimpleVerbatimBlackBox{{.*}}

  firrtl.extmodule @SimpleVerbatimBlackBox(
    in clk: !firrtl.clock,
    in rst: !firrtl.uint<1>,
    out out: !firrtl.uint<1>
  ) attributes {
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            name = "SimpleVerbatimBlackBox.v",
            content = "module SimpleVerbatimBlackBox(\n  input  clk,\n  input  rst,\n  output out\n);\n  always @(posedge clk) begin\n    if (rst) out <= 1'b0;\n    else out <= 1'b1;\n  end\nendmodule",
            output_file = "simple_blackbox.v"
          }
        ]
      }
    ]
  }

  // CHECK: sv.verbatim.module @ParameterizedVerbatimBlackBox<
  // CHECK-SAME: WIDTH: i32
  // CHECK-SAME: content = {{.*}}module ParameterizedVerbatimBlackBox{{.*}}

  firrtl.extmodule @ParameterizedVerbatimBlackBox<WIDTH: i32 = 8>(
    in data_in: !firrtl.uint<8>,
    out data_out: !firrtl.uint<8>
  ) attributes {
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            name = "ParameterizedVerbatimBlackBox.v",
            content = "module ParameterizedVerbatimBlackBox #(\n  parameter WIDTH = 8\n) (\n  input  [WIDTH-1:0] data_in,\n  output [WIDTH-1:0] data_out\n);\n  assign data_out = data_in;\nendmodule",
            output_file = "param_blackbox.v"
          }
        ]
      }
    ]
  }

  // CHECK: emit.file "header.vh" sym @header.vh {
  // CHECK-NEXT:   emit.verbatim "`define MACRO_VALUE 1'b1"
  // CHECK: sv.verbatim.module @MultiFileVerbatimBlackBox(
  // CHECK-SAME: additional_files = [@header.vh]

  firrtl.extmodule @MultiFileVerbatimBlackBox(
    in clk: !firrtl.clock,
    out out: !firrtl.uint<1>
  ) attributes {
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            name = "MultiFileVerbatimBlackBox.v",
            content = "module MultiFileVerbatimBlackBox(\n  input  clk,\n  output out\n);\n  `include \"header.vh\"\n  always @(posedge clk) out <= MACRO_VALUE;\nendmodule",
            output_file = "multi_blackbox.v"
          },
          {
            name = "header.vh",
            content = "`define MACRO_VALUE 1'b1",
            output_file = "header.vh"
          }
        ]
      }
    ]
  }

  // CHECK: hw.module.extern @RegularExtModule(in %data : i8, out out : i8)

  firrtl.extmodule @RegularExtModule(
    in data: !firrtl.uint<8>,
    out out: !firrtl.uint<8>
  )


  // CHECK: sv.verbatim.module @AnalogBlackBox(inout %bus : i32) attributes {content = "module AnalogBlackBox (inout [31:0] bus); endmodule", output_file = #hw.output_file<"AnalogBlackBox.v">, verilogName = "AnalogBlackBox"}
  firrtl.extmodule @AnalogBlackBox(out bus: !firrtl.analog<32>) attributes {
      annotations = [
        {
          class = "circt.VerbatimBlackBoxAnno",
          files = [
            {
                name = "AnalogBlackBox.v",
                content = "module AnalogBlackBox (inout [31:0] bus); endmodule",
                output_file = "AnalogBlackBox.v"
            }
          ]
        }
      ],
      convention = #firrtl<convention scalarized>, defname = "AnalogBlackBox"
    }

  // CHECK: hw.module @VerbatimBlackBoxTest(
  // CHECK-SAME: in %clk : !seq.clock, in %rst : i1, in %data : i8, out result : i1

  firrtl.module @VerbatimBlackBoxTest(
    in %clk: !firrtl.clock,
    in %rst: !firrtl.uint<1>,
    in %data: !firrtl.uint<8>,
    out %result: !firrtl.uint<1>
  ) {
    // CHECK: hw.instance "simple" @SimpleVerbatimBlackBox
    %simple_clk, %simple_rst, %simple_out = firrtl.instance simple @SimpleVerbatimBlackBox(
      in clk: !firrtl.clock,
      in rst: !firrtl.uint<1>,
      out out: !firrtl.uint<1>
    )
    firrtl.connect %simple_clk, %clk : !firrtl.clock, !firrtl.clock
    firrtl.connect %simple_rst, %rst : !firrtl.uint<1>, !firrtl.uint<1>

    // Simple test without parameters for now
    %param_out = firrtl.constant 0 : !firrtl.uint<8>

    // CHECK: hw.instance "multi" @MultiFileVerbatimBlackBox
    %multi_clk, %multi_out = firrtl.instance multi @MultiFileVerbatimBlackBox(
      in clk: !firrtl.clock,
      out out: !firrtl.uint<1>
    )
    firrtl.connect %multi_clk, %clk : !firrtl.clock, !firrtl.clock

    // CHECK: hw.instance "regular" @RegularExtModule
    %regular_in, %regular_out = firrtl.instance regular @RegularExtModule(
      in data: !firrtl.uint<8>,
      out out: !firrtl.uint<8>
    )
    firrtl.connect %regular_in, %data : !firrtl.uint<8>, !firrtl.uint<8>

    // Combine outputs
    %param_bit = firrtl.bits %param_out 0 to 0 : (!firrtl.uint<8>) -> !firrtl.uint<1>
    %combined1 = firrtl.and %simple_out, %param_bit : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %combined2 = firrtl.and %combined1, %multi_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %result, %combined2 : !firrtl.uint<1>, !firrtl.uint<1>

    %analog_bus = firrtl.instance analog @AnalogBlackBox(out bus: !firrtl.analog<32>)
  }
}
