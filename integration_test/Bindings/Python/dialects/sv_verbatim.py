# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import hw, sv, emit
from circt import ir

with ir.Context() as ctx, ir.Location.unknown():
  circt.register_dialects(ctx)

  i1 = ir.IntegerType.get_signless(1)
  i8 = ir.IntegerType.get_signless(8)
  i32 = ir.IntegerType.get_signless(32)

  m = ir.Module.create()
  with ir.InsertionPoint(m.body):
    verbatim_content = "module MyVerbatim(input clk, output reg out); always @(posedge clk) out <= ~out; endmodule"
    verbatim_source = sv.SVVerbatimSourceOp(
        sym_name="MyVerbatim.v",
        content=verbatim_content,
        output_file=hw.OutputFileAttr.get_from_filename(
            ir.StringAttr.get("MyVerbatim.v"), False, False),
        verilog_name="MyVerbatim"
    )

    # CHECK:      sv.verbatim.source @MyVerbatim.v
    # CHECK-SAME:   attributes {
    # CHECK-SAME:     content = "module MyVerbatim(input clk, output reg out); always @(posedge clk) out <= ~out; endmodule",
    # CHECK-SAME:     output_file = #hw.output_file<"MyVerbatim.v">,
    # CHECK-SAME:     verilogName = "MyVerbatim"
    # CHECK-SAME:   }
    print(verbatim_source)

    verbatim_module = sv.SVVerbatimModuleOp(
        name="MyVerbatim",
        source=ir.FlatSymbolRefAttr.get("MyVerbatim.v"),
        input_ports=[("clk", i1)],
        output_ports=[("out", i1)],
        verilog_name="MyVerbatim"
    )

    # CHECK:      sv.verbatim.module @MyVerbatim
    # CHECK-SAME:   (in %clk : i1, out out : i1)
    # CHECK-SAME    attributes {
    # CHECK-SAME:     source = @MyVerbatim.v,
    # CHECK-SAME:     verilogName = "MyVerbatim"
    # CHECK-SAME:   }
    print(verbatim_module)

    param = hw.ParamDeclAttr.get("WIDTH", i32, ir.IntegerAttr.get(i32, 8))
    parameters = ir.ArrayAttr.get([param])

    param_verbatim_source = sv.SVVerbatimSourceOp(
        sym_name="ParametrizedVerbatim.v",
        content="module ParametrizedVerbatim #(parameter int WIDTH = 8)(); endmodule",
        output_file=hw.OutputFileAttr.get_from_filename(
            ir.StringAttr.get("ParametrizedVerbatim.v"), False, False),
        parameters=parameters,
        verilog_name="ParametrizedVerbatim"
    )

    # CHECK:      sv.verbatim.source @ParametrizedVerbatim.v
    # CHECK-SAME:   <WIDTH: i32 = 8>
    # CHECK-SAME:   attributes {
    # CHECK-SAME:     content = "module ParametrizedVerbatim #(parameter int WIDTH = 8)(); endmodule",
    # CHECK-SAME:     output_file = #hw.output_file<"ParametrizedVerbatim.v">,
    # CHECK-SAME:     verilogName = "ParametrizedVerbatim"
    # CHECK-SAME:   }
    print(param_verbatim_source)

    # Test parametrized verbatim module
    param_verbatim_module = sv.SVVerbatimModuleOp(
        name="ParametrizedVerbatim",
        source=ir.FlatSymbolRefAttr.get("ParametrizedVerbatim.v"),
        input_ports=[("data_in", i8)],
        output_ports=[("data_out", i8)],
        parameters=[param],
        verilog_name="ParametrizedVerbatim"
    )

    # CHECK:      sv.verbatim.module @ParametrizedVerbatim
    # CHECK-SAME:   <WIDTH: i32 = 8>
    # CHECK-SAME    (in %data_in : i8, out data_out : i8)
    # CHECK-SAME:   attributes {
    # CHECK-SAME:     source = @ParametrizedVerbatim.v,
    # CHECK-SAME:     verilogName = "ParametrizedVerbatim"
    # CHECK-SAME:   }
    print(param_verbatim_module)

  print("=== SV Verbatim Operations Test Completed ===")
  # CHECK: === SV Verbatim Operations Test Completed ===
