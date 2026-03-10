// RUN: circt-opt -lower-firrtl-to-hw %s -verify-diagnostics | FileCheck %s

firrtl.circuit "SVVerbatimTest" {

  //////////////////////////////////////////////////////////////////////////////
  // Simplest case; a single FIRRTL extmodule should lower to a
  // sv.verbatim.module + sv.verbatim.source.

  // CHECK-LABEL: sv.verbatim.source @SimpleVerbatim.v
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      content = "module SimpleVerbatim(); endmodule"
  // CHECK-SAME:      output_file = #hw.output_file<"SimpleVerbatim.v">
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.source
  //
  // CHECK:       sv.verbatim.module @SimpleVerbatim
  // CHECK-SAME:    (in %clk : !seq.clock, in %rst : i1, out out : i1)
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      source = @SimpleVerbatim.v
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.module
  // CHECK-NOT:   hw.module.extern
  firrtl.extmodule @SimpleVerbatim(
    in clk: !firrtl.clock,
    in rst: !firrtl.uint<1>,
    out out: !firrtl.uint<1>
  ) attributes {
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            content = "module SimpleVerbatim(); endmodule",
            output_file = "SimpleVerbatim.v"
          }
        ]
      }
    ]
  }

  //////////////////////////////////////////////////////////////////////////////
  // Duplicated extmodule definitions to the same output file. This can
  // happen when generators produce definitions of an inline black box
  // under each of multiple deduping scopes.

  // CHECK-LABEL: sv.verbatim.source @DuplicatedVerbatim.v
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      content = "module DuplicatedVerbatim(); endmodule"
  // CHECK-SAME:      output_file = #hw.output_file<"DuplicatedVerbatim.v">
  // CHECK-SAME:      verilogName = "DuplicatedVerbatim"
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.source
  //
  // CHECK:       sv.verbatim.module @DuplicatedVerbatim
  // CHECK-SAME:    (in %clk : !seq.clock, out out : i1)
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      source = @DuplicatedVerbatim.v
  // CHECK-SAME:      verilogName = "DuplicatedVerbatim"
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.module
  // CHECK-NOT:   hw.module.extern
  firrtl.extmodule @DuplicatedVerbatim(
    in clk: !firrtl.clock,
    out out: !firrtl.uint<1>
  ) attributes {
    defname = "DuplicatedVerbatim",
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            content = "module DuplicatedVerbatim(); endmodule",
            output_file = "DuplicatedVerbatim.v"
          }
        ]
      }
    ]
  }

  // CHECK:       sv.verbatim.module @DuplicatedBlackBox_1
  // CHECK-SAME:    (in %clk : !seq.clock, out out : i1)
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      source = @DuplicatedVerbatim.v
  // CHECK-SAME:      verilogName = "DuplicatedVerbatim"
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.module
  // CHECK-NOT:   hw.module.extern
  firrtl.extmodule @DuplicatedBlackBox_1(
    in clk: !firrtl.clock,
    out out: !firrtl.uint<1>
  ) attributes {
    defname = "DuplicatedVerbatim",
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            content = "module DuplicatedVerbatim(); endmodule",
            output_file = "DuplicatedVerbatim.v"
          }
        ]
      }
    ]
  }

  //////////////////////////////////////////////////////////////////////////////
  // Two unique parametrizations of a verbatim extmodule should share an
  // sv.verbatim.source, but each have sv.verbatim.module with the actual
  // port interfaces.

  // CHECK-LABEL: sv.verbatim.source @ParameterizedVerbatim.v
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      content = "module ParameterizedVerbatim #(parameter int WIDTH = 8)(); endmodule"
  // CHECK-SAME:      output_file = #hw.output_file<"ParameterizedVerbatim.v">
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.source
  //
  // CHECK:       sv.verbatim.module @ParameterizedVerbatim
  // CHECK-SAME:    <WIDTH: i32>
  // CHECK-SAME:    (in %data_in : i8, out data_out : i8)
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      source = @ParameterizedVerbatim.v
  // CHECK-SAME:    }
  firrtl.extmodule @ParameterizedVerbatim<WIDTH: i32 = 8>(
    in data_in: !firrtl.uint<8>,
    out data_out: !firrtl.uint<8>
  ) attributes {
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            content = "module ParameterizedVerbatim #(parameter int WIDTH = 8)(); endmodule",
            output_file = "ParameterizedVerbatim.v"
          }
        ]
      }
    ]
  }

  // CHECK:       sv.verbatim.module @ParameterizedVerbatim_1
  // CHECK-SAME:    <WIDTH: i32>
  // CHECK-SAME:    (in %data_in : i16, out data_out : i16)
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      source = @ParameterizedVerbatim.v
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.module
  // CHECK-NOT:   hw.module.extern
  firrtl.extmodule @ParameterizedVerbatim_1<WIDTH: i32 = 16>(
    in data_in: !firrtl.uint<16>,
    out data_out: !firrtl.uint<16>
  ) attributes {
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            content = "module ParameterizedVerbatim #(parameter int WIDTH = 8)(); endmodule",
            output_file = "ParameterizedVerbatim.v"
          }
        ]
      }
    ]
  }

  //////////////////////////////////////////////////////////////////////////////
  // Verbatim extmodule with multiple files should reference those files through
  // the "additional_files" attribute.

  // CHECK-LABEL: emit.file "MultiFileVerbatimChild.v" sym @MultiFileVerbatimChild.v
  // CHECK-NOT:   emit.file
  //
  // CHECK:       sv.verbatim.source @MultiFileVerbatim.v
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      additional_files = [@MultiFileVerbatimChild.v]
  // CHECK-SAME:      content = "module MultiFileVerbatim(); Child child(); endmodule"
  // CHECK-SAME:      output_file = #hw.output_file<"MultiFileVerbatim.v">
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.source
  //
  // CHECK:       sv.verbatim.module @MultiFileVerbatim
  // CHECK-SAME:    (out out : i1)
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      source = @MultiFileVerbatim.v
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.module
  // CHECK-NOT:   hw.module.extern
  firrtl.extmodule @MultiFileVerbatim(
    out out: !firrtl.uint<1>
  ) attributes {
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            content = "module MultiFileVerbatim(); Child child(); endmodule",
            output_file = "MultiFileVerbatim.v"
          },
          {
            content = "module Child(); endmodule",
            output_file = "MultiFileVerbatimChild.v"
          }
        ]
      }
    ]
  }

  //////////////////////////////////////////////////////////////////////////////
  // Check that two independent modules depending on the same auxillary file
  // results in two separate sv.verbatim.source ops but only one shared emit.file.

  // CHECK-LABEL: emit.file "Lib.v" sym @Lib.v
  // CHECK-NOT:   emit.file
  //
  // CHECK:       sv.verbatim.source @NeedsLibFoo.v
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      additional_files = [@Lib.v]
  // CHECK-SAME:      content = "module NeedsLibFoo(); endmodule"
  // CHECK-SAME:      output_file = #hw.output_file<"NeedsLibFoo.v">
  // CHECK-SAME:    }
  //
  // CHECK:       sv.verbatim.module @NeedsLibFoo
  // CHECK-SAME:    (out out : i1)
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      source = @NeedsLibFoo.v
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.module
  // CHECK-NOT:   hw.module.extern
  firrtl.extmodule @NeedsLibFoo(
    out out: !firrtl.uint<1>
  ) attributes {
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            content = "module NeedsLibFoo(); endmodule",
            output_file = "NeedsLibFoo.v"
          },
          {
            content = "module Lib(); endmodule",
            output_file = "Lib.v"
          }
        ]
      }
    ]
  }

  // CHECK:       sv.verbatim.source @NeedsLibBar.v
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      additional_files = [@Lib.v]
  // CHECK-SAME:      content = "module NeedsLibBar(); endmodule"
  // CHECK-SAME:      output_file = #hw.output_file<"NeedsLibBar.v">
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.source
  //
  // CHECK:       sv.verbatim.module @NeedsLibBar
  // CHECK-SAME:    (out out : i1)
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      source = @NeedsLibBar.v
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.module
  // CHECK-NOT:   hw.module.extern
  firrtl.extmodule @NeedsLibBar(
    out out: !firrtl.uint<1>
  ) attributes {
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            content = "module NeedsLibBar(); endmodule",
            output_file = "NeedsLibBar.v"
          },
          {
            content = "module Lib(); endmodule",
            output_file = "Lib.v"
          }
        ]
      }
    ]
  }

  //////////////////////////////////////////////////////////////////////////////
  // Check that analog ports are handled correctly.

  // CHECK-LABEL: sv.verbatim.source @AnalogBlackBox.v
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      content = "module AnalogBlackBox (); endmodule"
  // CHECK-SAME:      output_file = #hw.output_file<"AnalogBlackBox.v">
  // CHECK-SAME:      verilogName = "AnalogBlackBox"
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.source
  //
  // CHECK:       sv.verbatim.module @AnalogBlackBox(inout %bus : i32)
  // CHECK-SAME:    attributes {
  // CHECK-SAME:      source = @AnalogBlackBox.v
  // CHECK-SAME:      verilogName = "AnalogBlackBox"
  // CHECK-SAME:    }
  // CHECK-NOT:   sv.verbatim.module
  // CHECK-NOT:   hw.module.extern
  firrtl.extmodule @AnalogBlackBox(out bus: !firrtl.analog<32>) attributes {
      annotations = [
        {
          class = "circt.VerbatimBlackBoxAnno",
          files = [
            {
                content = "module AnalogBlackBox (); endmodule",
                output_file = "AnalogBlackBox.v"
            }
          ]
        }
      ],
      convention = #firrtl<convention scalarized>, defname = "AnalogBlackBox"
    }

  firrtl.module @SVVerbatimTest() {}

  //////////////////////////////////////////////////////////////////////////////
  // Check that externalRequirements property is transferred from
  // firrtl.extmodule to sv.verbatim.module during lowering.
  //
  // CHECK:       sv.verbatim.module @VerbatimWithRequirements
  // CHECK-SAME:    circt.external_requirements = ["dpi_lib", "sv_header"]
  firrtl.extmodule @VerbatimWithRequirements(
    in clk: !firrtl.clock
  ) attributes {
    annotations = [
      {
        class = "circt.VerbatimBlackBoxAnno",
        files = [
          {
            content = "module VerbatimWithRequirements(input clk); endmodule",
            output_file = "VerbatimWithRequirements.v"
          }
        ]
      }
    ],
    externalRequirements = ["dpi_lib", "sv_header"]
  }
}
