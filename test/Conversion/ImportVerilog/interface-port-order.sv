// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Regression test: when a module's port list interleaves an interface modport
// output with a regular output, the operands of the `moore.output` terminator
// must follow the *declaration* order of those outputs. Previously,
// ImportVerilog appended interface-modport outputs after all regular outputs,
// causing a verifier failure when the declaration order was the opposite.
//
// The terminator is rigid (no `materializeConversion` between operand and
// declared port type), so a swap between two same-bit-count but
// different-width outputs trips the verifier.

interface FooBus #(parameter int unsigned W = 1) ();
  logic [W-1:0] data;
  modport Slave (output data);
endinterface

// The wrapper has a non-trivial output ordering: the FooBus.Slave modport
// (output `data`) appears *before* the regular `out_o` output. With distinct
// widths the verifier detects any swap.
//
// CHECK-LABEL: moore.module private @WrapIntf(
// CHECK-SAME:    in %clk_i : !moore.l1
// CHECK-SAME:    out {{(bus[._]data)}} : !moore.l32
// CHECK-SAME:    out out_o : !moore.l64
// CHECK-SAME: ) {
// First operand drives `bus.data` (l32), second drives `out_o` (l64).
// CHECK:         moore.output %{{.+}}, %{{.+}} : !moore.l32, !moore.l64
// CHECK:       }
module WrapIntf (
  input  logic              clk_i,
  FooBus.Slave              bus,
  output logic [63:0]       out_o
);
  // `bus.data` is left undriven; ImportVerilog creates a default variable
  // that flows through the module's output. The point of this test is the
  // declaration-order plumbing, not the modport member's body access.
  assign out_o = '0;
endmodule

// At the instance call site, the result types of `moore.instance` must also
// follow declaration order across regular and interface-modport outputs.
//
// CHECK-LABEL: moore.module @TopWrapIntf
// CHECK:         %{{.*}} = moore.instance "i_w" @WrapIntf(
// CHECK-SAME:      clk_i: %{{.+}}: !moore.l1
// CHECK-SAME:    ) -> ({{(bus[._]data)}}: !moore.l32, out_o: !moore.l64)
module TopWrapIntf;
  logic clk;
  FooBus #(.W(32)) bus ();
  WrapIntf i_w (
    .clk_i(clk),
    .bus(bus),
    .out_o()
  );
endmodule
