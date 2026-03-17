// REQUIRES: slang
// UNSUPPORTED: valgrind

// Test that --emit-moore-dpi-call causes DPI calls to go through
// moore.func.dpi.call instead of func.call during ImportVerilog lowering.
//
// RUN: circt-verilog --ir-moore --emit-moore-dpi-call=1 %s \
// RUN:   | FileCheck %s --check-prefix=DPI
// RUN: circt-verilog --ir-moore --emit-moore-dpi-call=0 %s \
// RUN:   | FileCheck %s --check-prefix=DEFAULT

import "DPI-C" function int my_dpi_add(input int a, input int b);
import "DPI-C" function void my_dpi_out(input int a, output int b);
import "DPI-C" function int my_dpi_inout(input int a, inout int b);
import "DPI-C" function void my_dpi_open_array(input byte wd[], output byte rd[]);

// DPI: moore.func.dpi private @my_dpi_add(in %a : !moore.i32, in %b : !moore.i32, out return : !moore.i32 {moore.func.explicitly_returned})
// DPI: moore.func.dpi private @my_dpi_out(in %a : !moore.i32, out b : !moore.i32)
// DPI: moore.func.dpi private @my_dpi_inout(in %a : !moore.i32, inout %b : !moore.i32, out return : !moore.i32 {moore.func.explicitly_returned})
// DPI: moore.func.dpi private @my_dpi_open_array(in %wd : !moore.open_uarray<i8>, out rd : !moore.ref<open_uarray<i8>>)
// DEFAULT: func.func private @my_dpi_add(!moore.i32, !moore.i32) -> !moore.i32

module DpiMooreCallTest(
  input logic clk,
  input int in_a,
  input int in_b,
  output int out_val,
  output int state_val
);
  // DPI: %[[ADD:.*]] = moore.func.dpi.call @my_dpi_add(%{{.*}}, %{{.*}}) : (!moore.i32, !moore.i32) -> !moore.i32
  // DPI: %[[OUT:.*]] = moore.func.dpi.call @my_dpi_out(%{{.*}}) : (!moore.i32) -> !moore.i32
  // DPI: moore.blocking_assign %{{.*}}, %[[OUT]] : i32
  // DPI: %[[INOUT:.*]]:2 = moore.func.dpi.call @my_dpi_inout(%{{.*}}, %{{.*}}) : (!moore.i32, !moore.i32) -> (!moore.i32, !moore.i32)
  // DPI: moore.blocking_assign %{{.*}}, %[[INOUT]]#0 : i32
  // DPI: moore.blocking_assign %{{.*}}, %[[INOUT]]#1 : i32
  // DPI: %[[WD:.*]] = moore.conversion %{{.*}} : !moore.uarray<8 x i8> -> !moore.open_uarray<i8>
  // DPI: %[[RD:.*]] = moore.conversion %{{.*}} : !moore.ref<uarray<8 x i8>> -> !moore.ref<open_uarray<i8>>
  // DPI: moore.func.dpi.call @my_dpi_open_array(%[[WD]], %[[RD]]) : (!moore.open_uarray<i8>, !moore.ref<open_uarray<i8>>) -> ()
  // DEFAULT-NOT: moore.func.dpi.call
  // DEFAULT: func.call @my_dpi_add

  int state;
  byte wd[8];
  byte rd[8];

  always_ff @(posedge clk)
    out_val <= my_dpi_add(in_a, in_b);

  always_comb begin
    my_dpi_out(in_a, state_val);
    out_val = my_dpi_inout(in_b, state);
  end

  always_ff @(posedge clk)
    my_dpi_open_array(wd, rd);

endmodule