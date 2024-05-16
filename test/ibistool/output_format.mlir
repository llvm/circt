// XFAIL: *
// See https://github.com/llvm/circt/issues/6658
// RUN: ibistool %s --lo --post-ibis-ir | FileCheck %s --check-prefix=CHECK-POST-IBIS
// RUN: ibistool %s --lo --ir | FileCheck %s --check-prefix=CHECK-IR
// RUN: ibistool %s --lo --verilog | FileCheck %s --check-prefix=CHECK-VERILOG

// CHECK-POST-IBIS-LABEL:   hw.module @A_B(
// CHECK-POST-IBIS-SAME:            in %[[VAL_0:.*]] : !seq.clock, in %[[VAL_1:.*]] : i1, out p_out : i1) {
// CHECK-POST-IBIS:           %[[VAL_2:.*]] = seq.compreg %[[VAL_1]], %[[VAL_0]] : i1
// CHECK-POST-IBIS:           hw.output %[[VAL_2]] : i1
// CHECK-POST-IBIS:         }

// CHECK-POST-IBIS-LABEL:   hw.module @A(
// CHECK-POST-IBIS-SAME:           in %[[VAL_0:.*]] : i1, in %[[VAL_1:.*]] : !seq.clock, out out : i1) {
// CHECK-POST-IBIS:           %[[VAL_2:.*]] = hw.instance "A_B" @A_B(p_clk: %[[VAL_1]]: !seq.clock, p_in: %[[VAL_0]]: i1) -> (p_out: i1)
// CHECK-POST-IBIS:           hw.output %[[VAL_2]] : i1
// CHECK-POST-IBIS:         }

// CHECK-IR-LABEL:   hw.module @A_B(
// CHECK-IR-SAME:              in %[[VAL_0:.*]] : i1, in %[[VAL_1:.*]] : i1, out p_out : i1) {
// CHECK-IR:           %[[VAL_2:.*]] = sv.reg : !hw.inout<i1>
// CHECK-IR:           %[[VAL_3:.*]] = sv.read_inout %[[VAL_2]] : !hw.inout<i1>
// CHECK-IR:           sv.alwaysff(posedge %[[VAL_0]]) {
// CHECK-IR:             sv.passign %[[VAL_2]], %[[VAL_1]] : i1
// CHECK-IR:           }
// CHECK-IR:           hw.output %[[VAL_3]] : i1
// CHECK-IR:         }

// CHECK-IR-LABEL:   hw.module @A(
// CHECK-IR-SAME:            in %[[VAL_0:.*]] : i1, in %[[VAL_1:.*]] : i1, out out : i1) {
// CHECK-IR:           %[[VAL_2:.*]] = hw.instance "A_B" @A_B(p_clk: %[[VAL_1]]: i1, p_in: %[[VAL_0]]: i1) -> (p_out: i1)
// CHECK-IR:           hw.output %[[VAL_2]] : i1
// CHECK-IR:         }

// CHECK-VERILOG-LABEL: module A_B(
// CHECK-VERILOG:   input  p_clk,
// CHECK-VERILOG:          p_in,
// CHECK-VERILOG:   output p_out
// CHECK-VERILOG: );
// CHECK-VERILOG:   reg r;
// CHECK-VERILOG:   always_ff @(posedge p_clk)
// CHECK-VERILOG:     r <= p_in;
// CHECK-VERILOG:   assign p_out = r;
// CHECK-VERILOG: endmodule

// CHECK-VERILOG-LABEL: module A(
// CHECK-VERILOG:   input  in,
// CHECK-VERILOG:          clk,
// CHECK-VERILOG:   output out
// CHECK-VERILOG: );
// CHECK-VERILOG:   A_B A_B (
// CHECK-VERILOG:     .p_clk (clk),
// CHECK-VERILOG:     .p_in  (in),
// CHECK-VERILOG:     .p_out (out)
// CHECK-VERILOG:   );
// CHECK-VERILOG: endmodule

ibis.design @foo {
ibis.class @A {
  %this = ibis.this @A
  ibis.port.input "in" sym @in : i1
  ibis.port.output "out" sym @out : i1
  ibis.port.input "clk" sym @clk : !seq.clock
  
  ibis.container@B {
    %B_this = ibis.this @B
    %parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@A>>
    ]
    %a_in = ibis.get_port %parent, @in : !ibis.scoperef<@A> -> !ibis.portref<out i1>
    %a_clk = ibis.get_port %parent, @clk : !ibis.scoperef<@A> -> !ibis.portref<out !seq.clock>
    %a_out = ibis.get_port %parent, @out : !ibis.scoperef<@A> -> !ibis.portref<in i1>
    %in = ibis.port.read %a_in : !ibis.portref<out i1>
    %clk = ibis.port.read %a_clk : !ibis.portref<out !seq.clock>
    %r = seq.compreg %in, %clk: i1
    ibis.port.write %a_out, %r : !ibis.portref<in i1>
  }
}
}
