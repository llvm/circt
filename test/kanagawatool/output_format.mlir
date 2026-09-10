// XFAIL: *
// See https://github.com/llvm/circt/issues/6658
// RUN: kanagawatool %s --lo --post-kanagawa-ir | FileCheck %s --check-prefix=CHECK-POST-KANAGAWA
// RUN: kanagawatool %s --lo --ir | FileCheck %s --check-prefix=CHECK-IR
// RUN: kanagawatool %s --lo --verilog | FileCheck %s --check-prefix=CHECK-VERILOG

// CHECK-POST-KANAGAWA-LABEL:   hw.module @A_B(
// CHECK-POST-KANAGAWA-SAME:            in %[[VAL_0:.*]] : !seq.clock, in %[[VAL_1:.*]] : i1, out p_out : i1) {
// CHECK-POST-KANAGAWA:           %[[VAL_2:.*]] = seq.compreg %[[VAL_1]], %[[VAL_0]] : i1
// CHECK-POST-KANAGAWA:           hw.output %[[VAL_2]] : i1
// CHECK-POST-KANAGAWA:         }

// CHECK-POST-KANAGAWA-LABEL:   hw.module @A(
// CHECK-POST-KANAGAWA-SAME:           in %[[VAL_0:.*]] : i1, in %[[VAL_1:.*]] : !seq.clock, out out : i1) {
// CHECK-POST-KANAGAWA:           %[[VAL_2:.*]] = hw.instance "A_B" @A_B(p_clk: %[[VAL_1]]: !seq.clock, p_in: %[[VAL_0]]: i1) -> (p_out: i1)
// CHECK-POST-KANAGAWA:           hw.output %[[VAL_2]] : i1
// CHECK-POST-KANAGAWA:         }

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

kanagawa.design @foo {
kanagawa.class sym @A {
  kanagawa.port.input "in" sym @in : i1
  kanagawa.port.output "out" sym @out : i1
  kanagawa.port.input "clk" sym @clk : !seq.clock
  
  kanagawa.container@B {
    %parent = kanagawa.path [
      #kanagawa.step<parent : !kanagawa.scoperef<@foo::@A>>
    ]
    %a_in = kanagawa.get_port %parent, @in : !kanagawa.scoperef<@foo::@A> -> !kanagawa.portref<out i1>
    %a_clk = kanagawa.get_port %parent, @clk : !kanagawa.scoperef<@foo::@A> -> !kanagawa.portref<out !seq.clock>
    %a_out = kanagawa.get_port %parent, @out : !kanagawa.scoperef<@foo::@A> -> !kanagawa.portref<in i1>
    %in = kanagawa.port.read %a_in : !kanagawa.portref<out i1>
    %clk = kanagawa.port.read %a_clk : !kanagawa.portref<out !seq.clock>
    %r = seq.compreg %in, %clk: i1
    kanagawa.port.write %a_out, %r : !kanagawa.portref<in i1>
  }
}
}
