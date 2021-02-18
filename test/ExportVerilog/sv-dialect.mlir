// RUN: circt-translate %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

// CHECK-LABEL: module M1(
rtl.module @M1(%clock : i1, %cond : i1, %val : i8) {
  %wire42 = sv.wire : !rtl.inout<i42>

  // CHECK:      always @(posedge clock) begin
  // CHECK-NEXT:   `ifndef SYNTHESIS
  // CHECK-NEXT:     if (PRINTF_COND_ & cond)
  // CHECK-NEXT:       $fwrite(32'h80000002, "Hi\n");
  // CHECK-NEXT:   `endif
  // CHECK-NEXT: end // always @(posedge)
  sv.always posedge %clock {
    sv.ifdef "!SYNTHESIS" {
      %tmp = sv.textual_value "PRINTF_COND_" : i1
      %tmp2 = comb.and %tmp, %cond : i1
      sv.if %tmp2 {
        sv.fwrite "Hi\n"
      }
    }
  }

  // CHECK-NEXT: always @(negedge clock) begin
  // CHECK-NEXT: end // always @(negedge)
  sv.always negedge %clock {
  }

  // CHECK-NEXT: always @(edge clock) begin
  // CHECK-NEXT: end // always @(edge)
  sv.always edge %clock {
  }

  // CHECK-NEXT: always @* begin
  // CHECK-NEXT: end // always
  sv.always {
  }

  // CHECK-NEXT: always @(posedge clock or negedge cond) begin
  // CHECK-NEXT: end // always @(posedge, negedge)
  sv.always posedge %clock, negedge %cond {
  }

  // CHECK-NEXT: always_ff @(posedge clock)
  // CHECK-NEXT:   $fwrite(32'h80000002, "Yo\n");
  sv.alwaysff(posedge %clock) {
    sv.fwrite "Yo\n"
  }
  
  // CHECK-NEXT: always_ff @(posedge clock) begin
  // CHECK-NEXT:   if (cond)
  // CHECK-NEXT:     $fwrite(32'h80000002, "Sync Reset Block\n")
  // CHECK-NEXT:   else
  // CHECK-NEXT:     $fwrite(32'h80000002, "Sync Main Block\n");
  // CHECK-NEXT: end // always_ff @(posedge)
  sv.alwaysff(posedge %clock) {
    sv.fwrite "Sync Main Block\n"
  } ( syncreset : posedge %cond) {
    sv.fwrite "Sync Reset Block\n"
  }

  // CHECK-NEXT: always_ff @(posedge clock or negedge cond) begin
  // CHECK-NEXT:   if (!cond)
  // CHECK-NEXT:     $fwrite(32'h80000002, "Async Reset Block\n");
  // CHECK-NEXT:   else
  // CHECK-NEXT:     $fwrite(32'h80000002, "Async Main Block\n");
  // CHECK-NEXT: end // always_ff @(posedge or negedge)
  sv.alwaysff(posedge %clock) {
    sv.fwrite "Async Main Block\n"
  } ( asyncreset : negedge %cond) {
    sv.fwrite "Async Reset Block\n"
  } 

  %c42 = comb.constant (42 : i42) : i42

  // CHECK-NEXT:   if (cond)
  sv.if %cond {
    // CHECK-NEXT: wire42 = 42'h2A;
    sv.bpassign %wire42, %c42 : i42
  }

  // CHECK-NEXT:   if (cond) begin
  sv.if %cond {
    // CHECK-NEXT:     $fwrite(32'h80000002, "Hi\n");
    sv.fwrite "Hi\n"

    // CHECK-NEXT:     $fwrite(32'h80000002, "Bye %x\n", val + val);
    %tmp = comb.add %val, %val : i8
    sv.fwrite "Bye %x\n"(%tmp) : i8

    // CHECK-NEXT:     assert(cond);
    sv.assert %cond : i1
    // CHECK-NEXT:     assume(cond);
    sv.assume %cond : i1
    // CHECK-NEXT:     cover(cond);
    sv.cover %cond : i1

    // CHECK-NEXT:   $fatal
    sv.fatal
    // CHECK-NEXT:   $finish
    sv.finish

    // CHECK-NEXT: Emit some stuff in verilog
    // CHECK-NEXT: Great power and responsibility!
    sv.verbatim "Emit some stuff in verilog\nGreat power and responsibility!"
  }// CHECK-NEXT:   {{end$}}

  // CHECK-NEXT: initial
  // CHECK-NOT: begin
  sv.initial {
    // CHECK-NEXT: $fatal
    sv.fatal
  }
 
  // CHECK-NEXT: initial begin
  sv.initial {
    // CHECK-NEXT: logic [41:0] _T = THING;
    %thing = sv.textual_value "THING" : i42
    // CHECK-NEXT: wire42 = _T;
    sv.bpassign %wire42, %thing : i42

    sv.ifdef "FOO" {
      // CHECK-NEXT: `ifdef FOO
      %c1 = sv.textual_value "\"THING\"" : i1
      // CHECK-NEXT: logic {{.+}} = "THING";
      sv.fwrite "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", {{.+}});
      sv.fwrite "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", {{.+}});
      // CHECK-NEXT: `endif
    }

    // CHECK-NEXT: wire42 <= _T;
    sv.passign %wire42, %thing : i42
  }// CHECK-NEXT:   {{end // initial$}}

  sv.ifdef "!VERILATOR"  {         // CHECK-NEXT: `ifndef VERILATOR
    sv.verbatim "`define Thing1"   // CHECK-NEXT:   `define Thing1
  } else  {                        // CHECK-NEXT: `else
    sv.verbatim "`define Thing2"   // CHECK-NEXT:   `define Thing2
  }                                // CHECK-NEXT: `endif

  %add = comb.add %val, %val : i8

  // CHECK-NEXT: `define STUFF "wire42 (val + val)"
  sv.verbatim "`define STUFF \"{{0}} ({{1}})\"" (%wire42, %add) : !rtl.inout<i42>, i8

  sv.ifdef "FOO" {
    // CHECK-NEXT: `ifdef FOO
    %c1 = sv.textual_value "\"THING\"" : i1
    // CHECK-NEXT: wire {{.+}} = "THING";
    sv.fwrite "%d" (%c1) : i1
    // CHECK-NEXT: fwrite(32'h80000002, "%d", {{.+}});
    sv.fwrite "%d" (%c1) : i1
    // CHECK-NEXT: fwrite(32'h80000002, "%d", {{.+}});
    // CHECK-NEXT: `endif
  }
}

// CHECK-LABEL: module Aliasing(
// CHECK-NEXT:             inout [41:0] a, b, c
rtl.module @Aliasing(%a : !rtl.inout<i42>, %b : !rtl.inout<i42>,
                      %c : !rtl.inout<i42>) {

  // CHECK: alias a = b;
  sv.alias %a, %b     : !rtl.inout<i42>, !rtl.inout<i42>
  // CHECK: alias a = b = c;
  sv.alias %a, %b, %c : !rtl.inout<i42>, !rtl.inout<i42>, !rtl.inout<i42>
}

rtl.module @reg(%in4: i4, %in8: i8) -> (%a: i8, %b: i8) {
  // CHECK-LABEL: module reg(
  // CHECK-NEXT:   input  [3:0] in4,
  // CHECK-NEXT:   input  [7:0] in8,
  // CHECK-NEXT:   output [7:0] a, b);

  // CHECK-EMPTY:
  // CHECK-NEXT: reg [7:0]       myReg;
  %myReg = sv.reg : !rtl.inout<i8>

  // CHECK-NEXT: reg [41:0][7:0] myRegArray1;
  %myRegArray1 = sv.reg : !rtl.inout<array<42 x i8>>

  // CHECK-EMPTY:
  sv.connect %myReg, %in8 : i8        // CHECK-NEXT: assign myReg = in8;

  %subscript1 = sv.array_index_inout %myRegArray1[%in4] : !rtl.inout<array<42 x i8>>, i4
  sv.connect %subscript1, %in8 : i8   // CHECK-NEXT: assign myRegArray1[in4] = in8;

  %regout = sv.read_inout %myReg : !rtl.inout<i8>

  %subscript2 = sv.array_index_inout %myRegArray1[%in4] : !rtl.inout<array<42 x i8>>, i4
  %memout = sv.read_inout %subscript2 : !rtl.inout<i8>

  // CHECK-NEXT: assign a = myReg;
  // CHECK-NEXT: assign b = myRegArray1[in4];
  rtl.output %regout, %memout : i8, i8
}

// CHECK-LABEL: issue508
// https://github.com/llvm/circt/issues/508
rtl.module @issue508(%in1: i1, %in2: i1) {
  // CHECK: wire _T = in1 | in2;
  %clock = comb.or %in1, %in2 : i1 

  // CHECK-NEXT: always @(posedge _T)
  sv.always posedge %clock {
  }
}

// CHECK-LABEL: exprInlineTestIssue439
// https://github.com/llvm/circt/issues/439
rtl.module @exprInlineTestIssue439(%clk: i1) {
  // CHECK: wire [31:0] _T = 32'h0;
  %c = comb.constant (0:i32) : i32

  // CHECK: always @(posedge clk) begin
  sv.always posedge %clk {
    // CHECK: logic [15:0] _T_0 = _T[15:0];
    %e = comb.extract %c from 0 : (i32) -> i16
    %f = comb.add %e, %e : i16
    sv.fwrite "%d"(%f) : i16
    // CHECK: $fwrite(32'h80000002, "%d", _T_0 + _T_0);
    // CHECK: end // always @(posedge)
  }
}

// CHECK-LABEL: module issue439(
// https://github.com/llvm/circt/issues/439
rtl.module @issue439(%in1: i1, %in2: i1) {
  // CHECK: wire _T_0;
  // CHECK: wire _T = in1 | in2;
  %clock = comb.or %in1, %in2 : i1

  // CHECK-NEXT: always @(posedge _T)
  sv.always posedge %clock {
    // CHECK-NEXT: assign _T_0 = in1;
    // CHECK-NEXT: assign _T_0 = in2;
    %merged = comb.merge %in1, %in2 : i1
    // CHECK-NEXT: $fwrite(32'h80000002, "Bye %x\n", _T_0);
    sv.fwrite "Bye %x\n"(%merged) : i1
  }
}
