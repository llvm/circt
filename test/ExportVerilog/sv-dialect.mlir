// RUN: circt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

// CHECK-LABEL: module M1(
rtl.module @M1(%clock : i1, %cond : i1, %val : i8) {
  // CHECK:      always @(posedge clock) begin
  // CHECK-NEXT:   `ifndef SYNTHESIS
  // CHECK-NEXT:     if (PRINTF_COND_ & cond)
  // CHECK-NEXT:       $fwrite(32'h80000002, "Hi\n");
  // CHECK-NEXT:   `endif
  // CHECK-NEXT: end // always @(posedge)
  sv.alwaysat_posedge %clock {
    sv.ifdef "!SYNTHESIS" {
      %tmp = sv.textual_value "PRINTF_COND_" : i1
      %tmp2 = rtl.and %tmp, %cond : i1
      sv.if %tmp2 {
        sv.fwrite "Hi\n"
      }
    }
  }

  // CHECK-NEXT:   if (cond) begin
  sv.if %cond {
    // CHECK-NEXT:     $fwrite(32'h80000002, "Hi\n");
    sv.fwrite "Hi\n"

    // CHECK-NEXT:     $fwrite(32'h80000002, "Bye %x\n", val + val);
    %tmp = rtl.add %val, %val : i8
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
  
  %wire42 = rtl.wire : !rtl.inout<i42>

  // CHECK-NEXT: initial begin
  sv.initial {
    %thing = sv.textual_value "THING" : i42
    // CHECK-NEXT: wire42 = THING;
    sv.bpassign %wire42, %thing : i42
    // CHECK-NEXT: wire42 <= THING;
    sv.passign %wire42, %thing : i42
  }// CHECK-NEXT:   {{end // initial$}}
}

// CHECK-LABEL: module Aliasing(
// CHECK-NEXT:             inout [41:0] a, b, c);
rtl.module @Aliasing(%a : !rtl.inout<i42>, %b : !rtl.inout<i42>,
                      %c : !rtl.inout<i42>) {

  // CHECK: alias a = b;
  sv.alias %a, %b     : !rtl.inout<i42>, !rtl.inout<i42>
  // CHECK: alias a = b = c;
  sv.alias %a, %b, %c : !rtl.inout<i42>, !rtl.inout<i42>, !rtl.inout<i42>
}


