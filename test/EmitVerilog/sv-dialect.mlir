// RUN: circt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

firrtl.circuit "M1" {
  // CHECK-LABEL: module M1(
  firrtl.module @M1(%clock : i1, %cond : i1, %val : i8) {

    // CHECK:      always @(posedge clock) begin
    // CHECK-NEXT:   #ifndef SYNTHESIS
    // CHECK-NEXT:     if (PRINTF_COND_ & cond)
    // CHECK-NEXT:       $fwrite(32'h80000002, "Hi\n");
    // CHECK-NEXT:   #endif
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
      // CHECK-NEXT:   {{end$}}
    }
  }
}


