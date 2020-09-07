// RUN: circt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

firrtl.circuit "Circuit" {
  // CHECK-LABEL: module M1(
  firrtl.module @M1(%clock : i1, %cond : i1, %val : i8) {

    sv.alwaysat_posedge %clock {
      sv.ifdef "!SYNTHESIS" {
        %tmp = sv.textual_value "PRINTF_COND_" : i1
        %tmp2 = rtl.and %tmp, %cond : i1
        sv.if %tmp2 {
          sv.fwrite "Hi\n" 
        }
      }
    }

    // CHECK:      always @(posedge clock) begin
    // CHECK-NEXT:   #ifndef SYNTHESIS
    // CHECK-NEXT:     if (PRINTF_COND_ & cond)
    // CHECK-NEXT:       $fwrite(32'h80000002, "Hi\n");
    // CHECK-NEXT:   #endif
    // CHECK-NEXT: end // always @(posedge)
    sv.if %cond {
      sv.fwrite "Hi\n"

      %tmp = rtl.add %val, %val : i8
      sv.fwrite "Bye %x\n"(%tmp) : i8 
    }

    // CHECK-NEXT:   if (cond) begin
    // CHECK-NEXT:     $fwrite(32'h80000002, "Hi\n");
    // CHECK-NEXT:     $fwrite(32'h80000002, "Bye %x\n", val + val);
    // CHECK-NEXT:   end
  }
}


