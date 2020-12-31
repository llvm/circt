// RUN: circt-opt %s | FileCheck %s
// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @test1(%arg0: i1, %arg1: i1) {
func @test1(%arg0: i1, %arg1: i1) {

  // This corresponds to this block of system verilog code:
  //    always @(posedge arg0) begin
  //      `ifndef SYNTHESIS
  //         if (`PRINTF_COND_ && arg1) $fwrite(32'h80000002, "Hi\n");
  //      `endif
  //    end // always @(posedge)

  sv.always posedge  %arg0 {
    sv.ifdef "!SYNTHESIS" {
      %tmp = sv.textual_value "PRINTF_COND_" : i1
      %tmp2 = rtl.and %tmp, %arg1 : i1
      sv.if %tmp2 {
        sv.fwrite "Hi\n" 

        // Test fwrite with operands.
        sv.fwrite "%x"(%tmp2) : i1
      }
    }
  }

  // CHECK-NEXT: sv.always posedge %arg0 {
  // CHECK-NEXT:   sv.ifdef "!SYNTHESIS" {
  // CHECK-NEXT:     %0 = sv.textual_value "PRINTF_COND_" : i1
  // CHECK-NEXT:     %1 = rtl.and %0, %arg1 : i1
  // CHECK-NEXT:     sv.if %1 {
  // CHECK-NEXT:       sv.fwrite "Hi\0A" 
  // CHECK-NEXT:       sv.fwrite "%x"(%1) : i1
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }


// Smoke test generic syntax.
   "sv.if"(%arg0) ( {
      "sv.yield"() : () -> ()
   }) : (i1) -> ()

  // CHECK-NEXT:     sv.if %arg0 {
  // CHECK-NEXT:     }

  // CHECK-NEXT: return
  return
}
