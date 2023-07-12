// RUN: circt-opt %s -export-verilog -verify-diagnostics --split-input-file | FileCheck %s --strict-whitespace

hw.module @MultiUseExpr(%a: i4) -> (b0: i1) {
  %0 = comb.parity %a : i4
  hw.output %0 : i1
}
// Line 2: module MultiUseExpr(
//   input  [3:0] a,
//   output       b0
// );
// 
//   assign b0 = ^a;
// endmodule

// CHECK:   #loc = loc("":2:0)
// CHECK:   #loc1 = loc("":8:9)
// CHECK:   #loc2 = loc("":7:14)
// CHECK:   #loc3 = loc("":7:16)
// CHECK:   #loc4 = loc("":7:2)
// CHECK:   #loc5 = loc("":7:17)
// CHECK:   #loc6 = loc(fused<"Range">[#loc, #loc1])
// CHECK:   #loc7 = loc(fused<"Range">[#loc2, #loc3])
// CHECK:   #loc8 = loc(fused<"Range">[#loc4, #loc5])
// CHECK:   hw.module @MultiUseExpr(%a: i4) -> (b0: i1) attributes {verilogLocations = #loc6}
// CHECK:     %0 = comb.parity %a {verilogLocations = #loc7} : i4
// CHECK:     hw.output {verilogLocations = #loc8} %0 : i1

// -----

module attributes {circt.loweringOptions = "locationInfoStyle=none"} {
hw.module @SimpleConstPrintReset(%clock: i1, %reset: i1, %in4: i4) -> () {
  %w = sv.wire : !hw.inout<i4>
  %q = sv.reg : !hw.inout<i4>
  %c1_i4 = hw.constant 1 : i4
  sv.assign %w, %c1_i4 : i4
  sv.always posedge %clock, posedge %reset {
    sv.if %reset {
        sv.passign %q, %c1_i4 : i4
      } else {
        sv.passign %q, %in4 : i4
      }
    }
    hw.output
}
}
// module SimpleConstPrintReset(
//   input       clock,
//               reset,
//   input [3:0] in4
// );
// 
//   wire [3:0] w = 4'h1;
//   reg  [3:0] q;
//   always @(posedge clock or posedge reset) begin
//     if (reset)
//       q <= 4'h1;
//     else
//       q <= in4;
//   end // always @(posedge, posedge)
// endmodule
// CHECK:   #loc = loc("":2:0)
// CHECK:   #loc1 = loc("":16:9)
// CHECK:   #loc2 = loc("":8:2)
// CHECK:   #loc3 = loc("":8:22)
// CHECK:   #loc4 = loc("":9:2)
// CHECK:   #loc5 = loc("":9:15)
// CHECK:   #loc6 = loc("":8:17)
// CHECK:   #loc7 = loc("":8:21)
// CHECK:   #loc8 = loc("":12:11)
// CHECK:   #loc9 = loc("":12:15)
// CHECK:   #loc10 = loc("":12:6)
// CHECK:   #loc11 = loc("":12:16)
// CHECK:   #loc12 = loc("":14:6)
// CHECK:   #loc13 = loc("":14:15)
// CHECK:   #loc14 = loc("":11:4)
// CHECK:   #loc15 = loc("":10:2)
// CHECK:   #loc16 = loc("":15:35)
// CHECK:   #loc17 = loc(fused<"Range">[#loc, #loc1])
// CHECK:   #loc18 = loc(fused<"Range">[#loc2, #loc3])
// CHECK:   #loc19 = loc(fused<"Range">[#loc4, #loc5])
// CHECK:   #loc20 = loc(fused<"Range">[#loc6, #loc7])
// CHECK:   #loc21 = loc(fused<"Range">[#loc8, #loc9])
// CHECK:   #loc22 = loc(fused<"Range">[#loc10, #loc11])
// CHECK:   #loc23 = loc(fused<"Range">[#loc12, #loc13])
// CHECK:   #loc24 = loc(fused<"Range">[#loc14, #loc13])
// CHECK:   #loc25 = loc(fused<"Range">[#loc15, #loc16])
// CHECK:   #loc26 = loc(fused[#loc20, #loc21])
// CHECK:   hw.module @SimpleConstPrintReset(%clock: i1, %reset: i1, %in4: i4) attributes {verilogLocations = #loc17} {
// CHECK:     %w = sv.wire {hw.verilogName = "w", verilogLocations = #loc18} : !hw.inout<i4>
// CHECK:     %q = sv.reg {hw.verilogName = "q", verilogLocations = #loc19} : !hw.inout<i4>
// CHECK:     %c1_i4 = hw.constant 1 : i4 {verilogLocations = #loc26}
// CHECK:     sv.assign %w, %c1_i4 : i4
// CHECK:     sv.always posedge %clock, posedge %reset {
// CHECK:       sv.if %reset {
// CHECK:         sv.passign %q, %c1_i4 {verilogLocations = #loc22} : i4
// CHECK:       } else {
// CHECK:         sv.passign %q, %in4 {verilogLocations = #loc23} : i4
// CHECK:       } {verilogLocations = #loc24}
// CHECK:     } {verilogLocations = #loc25}
// CHECK:     hw.output
// CHECK:   }

// -----

hw.module @InlineDeclAssignment(%a: i1) {
  %b = sv.wire : !hw.inout<i1>
  sv.assign %b, %a : i1

  %0 = comb.add %a, %a : i1
  %c = sv.wire : !hw.inout<i1>
  sv.assign %c, %0 : i1
}

// module InlineDeclAssignment(
//   input a
// );
// 
//   wire b = a;
//   wire c = a + a;
// endmodule
// 
// // CHECK:  #loc = loc("":2:0)
// // CHECK:  #loc1 = loc("":8:9)
// // CHECK:  #loc2 = loc("":6:2)
// // CHECK:  #loc3 = loc("":6:13)
// // CHECK:  #loc4 = loc("":7:11)
// // CHECK:  #loc5 = loc("":7:16)
// // CHECK:  #loc6 = loc("":7:2)
// // CHECK:  #loc7 = loc("":7:17)
// // CHECK:  #loc8 = loc(fused<"Range">[#loc, #loc1])
// // CHECK:  #loc9 = loc(fused<"Range">[#loc2, #loc3])
// // CHECK:  #loc10 = loc(fused<"Range">[#loc4, #loc5])
// // CHECK:  #loc11 = loc(fused<"Range">[#loc6, #loc7])
// // CHECK:    hw.module @InlineDeclAssignment(%a: i1) attributes {verilogLocations = #loc8} {
// // CHECK:      %b = sv.wire {hw.verilogName = "b", verilogLocations = #loc9} : !hw.inout<i1>
// // CHECK:      sv.assign %b, %a : i1
// // CHECK:      %0 = comb.add %a, %a {verilogLocations = #loc10} : i1
// // CHECK:      %c = sv.wire {hw.verilogName = "c", verilogLocations = #loc11} : !hw.inout<i1>

// -----


hw.module.extern @MyExtModule()
hw.module.extern @AParameterizedExtModule<CFG: none>()

// CHECK:  #loc = loc("":2:0)
// CHECK:  #loc1 = loc("":3:0)
// CHECK:  #loc2 = loc("":4:0)
// CHECK:  #loc3 = loc("":5:0)
// CHECK:  #loc4 = loc(fused<"Range">[#loc, #loc1])
// CHECK:  #loc5 = loc(fused<"Range">[#loc2, #loc3])
// CHECK:    hw.module.extern @MyExtModule() attributes {verilogLocations = #loc4}
// CHECK:    hw.module.extern @AParameterizedExtModule<CFG: none>() attributes {verilogLocations = #loc5}

