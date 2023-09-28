// RUN: circt-opt %s -export-verilog -verify-diagnostics --mlir-print-debuginfo --split-input-file | FileCheck %s --strict-whitespace

module attributes {circt.loweringOptions = "emitVerilogLocations"} {
hw.module @MultiUseExpr(in %a: i4, out b0: i1) {
  %0 = comb.parity %a : i4
  hw.output %0 : i1
}
}
// Line 2: module MultiUseExpr(
//   input  [3:0] a,
//   output       b0
// );
// 
//   assign b0 = ^a;
// endmodule

// CHECK-LABEL:   hw.module @MultiUseExpr
// CHECK:     %[[v0:.+]] = comb.parity %a : i4 loc(#loc19)
// CHECK:     hw.output %[[v0]] : i1 loc(#loc20)
// CHECK:   } loc(#loc)
// CHECK: #loc = loc("{{.+}}verilog-locations.mlir{{.*}})
// CHECK: #loc1 = loc("{{.+}}verilog-locations.mlir{{.*}})
// CHECK: #loc2 = loc("":2:0)
// CHECK: #loc3 = loc("":8:9)
// CHECK: #loc6 = loc("{{.+}}verilog-locations.mlir{{.*}})
// CHECK: #loc7 = loc("":7:14)
// CHECK: #loc8 = loc("":7:16)
// CHECK: #loc9 = loc("{{.+}}verilog-locations.mlir{{.*}})
// CHECK: #loc10 = loc("":7:2)
// CHECK: #loc11 = loc("":7:17)
// CHECK: #loc12 = loc(fused<"Range">[#loc2, #loc3])
// CHECK: #loc13 = loc(fused<"Range">[#loc7, #loc8])
// CHECK: #loc14 = loc(fused<"Range">[#loc10, #loc11])
// CHECK: #loc15 = loc(fused<"verilogLocations">[#loc12])
// CHECK: #loc16 = loc(fused<"verilogLocations">[#loc13])
// CHECK: #loc17 = loc(fused<"verilogLocations">[#loc14])
// CHECK: #loc18 = loc(fused[#loc1, #loc15])
// CHECK: #loc19 = loc(fused[#loc6, #loc16])
// CHECK: #loc20 = loc(fused[#loc9, #loc17])

// -----

module attributes {circt.loweringOptions = "locationInfoStyle=none,emitVerilogLocations"} {
hw.module @SimpleConstPrintReset(in %clock: i1, in %reset: i1, in %in4: i4) {
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
// CHECK:   hw.module @SimpleConstPrintReset
// CHECK:     %w = sv.wire {hw.verilogName = "w"} : !hw.inout<i4> loc(#loc49)
// CHECK:     %q = sv.reg {hw.verilogName = "q"} : !hw.inout<i4>  loc(#loc50)
// CHECK:     %c1_i4 = hw.constant 1 : i4 loc(#loc51)
// CHECK:     sv.assign %w, %c1_i4 : i4 loc(#loc18)
// CHECK:     sv.always posedge %clock, posedge %reset {
// CHECK:       sv.if %reset {
// CHECK:         sv.passign %q, %c1_i4 : i4 loc(#loc54)
// CHECK:       } else {
// CHECK:         sv.passign %q, %in4 : i4 loc(#loc55)
// CHECK:       } loc(#loc53)
// CHECK:     } loc(#loc52)
// CHECK:     hw.output loc(#loc30)
// CHECK:   } loc(#loc48)
// CHECK: } loc(#loc)
// CHECK: #loc = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc1 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc2 = loc("":2:0)
// CHECK: #loc3 = loc("":16:9)
// CHECK: #loc7 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc8 = loc("":8:2)
// CHECK: #loc9 = loc("":8:22)
// CHECK: #loc10 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc11 = loc("":9:2)
// CHECK: #loc12 = loc("":9:15)
// CHECK: #loc13 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc14 = loc("":8:17)
// CHECK: #loc15 = loc("":8:21)
// CHECK: #loc16 = loc("":12:11)
// CHECK: #loc17 = loc("":12:15)
// CHECK: #loc18 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc19 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc20 = loc("":10:2)
// CHECK: #loc21 = loc("":15:35)
// CHECK: #loc22 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc23 = loc("":11:4)
// CHECK: #loc24 = loc("":14:15)
// CHECK: #loc25 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc26 = loc("":12:6)
// CHECK: #loc27 = loc("":12:16)
// CHECK: #loc28 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc29 = loc("":14:6)
// CHECK: #loc30 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc31 = loc(fused<"Range">[#loc2, #loc3])
// CHECK: #loc32 = loc(fused<"Range">[#loc8, #loc9])
// CHECK: #loc33 = loc(fused<"Range">[#loc11, #loc12])
// CHECK: #loc34 = loc(fused<"Range">[#loc14, #loc15])
// CHECK: #loc35 = loc(fused<"Range">[#loc16, #loc17])
// CHECK: #loc36 = loc(fused<"Range">[#loc20, #loc21])
// CHECK: #loc37 = loc(fused<"Range">[#loc23, #loc24])
// CHECK: #loc38 = loc(fused<"Range">[#loc26, #loc27])
// CHECK: #loc39 = loc(fused<"Range">[#loc29, #loc24])
// CHECK: #loc40 = loc(fused<"verilogLocations">[#loc31])
// CHECK: #loc41 = loc(fused<"verilogLocations">[#loc32])
// CHECK: #loc42 = loc(fused<"verilogLocations">[#loc33])
// CHECK: #loc43 = loc(fused<"verilogLocations">[#loc34, #loc35])
// CHECK: #loc44 = loc(fused<"verilogLocations">[#loc36])
// CHECK: #loc45 = loc(fused<"verilogLocations">[#loc37])
// CHECK: #loc46 = loc(fused<"verilogLocations">[#loc38])
// CHECK: #loc47 = loc(fused<"verilogLocations">[#loc39])
// CHECK: #loc48 = loc(fused[#loc1, #loc40])
// CHECK: #loc49 = loc(fused[#loc7, #loc41])
// CHECK: #loc50 = loc(fused[#loc10, #loc42])
// CHECK: #loc51 = loc(fused[#loc13, #loc43])
// CHECK: #loc52 = loc(fused[#loc19, #loc44])
// CHECK: #loc53 = loc(fused[#loc22, #loc45])
// CHECK: #loc54 = loc(fused[#loc25, #loc46])
// CHECK: #loc55 = loc(fused[#loc28, #loc47])

// -----

module attributes {circt.loweringOptions = "emitVerilogLocations"} {
hw.module @InlineDeclAssignment(in %a: i1) {
  %b = sv.wire : !hw.inout<i1>
  sv.assign %b, %a : i1

  %0 = comb.add %a, %a : i1
  %c = sv.wire : !hw.inout<i1>
  sv.assign %c, %0 : i1
}
}

// module InlineDeclAssignment(
//   input a
// );
// 
//   wire b = a;
//   wire c = a + a;
// endmodule
// 
// CHECK:   hw.module @InlineDeclAssignment
// CHECK:     %b = sv.wire {hw.verilogName = "b"} : !hw.inout<i1> loc(#loc25)
// CHECK:     sv.assign %b, %a : i1 loc(#loc8)
// CHECK:     %[[v0:.+]] = comb.add %a, %a : i1 loc(#loc26)
// CHECK:     %c = sv.wire {hw.verilogName = "c"} : !hw.inout<i1> loc(#loc27)
// CHECK:     sv.assign %c, %[[v0]] : i1 loc(#loc15)
// CHECK:     hw.output loc(#loc1)
// CHECK:   } loc(#loc24)

// CHECK: #loc = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc1 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc2 = loc("":2:0)
// CHECK: #loc3 = loc("":8:9)
// CHECK: #loc5 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc6 = loc("":6:2)
// CHECK: #loc7 = loc("":6:13)
// CHECK: #loc8 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc9 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc10 = loc("":7:11)
// CHECK: #loc11 = loc("":7:16)
// CHECK: #loc12 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc13 = loc("":7:2)
// CHECK: #loc14 = loc("":7:17)
// CHECK: #loc15 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc16 = loc(fused<"Range">[#loc2, #loc3])
// CHECK: #loc17 = loc(fused<"Range">[#loc6, #loc7])
// CHECK: #loc18 = loc(fused<"Range">[#loc10, #loc11])
// CHECK: #loc19 = loc(fused<"Range">[#loc13, #loc14])
// CHECK: #loc20 = loc(fused<"verilogLocations">[#loc16])
// CHECK: #loc21 = loc(fused<"verilogLocations">[#loc17])
// CHECK: #loc22 = loc(fused<"verilogLocations">[#loc18])
// CHECK: #loc23 = loc(fused<"verilogLocations">[#loc19])
// CHECK: #loc24 = loc(fused[#loc1, #loc20])
// CHECK: #loc25 = loc(fused[#loc5, #loc21])
// CHECK: #loc26 = loc(fused[#loc9, #loc22])
// CHECK: #loc27 = loc(fused[#loc12, #loc23])

// -----


module attributes {circt.loweringOptions = "emitVerilogLocations"} {
hw.module.extern @MyExtModule()
hw.module.extern @AParameterizedExtModule<CFG: none>()
}
// CHECK:   hw.module.extern @MyExtModule() loc(#loc11)
// CHECK:   hw.module.extern @AParameterizedExtModule<CFG: none>() loc(#loc12)

// CHECK: #loc = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc1 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc2 = loc("":2:0)
// CHECK: #loc3 = loc("":3:0)
// CHECK: #loc4 = loc("{{.*}}verilog-locations.mlir{{.*}})
// CHECK: #loc5 = loc("":4:0)
// CHECK: #loc6 = loc("":5:0)
// CHECK: #loc7 = loc(fused<"Range">[#loc2, #loc3])
// CHECK: #loc8 = loc(fused<"Range">[#loc5, #loc6])
// CHECK: #loc9 = loc(fused<"verilogLocations">[#loc7])
// CHECK: #loc10 = loc(fused<"verilogLocations">[#loc8])
// CHECK: #loc11 = loc(fused[#loc1, #loc9])
// CHECK: #loc12 = loc(fused[#loc4, #loc10])
