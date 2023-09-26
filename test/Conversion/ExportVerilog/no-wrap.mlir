// RUN: circt-opt -export-verilog --split-input-file %s | FileCheck %s
// RUN: circt-opt -test-apply-lowering-options='options=emittedLineLength=10' -export-verilog --split-input-file %s | FileCheck %s

// https://github.com/llvm/circt/issues/4181

// CHECK{LITERAL}: //  {{
// CHECK-NEXT: endmodule
hw.module @VerbatimWrapping(input %clock : i1, input %cond : i1, input %val : i8, input %a : i3, input %b : i3) {
      %x = comb.add %a, %b : i3
      %y = comb.xor %a, %b : i3
      %arr = hw.array_create %x, %y, %x, %y, %x, %y, %x, %y, %x, %y, %x, %y : i3
      sv.verbatim "// {{0}}" (%arr) : !hw.array<12xi3>
}

// -----
// https://github.com/llvm/circt/issues/4182

// CHECK-LABEL: TestZero(

// CHECK:      input  /*Zero Width*/      zeroBitWithAVeryLongNameWhichMightSeemUnlikelyButHappensAllTheTime
// CHECK-NEXT: input  [2:0]/*Zero Width*/ arrZero

// CHECK:      // Zero width: assign rZeroOutputWithAVeryLongName_YepThisToo_LongNamesAreTheWay_MoreText_GoGoGoGoGo
// CHECK-SAME: ;
// CHECK-NEXT: // Zero width: assign
// CHECK-SAME: ;
// CHECK-NEXT: endmodule
hw.module @TestZero(input %a: i4, input %zeroBitWithAVeryLongNameWhichMightSeemUnlikelyButHappensAllTheTime: i0, input %arrZero: !hw.array<3xi0>,
  output %r0: i4, output %rZeroOutputWithAVeryLongName_YepThisToo_LongNamesAreTheWay_MoreText_GoGoGoGoGo: i0, output %arrZero_0: !hw.array<3xi0>) {

  %b = comb.add %a, %a : i4
  %c = comb.add %zeroBitWithAVeryLongNameWhichMightSeemUnlikelyButHappensAllTheTime, %zeroBitWithAVeryLongNameWhichMightSeemUnlikelyButHappensAllTheTime : i0
  hw.output %b, %c, %arrZero : i4, i0, !hw.array<3xi0>
}

// Module ports:
// CHECK-LABEL: TestZeroInstance(
// CHECK:      // input  /*Zero Width*/      azeroBit
// CHECK-NEXT: // input  [2:0]/*Zero Width*/ aarrZero
// CHECK:      // output /*Zero Width*/      rZeroOutputWithAVeryLongNameYepThisToo

// Wire:
// CHECK: // Zero width: wire /*Zero Width*/ [[ZERO_WIRE:.+]];

// Instance ports:
// CHECK: //.zeroBitWithAVeryLongNameWhichMightSeemUnlikelyButHappensAllTheTime (azeroBit),
// CHECK: //.rZeroOutputWithAVeryLongName_YepThisToo_LongNamesAreTheWay_MoreText_GoGoGoGoGo ([[ZERO_WIRE]])

// Output:
// CHECK: // Zero width: assign
// CHECK-SAME: ;
// CHECK-NEXT: endmodule
hw.module @TestZeroInstance(input %aa: i4, input %azeroBit: i0, input %aarrZero: !hw.array<3xi0>,
  output %r0: i4, output %rZeroOutputWithAVeryLongNameYepThisToo: i0, output %arrZero_0: !hw.array<3xi0>) {


  %o1, %o2, %o3 = hw.instance "iii" @TestZero(a: %aa: i4, zeroBitWithAVeryLongNameWhichMightSeemUnlikelyButHappensAllTheTime: %azeroBit: i0, arrZero: %aarrZero: !hw.array<3xi0>) -> (r0: i4, rZeroOutputWithAVeryLongName_YepThisToo_LongNamesAreTheWay_MoreText_GoGoGoGoGo: i0, arrZero_0: !hw.array<3xi0>)
 %c = comb.add %o2, %o2 : i0

  hw.output %o1, %c, %o3 : i4, i0, !hw.array<3xi0>
}
