// RUN: circt-opt -prettify-verilog %s | FileCheck %s
// RUN: circt-opt -prettify-verilog %s | circt-translate  --export-verilog | FileCheck %s --check-prefix=VERILOG

// CHECK-LABEL: hw.module @unary_ops
hw.module @unary_ops(%arg0: i8, %arg1: i8, %arg2: i8) -> (%a: i8, %b: i8) {
  %c-1_i8 = hw.constant -1 : i8

  // CHECK: [[XOR1:%.+]] = comb.xor %arg0
  %unary = comb.xor %arg0, %c-1_i8 : i8
  // CHECK: comb.add [[XOR1]], %arg1
  %a = comb.add %unary, %arg1 : i8

  // CHECK: [[XOR2:%.+]] = comb.xor %arg0
  // CHECK: comb.add [[XOR2]], %arg2
  %b = comb.add %unary, %arg2 : i8
  hw.output %a, %b : i8, i8
}

// VERILOG: assign a = ~arg0 + arg1;
// VERILOG: assign b = ~arg0 + arg2;


/// The pass should sink constants in to the block where they are used.
// CHECK-LABEL: @sink_constants
hw.module @sink_constants(%clock :i1) -> (%out : i1){
  // CHECK: %false = hw.constant false
  %false = hw.constant false

  /// Constants not used should be removed.
  // CHECK-NOT: %true = hw.constant true
  %true = hw.constant true

  /// Simple constant sinking.
  sv.ifdef.procedural "FOO" {
    // CHECK: [[TRUE:%.*]] = hw.constant true
    // CHECK: [[FALSE:%.*]] = hw.constant false
    // CHECK: sv.fwrite "%x"([[TRUE]]) : i1
    sv.fwrite "%x"(%true) : i1
    // CHECK: sv.fwrite "%x"([[FALSE]]) : i1
    sv.fwrite "%x"(%false) : i1
  }

  /// Multiple uses in the same block should use the same constant.
  sv.ifdef.procedural "FOO" {
    // CHECK: [[TRUE:%.*]] = hw.constant true
    // CHECK: sv.fwrite "%x"([[TRUE]]) : i1
    // CHECK: sv.fwrite "%x"([[TRUE]]) : i1
    sv.fwrite "%x"(%true) : i1
    sv.fwrite "%x"(%true) : i1
  }

  // CHECK: hw.output %false : i1
  hw.output %false : i1
}

// VERILOG: `ifdef FOO
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h1);
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h0);
// VERILOG: `endif
// VERILOG: `ifdef FOO
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h1);
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h1);
// VERILOG: `endif



// CHECK-LABEL:   hw.module @AddNegLiteral
// Issue #1324: https://github.com/llvm/circt/issues/1324
hw.module @AddNegLiteral(%a: i8) -> (%x: i8) {

  // CHECK-NEXT: %c4_i8 = hw.constant 4 : i8
  %c = hw.constant -4 : i8
  // CHECK-NEXT: %0 = comb.sub %a, %c4_i8 : i8
  %1 = comb.add %a, %c : i8

  // CHECK-NEXT: hw.output %0
  hw.output %1 : i8
}
// VERILOG-LABEL: module AddNegLiteral(
// VERILOG: assign x = a - 8'h4;
