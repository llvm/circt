// RUN: circt-opt -prettify-verilog %s | FileCheck %s
// RUN: circt-opt -prettify-verilog %s | circt-translate  --export-verilog | FileCheck %s --check-prefix=VERILOG

// CHECK-LABEL: hw.module @unary_ops
hw.module @unary_ops(%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i1)
   -> (%a: i8, %b: i8, %c: i1) {
  %c-1_i8 = hw.constant -1 : i8

  // CHECK: [[XOR1:%.+]] = comb.xor %arg0
  %unary = comb.xor %arg0, %c-1_i8 : i8
  // CHECK: %1 = comb.add [[XOR1]], %arg1
  %a = comb.add %unary, %arg1 : i8

  // CHECK: [[XOR2:%.+]] = comb.xor %arg0
  // CHECK: %3 = comb.add [[XOR2]], %arg2
  %b = comb.add %unary, %arg2 : i8


  // Multi-use xor gets duplicated, and we need to make sure there is a local
  // constant as well.
  %true = hw.constant true
  %c = comb.xor %arg3, %true : i1

  // CHECK: [[TRUE1:%.+]] = hw.constant true
  sv.initial {
    // CHECK: [[TRUE2:%.+]] = hw.constant true
    // CHECK: [[XOR3:%.+]] = comb.xor %arg3, [[TRUE2]]
    // CHECK: sv.if [[XOR3]]
    sv.if %c {
      sv.fatal
    }
  }

  // CHECK: [[XOR4:%.+]] = comb.xor %arg3, [[TRUE1]]
  // CHECK: hw.output %1, %3, [[XOR4]]
  hw.output %a, %b, %c : i8, i8, i1
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
  sv.ifdef "FOO" {
    sv.initial {
      // CHECK: [[TRUE:%.*]] = hw.constant true
      // CHECK: [[FALSE:%.*]] = hw.constant false
      // CHECK: sv.fwrite "%x"([[TRUE]]) : i1
      sv.fwrite "%x"(%true) : i1
      // CHECK: sv.fwrite "%x"([[FALSE]]) : i1
      sv.fwrite "%x"(%false) : i1
    }
  }

  /// Multiple uses in the same block should use the same constant.
  sv.ifdef "FOO" {
    sv.initial {
      // CHECK: [[TRUE:%.*]] = hw.constant true
      // CHECK: sv.fwrite "%x"([[TRUE]]) : i1
      // CHECK: sv.fwrite "%x"([[TRUE]]) : i1
      sv.fwrite "%x"(%true) : i1
      sv.fwrite "%x"(%true) : i1
    }
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

// Prettify should always sink ReadInOut to its usage.
// CHECK-LABEL: @sinkReadInOut
hw.module @sinkReadInOut(%clk: i1) {
  %myreg = sv.reg  : !hw.inout<i48>
  %1 = sv.read_inout %myreg : !hw.inout<i48>
  sv.alwaysff(posedge %clk)  {
    sv.passign %myreg, %1 : i48
  }
}
// CHECK:  sv.reg  : !hw.inout<i48>
// CHECK:  sv.alwaysff(posedge %clk)
// CHECK:    sv.read_inout

// VERILOG:  reg [47:0] myreg;
// VERILOG:  always @(posedge clk)
// VERILOG:    myreg <= myreg;


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
