// RUN: circt-opt -prettify-verilog %s | FileCheck %s
// RUN: circt-opt -prettify-verilog %s | circt-opt --export-verilog | FileCheck %s --check-prefix=VERILOG

// CHECK-LABEL: hw.module @unary_ops
hw.module @unary_ops(%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i1)
   -> (a: i8, b: i8, c: i1) {
  %c-1_i8 = hw.constant -1 : i8

  // CHECK: [[XOR1:%.+]] = comb.xor %arg0
  %unary = comb.xor %arg0, %c-1_i8 : i8
  // CHECK: %1 = comb.add [[XOR1]], %arg1
  %a = comb.add %unary, %arg1 : i8

  // CHECK: [[XOR2:%.+]] = comb.xor %arg0
  // CHECK: %3 = comb.add [[XOR2]], %arg2
  %b = comb.add %unary, %arg2 : i8


  // Multi-use arith.xori gets duplicated, and we need to make sure there is a local
  // constant as well.
  %true = hw.constant true
  %c = comb.xor %arg3, %true : i1

  // CHECK: [[TRUE1:%.+]] = hw.constant true
  sv.initial {
    // CHECK: [[TRUE2:%.+]] = hw.constant true
    // CHECK: [[XOR3:%.+]] = comb.xor %arg3, [[TRUE2]]
    // CHECK: sv.if [[XOR3]]
    sv.if %c {
      sv.fatal 1
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
// VERILOG-LABEL: sink_constants
hw.module @sink_constants(%clock :i1) -> (out : i1){
  // CHECK: %false = hw.constant false
  %false = hw.constant false

  /// Constants not used should be removed.
  // CHECK-NOT: %true = hw.constant true
  %true = hw.constant true

  /// Simple constant sinking.
  sv.ifdef "FOO" {
    sv.initial {
      // CHECK: [[FALSE:%.*]] = hw.constant false
      // CHECK: [[TRUE:%.*]] = hw.constant true
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
// VERILOG-LABEL: sinkReadInOut
hw.module @sinkReadInOut(%clk: i1) {
  %myreg = sv.reg  : !hw.inout<array<1xstruct<a:i48>>>
  %false = hw.constant false
  %0 = sv.array_index_inout %myreg[%false]: !hw.inout<array<1xstruct<a: i48>>>, i1
  %1 = sv.struct_field_inout %0["a"]: !hw.inout<struct<a: i48>>
  %2 = sv.read_inout %1 : !hw.inout<i48>
  sv.alwaysff(posedge %clk)  {
    sv.passign %1, %2 : i48
  }
}
// CHECK:  %myreg = sv.reg
// CHECK:  sv.alwaysff(posedge %clk)
// CHECK:    sv.array_index_inout
// CHECK:    sv.struct_field_inout
// CHECK:    sv.read_inout

// VERILOG:  struct packed {logic [47:0] a; }[0:0] myreg;
// VERILOG:  always_ff @(posedge clk)
// VERILOG:    myreg[1'h0].a <= myreg[1'h0].a;


// CHECK-LABEL: @sink_expression
// VERILOG-LABEL: sink_expression
hw.module @sink_expression(%clock: i1, %a: i1, %a2: i1, %a3: i1, %a4: i1) {
  // This or is used in one place.
  %0 = comb.or %a2, %a3 : i1
  // This and/xor chain is used in two.  Both should be sunk.
  %1 = comb.and %a2, %a3 : i1
  %2 = comb.xor %1, %a4 : i1
  // CHECK: sv.always
  sv.always posedge %clock  {
    // CHECK: [[AND:%.*]] = comb.and %a2, %a3 : i1
    // CHECK: [[XOR:%.*]] = comb.xor [[AND]], %a4 : i1

    // CHECK: sv.ifdef.procedural
    sv.ifdef.procedural "SOMETHING"  {
      // CHECK: [[OR:%.*]] = comb.or %a2, %a3 : i1
      // CHECK: sv.if [[OR]]
      sv.if %0  {
        sv.fatal 1
      }
      // CHECK: sv.if [[XOR]]
      sv.if %2  {
        sv.fatal 1
      }
    }

    // CHECK: sv.if [[XOR]]
    sv.if %2  {
      sv.fatal 1
    }
  }
  hw.output
}

// CHECK-LABEL: @dont_sink_se_expression
hw.module @dont_sink_se_expression(%clock: i1, %a: i1, %a2: i1, %a3: i1, %a4: i1) {

  // CHECK: [[DONT_TOUCH:%.*]] = sv.verbatim.expr.se "DONT_TOUCH"
  %0 = sv.verbatim.expr "SINK_ME" : () -> i1
  %1 = sv.verbatim.expr.se "DONT_TOUCH" : () -> i1

  // CHECK: sv.always
  sv.always posedge %clock  {
    // CHECK: [[SINK:%.*]] = sv.verbatim.expr "SINK_ME"
    // CHECK: sv.if [[SINK]]
    sv.if %0  {
      sv.fatal 1
    }

    // CHECK: sv.if [[DONT_TOUCH]]
    sv.if %1  {
      sv.fatal 1
    }
  }
  hw.output
}

hw.module.extern @MyExtModule(%in: i8)

// CHECK-LABEL: hw.module @MoveInstances
// VERILOG-LABEL: module MoveInstances
hw.module @MoveInstances(%a_in: i8) {
  // CHECK: %0 = comb.add %a_in, %a_in : i8
  // CHECK: hw.instance "xyz3" @MyExtModule(in: %0: i8)
  // VERILOG: MyExtModule xyz3 (
  // VERILOG:   .in (a_in + a_in)
  // VERILOG: );
  hw.instance "xyz3" @MyExtModule(in: %b: i8) -> ()

  %b = comb.add %a_in, %a_in : i8
}


// CHECK-LABEL: hw.module @unary_sink_crash
hw.module @unary_sink_crash(%arg0: i1) {
  %true = hw.constant true
  %c = comb.xor %arg0, %true : i1
  // CHECK-NOT: hw.constant
  // CHECK-NOT: comb.xor
  // CHECK: sv.initial
  sv.initial {
    // CHECK: [[TRUE1:%.+]] = hw.constant true
    // CHECK: [[XOR1:%.+]] = comb.xor %arg0, [[TRUE1]]
    // CHECK: sv.if [[XOR1]]
    sv.if %c {
      sv.fatal 1
    }

    // CHECK: [[TRUE2:%.+]] = hw.constant true
    // CHECK: [[XOR2:%.+]] = comb.xor %arg0, [[TRUE2]]
    // CHECK: sv.if [[XOR2]]
    sv.if %c {
      sv.fatal 1
    }
  }
}


// CHECK-LABEL: hw.module @unary_sink_no_duplicate
// https://github.com/llvm/circt/issues/2097
hw.module @unary_sink_no_duplicate(%arg0: i4) -> (result: i4) {
  %ones = hw.constant 15: i4

  // CHECK-NOT: comb.xor
 
  // We normally duplicate unary operations like this one so they can be inlined
  // into the using expressions.  However, not all users can be inlined *into*.
  // Things like extract/sext do not support this, so do not duplicate if used
  // by one of them.

  // CHECK: comb.xor %arg0,
  %0 = comb.xor %arg0, %ones : i4
 
 // CHECK-NOT: comb.xor
 
  %a = comb.extract %0 from 0 : (i4) -> i1
  %b = comb.extract %0 from 1 : (i4) -> i1
  %c = comb.extract %0 from 2 : (i4) -> i2


  // CHECK: hw.output
  %r = comb.concat %a, %b, %c : i1, i1, i2
  hw.output %r : i4
}


